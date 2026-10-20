"""A live recording: two captures (mic + call audio), live captions, then hand-off to the final pass."""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime

import numpy as np

from . import config, storage
from .audio import Capture, raw_to_wav, resolve_source
from .chunker import Chunk, Chunker
from .events import bus
from .whisper import is_hallucination, pool

log = logging.getLogger(__name__)

STREAMS = {"mic": "Me", "call": "Remote"}
MAX_BACKLOG = 12  # live captions only; the final pass re-transcribes everything


class Recording:
    def __init__(self, meta: dict, mic_source: str, call_source: str, use_mic: bool = True):
        self.meta = meta
        self.id = meta["id"]
        self.dir = storage.meeting_dir(self.id)
        self.started = time.time()
        self.last_speech = time.time()
        self._chunks: queue.Queue[tuple[str, Chunk] | None] = queue.Queue()
        self._chunkers = {"mic": Chunker(), "call": Chunker()}
        self.captures: dict[str, Capture] = {}
        sources = {"call": resolve_source(call_source, "call")}
        if use_mic:
            sources["mic"] = resolve_source(mic_source, "mic")
        for stream, src in sources.items():
            self.captures[stream] = Capture(src, self.dir / f"{stream}.raw", self._on_block(stream))
        self._worker = threading.Thread(target=self._transcribe_loop, name="live-transcribe", daemon=True)
        self._ticker = threading.Thread(target=self._tick, name="live-levels", daemon=True)
        self._stopped = threading.Event()

    def _on_block(self, stream: str):
        chunker = self._chunkers[stream]

        def cb(block: np.ndarray) -> None:
            for c in chunker.feed(block):
                self.last_speech = time.time()
                self._chunks.put((stream, c))

        return cb

    def start(self) -> None:
        for c in self.captures.values():
            c.start()
        self._worker.start()
        self._ticker.start()
        # warm the model so the first caption doesn't wait for the load
        threading.Thread(target=self._warm, daemon=True).start()

    def _warm(self) -> None:
        try:
            pool.live().start()
        except Exception as e:
            log.error("whisper-server failed to start: %s", e)
            bus.publish({"type": "error", "meeting": self.id, "message": str(e)})

    def elapsed(self) -> float:
        return time.time() - self.started

    def levels(self) -> dict:
        return {k: round(c.level, 4) for k, c in self.captures.items()}

    def errors(self) -> dict:
        return {k: c.error for k, c in self.captures.items() if c.error}

    def _tick(self) -> None:
        while not self._stopped.wait(0.25):
            bus.publish({
                "type": "levels",
                "meeting": self.id,
                "levels": self.levels(),
                "elapsed": round(self.elapsed(), 1),
                "backlog": self._chunks.qsize(),
                "errors": self.errors(),
            })

    def _transcribe_loop(self) -> None:
        s = config.load_settings()
        prompts = {"mic": "", "call": ""}
        while True:
            item = self._chunks.get()
            if item is None:
                return
            # if we fall behind, skip the oldest chunks: live view must stay current
            while self._chunks.qsize() > MAX_BACKLOG:
                nxt = self._chunks.get()
                if nxt is None:
                    return
                item = nxt
            stream, chunk = item
            try:
                segs, lang = pool.live().transcribe(
                    chunk.samples, language=s.language, vad=False, words=False, prompt=prompts[stream], timeout=120
                )
            except Exception as e:
                log.warning("live transcription failed: %s", e)
                continue
            text = " ".join(seg.text for seg in segs if not is_hallucination(seg, chunk.energy)).strip()
            if not text:
                continue
            prompts[stream] = text[-200:]  # a little context helps names and code-switching
            item = {
                "stream": stream,
                "speaker": STREAMS[stream],
                "start": round(chunk.start, 2),
                "end": round(chunk.start + chunk.duration, 2),
                "text": text,
                "language": lang,
            }
            storage.append_live(self.id, item)
            bus.publish({"type": "caption", "meeting": self.id, **item})

    def stop(self) -> dict:
        """Stop capturing; returns updated meta. The final pass is queued by the caller."""
        self._stopped.set()
        for c in self.captures.values():
            c.stop()
        # don't wait for pending live chunks: the final pass needs the whisper server
        while not self._chunks.empty():
            try:
                self._chunks.get_nowait()
            except queue.Empty:
                break
        self._chunks.put(None)
        self._worker.join(timeout=150)
        finalize_raw(self.id)
        return storage.update_meta(
            self.id,
            status="queued",
            duration=round(self.elapsed(), 1),
            ended_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )


def finalize_raw(mid: str) -> None:
    """Turn mic.raw / call.raw into WAV files (also used to recover after a crash)."""
    d = storage.meeting_dir(mid)
    for stream in STREAMS:
        raw = d / f"{stream}.raw"
        if raw.exists():
            raw_to_wav(raw, d / f"{stream}.wav")
            raw.unlink()


class RecorderState:
    """At most one recording at a time."""

    def __init__(self) -> None:
        self.current: Recording | None = None
        self.lock = threading.Lock()


recorder = RecorderState()
