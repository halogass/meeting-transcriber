"""Final processing after a meeting: full transcription + diarization + merge.

Transcription (whisper.cpp) and diarization (pyannote) run in parallel; on a two-GPU
machine they use different cards.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import config, storage
from .audio import make_playback_audio, read_wav
from .chunker import split_long_audio
from .config import SAMPLE_RATE
from .diarize import Turn, diarize
from .events import bus
from .merge import ME, build_transcript, speakers_in
from .whisper import Segment, drop_repeats, is_hallucination, pool

log = logging.getLogger(__name__)

SILENT_RMS = 0.002  # a stream quieter than this is treated as empty


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0


def transcribe_full(samples: np.ndarray, progress) -> tuple[list[Segment], str]:
    """Transcribe a long recording in ~3 minute windows (progress + bounded memory)."""
    s = config.load_settings()
    server = pool.final()
    windows = split_long_audio(samples, target_s=180.0)
    total = len(samples) or 1
    out: list[Segment] = []
    langs: dict[str, int] = {}
    for a, b in windows:
        progress(max(a / total, 0.01))  # also marks this step as started in the UI
        piece = samples[a:b]
        if _rms(piece) < SILENT_RMS:
            progress(b / total)
            continue
        segs, lang = server.transcribe(piece, language=s.language, vad=True, words=True)
        offset = a / SAMPLE_RATE
        for seg in segs:
            st, en = int(seg.start * SAMPLE_RATE), int(seg.end * SAMPLE_RATE)
            energy = _rms(piece[max(0, st) : max(st + 1, en)])
            if is_hallucination(seg, energy):
                continue
            seg.start += offset
            seg.end += offset
            for w in seg.words:
                w.start += offset
                w.end += offset
            out.append(seg)
        if lang:
            langs[lang] = langs.get(lang, 0) + (b - a)
        progress(b / total)
    main_lang = max(langs, key=langs.get) if langs else ""
    return drop_repeats(out), main_lang


class Progress:
    """Combines the progress of parallel steps into one number for the UI."""

    def __init__(self, mid: str, weights: dict[str, float]):
        self.mid = mid
        self.weights = weights
        self.parts = {k: 0.0 for k in weights}
        self.last_push = 0.0
        self.lock = threading.Lock()

    def cb(self, part: str):
        def f(v: float) -> None:
            with self.lock:
                self.parts[part] = min(1.0, max(0.0, v))
                total = sum(self.parts[k] * w for k, w in self.weights.items())
                if time.time() - self.last_push > 0.5 or v >= 1.0:
                    self.last_push = time.time()
                    stage = " + ".join(k for k, v2 in self.parts.items() if 0 < v2 < 1) or part
                    storage.update_meta(self.mid, progress=round(total, 3), stage=stage)
                    bus.publish({"type": "progress", "meeting": self.mid, "progress": round(total, 3), "stage": stage})

        return f


def process(mid: str) -> None:
    d = storage.meeting_dir(mid)
    meta = storage.update_meta(mid, status="processing", progress=0.0, stage="loading audio", error=None)
    pool.final().start()  # load the model first (can take a while on the first run)
    bus.publish({"type": "status", "meeting": mid, "status": "processing"})
    s = config.load_settings()

    if meta["mode"] == "upload":
        mic = None
        call = read_wav(d / "upload.wav")
    else:
        mic = read_wav(d / "mic.wav") if (d / "mic.wav").exists() else None
        call = read_wav(d / "call.wav") if (d / "call.wav").exists() else np.zeros(0, np.float32)
        if mic is not None and _rms(mic) < SILENT_RMS:
            mic = None

    has_call = len(call) > 0 and _rms(call) >= SILENT_RMS
    weights = {"transcribing your mic": 0.25 if mic is not None else 0.0,
               "transcribing call": 0.45 if has_call else 0.0,
               "identifying speakers": 0.3 if has_call else 0.0}
    norm = sum(weights.values()) or 1.0
    prog = Progress(mid, {k: v / norm for k, v in weights.items() if v})

    duration = max(len(call), len(mic) if mic is not None else 0) / SAMPLE_RATE
    with ThreadPoolExecutor(max_workers=2) as ex:
        diar_future = ex.submit(diarize, call, prog.cb("identifying speakers")) if has_call else None
        # whisper-server is single-request: transcribe the two streams one after the other
        mic_segs: list[Segment] = []
        call_segs: list[Segment] = []
        lang = ""
        if mic is not None:
            mic_segs, lang = transcribe_full(mic, prog.cb("transcribing your mic"))
        if has_call:
            call_segs, call_lang = transcribe_full(call, prog.cb("transcribing call"))
            lang = call_lang or lang
        turns: list[Turn] = []
        diar_error = None
        if diar_future:
            try:
                turns = diar_future.result()
            except Exception as e:  # keep the transcript even if diarization fails
                log.exception("diarization failed")
                diar_error = str(e)

    items = build_transcript(mic_segs, call_segs, turns, echo_dedup=s.echo_dedup)
    speakers = speakers_in(items)
    names = dict(meta.get("speakers") or {})
    for spk in speakers:
        names.setdefault(spk, (meta.get("user_name") or "Me") if spk == ME else spk)
    storage.write_transcript(mid, {"segments": items, "language": lang, "diarization_error": diar_error})

    storage.update_meta(mid, stage="preparing playback audio", progress=0.98)
    make_playback_audio([d / "mic.wav", d / "call.wav", d / "upload.wav"], d / "audio.opus")

    meta = storage.update_meta(
        mid, status="done", progress=1.0, stage="", speakers=names, language=lang,
        duration=meta.get("duration") or round(duration, 1),
        error=f"Speaker identification failed: {diar_error}" if diar_error else None,
    )
    bus.publish({"type": "status", "meeting": mid, "status": "done"})

    if s.auto_summary and s.summary_provider != "none" and items:
        from .summarize import summarize_meeting

        try:
            summarize_meeting(mid)
        except Exception as e:
            log.warning("auto summary failed: %s", e)
            storage.update_meta(mid, summary_error=str(e))


class JobQueue:
    """Processes meetings one at a time in a background thread."""

    def __init__(self) -> None:
        self.q: queue.Queue[str] = queue.Queue()
        self.current: str | None = None
        self._thread = threading.Thread(target=self._run, name="jobs", daemon=True)
        self._thread.start()

    def submit(self, mid: str) -> None:
        self.q.put(mid)

    def _run(self) -> None:
        while True:
            mid = self.q.get()
            self.current = mid
            try:
                process(mid)
            except Exception as e:
                log.exception("processing %s failed", mid)
                storage.update_meta(mid, status="error", error=str(e), stage="")
                bus.publish({"type": "status", "meeting": mid, "status": "error", "error": str(e)})
            finally:
                self.current = None


jobs = JobQueue()
