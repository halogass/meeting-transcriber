"""Runs whisper.cpp's `whisper-server` as a child process and talks to it over HTTP.

The server keeps the model loaded in (GPU) memory, so live chunks and the final pass
don't pay the model load cost. It handles one request at a time.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field

import httpx
import numpy as np

from . import config
from .audio import wav_bytes

log = logging.getLogger(__name__)

# Phrases Whisper tends to invent on silence/music, in both of our languages.
HALLUCINATIONS = {
    "thank you.", "thank you", "thanks for watching!", "thanks for watching.", "thank you for watching.",
    "terima kasih.", "terima kasih", "terima kasih telah menonton.", "terima kasih sudah menonton.",
    "subtitle by", "subtitles by the amara.org community", "[music]", "[musik]", "(music)", "you",
    "sampai jumpa di video selanjutnya.", "silakan subscribe", "...", ".",
}


@dataclass
class Word:
    text: str
    start: float
    end: float
    p: float = 1.0


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    language: str = ""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class WhisperServer:
    def __init__(self, model: str):
        self.model = model
        self.port = 0
        self.proc: subprocess.Popen | None = None
        self.gpu: int | None = None
        self._lock = threading.Lock()
        self._log_tail: list[str] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        with self._lock:
            if self.running():
                return
            binary = config.BIN_DIR / "whisper-server"
            model = config.model_path(self.model)
            if not binary.exists():
                raise RuntimeError(f"{binary} not found. Run scripts/build_whisper.sh")
            if not model.exists():
                raise RuntimeError(f"Model {model.name} not found. Run scripts/download_models.sh")
            s = config.load_settings()
            self.port = _free_port()
            self.gpu = config.whisper_gpu()
            cmd = [str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(self.port),
                   "-t", str(s.whisper_threads), "-l", "auto"]
            vad = config.model_path(s.vad_model)
            if vad.exists():
                cmd += ["-vm", str(vad)]
            if self.gpu is None:
                cmd += ["-ng"]
            else:
                cmd += ["-dev", str(self.gpu)]
            log.info("starting whisper-server: %s", " ".join(cmd))
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            threading.Thread(target=self._drain, daemon=True).start()
            self._wait_ready()

    def _drain(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            self._log_tail = (self._log_tail + [line.rstrip()])[-40:]

    def _wait_ready(self, timeout: float = 180) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.running():
                raise RuntimeError("whisper-server exited:\n" + "\n".join(self._log_tail[-15:]))
            try:
                r = httpx.get(self.url + "/health", timeout=2)
                if r.status_code == 200:
                    log.info("whisper-server ready on port %s (gpu=%s)", self.port, self.gpu)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise RuntimeError("whisper-server did not become ready in time")

    def stop(self) -> None:
        with self._lock:
            if self.running():
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            self.proc = None

    def transcribe(
        self,
        samples: np.ndarray,
        language: str = "auto",
        vad: bool = False,
        words: bool = True,
        prompt: str = "",
        timeout: float = 3600,
    ) -> tuple[list[Segment], str]:
        """Transcribe float32 16 kHz mono audio. Returns (segments, detected language)."""
        if not self.running():
            self.start()
        form = {
            "response_format": "verbose_json",
            "language": language or "auto",
            "temperature": "0.0",
            "token_timestamps": "true" if words else "false",
            "split_on_word": "true",
            "vad": "true" if vad else "false",
            "no_language_probabilities": "true",
        }
        if prompt:
            form["prompt"] = prompt
        files = {"file": ("audio.wav", wav_bytes(samples), "audio/wav")}
        r = httpx.post(self.url + "/inference", data=form, files=files, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(data["error"])
        return parse_segments(data), data.get("language", "")


def _clean_word(text: str) -> str:
    return text.replace("▁", " ")


def parse_segments(data: dict) -> list[Segment]:
    out: list[Segment] = []
    for seg in data.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        words = []
        for w in seg.get("words") or []:
            wt = _clean_word(w.get("word", ""))
            # skip special tokens such as [_BEG_] or [_TT_123]
            if not wt.strip() or (wt.strip().startswith("[_") and wt.strip().endswith("]")):
                continue
            if "start" in w:
                words.append(Word(wt, float(w["start"]), float(w["end"]), float(w.get("probability", 1.0))))
        out.append(
            Segment(
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=text,
                words=words,
                no_speech_prob=float(seg.get("no_speech_prob", 0.0)),
                avg_logprob=float(seg.get("avg_logprob", 0.0)),
            )
        )
    return out


def is_hallucination(seg: Segment, energy: float | None = None) -> bool:
    t = seg.text.strip().lower()
    if not t:
        return True
    if t in HALLUCINATIONS and (energy is None or energy < 0.02 or seg.no_speech_prob > 0.3):
        return True
    if seg.no_speech_prob > 0.8 and seg.avg_logprob < -1.0:
        return True
    # repeated-loop output ("ya ya ya ya ya ...")
    toks = t.split()
    if len(toks) >= 8 and len(set(toks)) <= 2:
        return True
    return False


def _norm(text: str) -> str:
    return " ".join("".join(c for c in text.lower() if c.isalnum() or c.isspace()).split())


def _sentences(seg: Segment) -> list[list[Word]]:
    out, cur = [], []
    for w in seg.words:
        cur.append(w)
        if w.text.rstrip().endswith((".", "?", "!")):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def drop_repeats(segments: list[Segment], lookback: int = 4, window_s: float = 90.0) -> list[Segment]:
    """Remove Whisper's repetition loops: a sentence or segment that just repeats the
    previous one(s) word for word. Real speech rarely repeats a whole sentence verbatim."""
    out: list[Segment] = []
    recent: list[tuple[float, str]] = []  # (end time, normalized sentence)

    def seen(text: str, t: float) -> bool:
        n = _norm(text)
        return len(n) >= 12 and any(n == r and t - end <= window_s for end, r in recent[-lookback:])

    for seg in segments:
        if seg.words:
            kept_words: list[Word] = []
            for sent in _sentences(seg):
                text = "".join(w.text for w in sent)
                if seen(text, sent[0].start):
                    continue
                kept_words += sent
                recent.append((sent[-1].end, _norm(text)))
            if not kept_words:
                continue
            if len(kept_words) != len(seg.words):
                seg.words = kept_words
                seg.text = " ".join("".join(w.text for w in kept_words).split())
                seg.start, seg.end = kept_words[0].start, kept_words[-1].end
            out.append(seg)
        else:
            if seen(seg.text, seg.start):
                continue
            recent.append((seg.end, _norm(seg.text)))
            out.append(seg)
    return out


class WhisperPool:
    """One server for everything, or a separate (smaller) one for live captions."""

    def __init__(self) -> None:
        self._final: WhisperServer | None = None
        self._live: WhisperServer | None = None
        self._lock = threading.Lock()

    def final(self) -> WhisperServer:
        with self._lock:
            s = config.load_settings()
            if self._final is None or self._final.model != s.whisper_model:
                if self._final:
                    self._final.stop()
                self._final = WhisperServer(s.whisper_model)
            return self._final

    def live(self) -> WhisperServer:
        s = config.load_settings()
        if not s.live_model or s.live_model == s.whisper_model:
            return self.final()
        with self._lock:
            if self._live is None or self._live.model != s.live_model:
                if self._live:
                    self._live.stop()
                self._live = WhisperServer(s.live_model)
            return self._live

    def status(self) -> dict:
        f = self._final
        return {
            "running": bool(f and f.running()),
            "model": f.model if f else config.load_settings().whisper_model,
            "gpu": f.gpu if f else config.whisper_gpu(),
        }

    def stop_all(self) -> None:
        for s in (self._final, self._live):
            if s:
                s.stop()


pool = WhisperPool()
