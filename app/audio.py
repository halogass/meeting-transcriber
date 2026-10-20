"""Audio devices and capture through PulseAudio / PipeWire (pipewire-pulse).

Capture uses `parec`, which works the same on PulseAudio and PipeWire and can record
from "monitor" sources, i.e. whatever the computer is playing (the remote side of a call).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from .config import SAMPLE_RATE

log = logging.getLogger(__name__)

CLIENT_NAME = "meeting-transcriber"
BLOCK_SAMPLES = SAMPLE_RATE // 10  # 100 ms


def capture_available() -> bool:
    return bool(shutil.which("parec") and shutil.which("pactl")) and _pactl_ok()


def _pactl_ok() -> bool:
    try:
        return subprocess.run(["pactl", "info"], capture_output=True, timeout=3).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _pactl_json(*args: str):
    out = subprocess.run(["pactl", "-f", "json", *args], capture_output=True, text=True, timeout=5)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or "pactl failed")
    return json.loads(out.stdout or "[]")


def _pactl_get(what: str) -> str:
    out = subprocess.run(["pactl", what], capture_output=True, text=True, timeout=3)
    return out.stdout.strip()


def list_sources() -> dict:
    """Microphones and monitor (loopback) sources, plus the current defaults."""
    if not capture_available():
        return {"available": False, "mics": [], "monitors": [], "default_mic": "", "default_monitor": ""}
    sources = _pactl_json("list", "sources")
    mics, monitors = [], []
    for s in sources:
        item = {"name": s["name"], "label": s.get("description") or s["name"]}
        if s.get("monitor_of_sink") not in (None, "", "n/a") or s["name"].endswith(".monitor"):
            monitors.append(item)
        else:
            mics.append(item)
    default_sink = _pactl_get("get-default-sink")
    return {
        "available": True,
        "mics": mics,
        "monitors": monitors,
        "default_mic": _pactl_get("get-default-source"),
        "default_monitor": f"{default_sink}.monitor" if default_sink else "",
    }


def resolve_source(name: str, kind: str) -> str:
    if name:
        return name
    return "@DEFAULT_SOURCE@" if kind == "mic" else "@DEFAULT_MONITOR@"


class Capture:
    """Records one source to a raw PCM file (crash-safe) and hands 100 ms blocks to a callback."""

    def __init__(self, source: str, raw_path: Path, on_block: Callable[[np.ndarray], None]):
        self.source = source
        self.raw_path = raw_path
        self.on_block = on_block
        self.level = 0.0  # RMS of the latest block, 0..1
        self.samples = 0
        self.error: str | None = None
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        cmd = [
            "parec",
            f"--device={self.source}",
            "--format=s16le",
            f"--rate={SAMPLE_RATE}",
            "--channels=1",
            "--latency-msec=100",
            f"--client-name={CLIENT_NAME}",
            "--stream-name=capture",
        ]
        from .config import supervised

        self._proc = subprocess.Popen(supervised(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._thread = threading.Thread(target=self._run, name=f"capture-{self.source}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._proc and self._proc.stdout
        nbytes = BLOCK_SAMPLES * 2
        with open(self.raw_path, "ab", buffering=0) as f:
            while not self._stop.is_set():
                data = self._proc.stdout.read(nbytes)
                if not data:
                    break
                f.write(data)
                block = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self.samples += len(block)
                self.level = float(np.sqrt(np.mean(block**2))) if len(block) else 0.0
                try:
                    self.on_block(block)
                except Exception:  # never let a consumer bug kill the recording
                    log.exception("audio consumer failed")
        if not self._stop.is_set():
            err = self._proc.stderr.read().decode(errors="replace").strip() if self._proc.stderr else ""
            self.error = err or "audio capture ended unexpectedly"
            log.error("capture %s stopped: %s", self.source, self.error)

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread:
            self._thread.join(timeout=3)


def raw_to_wav(raw: Path, wav: Path) -> None:
    """Wrap a raw s16le mono 16 kHz recording in a WAV header."""
    with open(raw, "rb") as src, wave.open(str(wav), "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(SAMPLE_RATE)
        while chunk := src.read(1 << 20):
            dst.writeframes(chunk)


def read_wav(path: Path) -> np.ndarray:
    """Read a 16 kHz mono WAV as float32 in -1..1."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sr}")
    return data


def wav_bytes(samples: np.ndarray) -> bytes:
    """Encode float32 samples as an in-memory 16 kHz mono s16 WAV."""
    import io

    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def convert_to_wav16k(src: Path, dst: Path) -> None:
    """Convert any audio/video file to 16 kHz mono WAV with ffmpeg."""
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {res.stderr.strip()}")


def make_playback_audio(inputs: list[Path], dst: Path) -> None:
    """Mix the recorded streams into one compact Opus file for playback in the browser."""
    inputs = [p for p in inputs if p.exists() and p.stat().st_size > 44]
    if not inputs:
        return
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error"]
    for p in inputs:
        cmd += ["-i", str(p)]
    if len(inputs) > 1:
        cmd += ["-filter_complex", f"amix=inputs={len(inputs)}:duration=longest:normalize=0"]
    cmd += ["-ac", "1", "-c:a", "libopus", "-b:a", "32k", str(dst)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log.error("ffmpeg mix failed: %s", res.stderr.strip())
