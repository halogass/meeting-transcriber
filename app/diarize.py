"""Speaker diarization with pyannote.audio (who spoke when)."""

from __future__ import annotations

import gc
import logging
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import config

log = logging.getLogger(__name__)
_lock = threading.Lock()  # one diarization at a time (GPU memory)


@dataclass
class Turn:
    start: float
    end: float
    speaker: str


class DiarizationError(RuntimeError):
    pass


def _load_pipeline(device: str):
    import torch
    from pyannote.audio import Pipeline

    s = config.load_settings()
    token = config.hf_token()
    if not token:
        raise DiarizationError("No Hugging Face token. Add one in Settings (needed to download the pyannote model).")
    try:
        pipeline = Pipeline.from_pretrained(s.diarization_model, token=token)
    except Exception as e:  # gated repo, bad token, offline...
        msg = str(e)
        if "403" in msg or "gated" in msg.lower() or "restricted" in msg.lower():
            raise DiarizationError(
                f"Access to {s.diarization_model} is not granted for this Hugging Face token. "
                f"Open https://huggingface.co/{s.diarization_model} and accept the terms."
            ) from e
        raise DiarizationError(f"Could not load {s.diarization_model}: {msg}") from e
    if pipeline is None:
        raise DiarizationError(f"Could not load {s.diarization_model} (check token and model access)")
    pipeline.to(torch.device(device))
    return pipeline


def diarize(samples: np.ndarray, progress: Callable[[float], None] | None = None) -> list[Turn]:
    """Diarize 16 kHz mono float32 audio. Returns non-overlapping speaker turns."""
    import torch

    from .config import SAMPLE_RATE

    if len(samples) < SAMPLE_RATE:  # under a second: nothing to split
        return []
    s = config.load_settings()
    device = config.diarization_torch_device()
    with _lock:
        log.info("diarizing %.0f s of audio on %s", len(samples) / SAMPLE_RATE, device)
        pipeline = _load_pipeline(device)
        try:
            kwargs = {}
            if s.min_speakers:
                kwargs["min_speakers"] = s.min_speakers
            if s.max_speakers:
                kwargs["max_speakers"] = s.max_speakers

            def hook(step_name, step_artifact, file=None, total=None, completed=None):
                if progress and total:
                    # segmentation and embeddings are the long steps
                    weight = {"segmentation": (0.0, 0.3), "embeddings": (0.3, 0.95)}.get(step_name)
                    if weight:
                        lo, hi = weight
                        progress(lo + (hi - lo) * (completed or 0) / total)

            waveform = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)
            output = pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, hook=hook, **kwargs)
            # pyannote 4 returns an object with an "exclusive" (one speaker at a time) view,
            # which maps cleanly onto transcript words. pyannote 3 returns an Annotation.
            annotation = getattr(output, "exclusive_speaker_diarization", None) or getattr(
                output, "speaker_diarization", output
            )
            turns = [
                Turn(float(seg.start), float(seg.end), str(label))
                for seg, _, label in annotation.itertracks(yield_label=True)
            ]
        finally:
            del pipeline
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()  # give VRAM back between meetings
    turns.sort(key=lambda t: t.start)
    return _relabel(turns)


def _relabel(turns: list[Turn]) -> list[Turn]:
    """SPEAKER_00.. in order of first appearance -> 'Speaker 1', 'Speaker 2', ..."""
    mapping: dict[str, str] = {}
    for t in turns:
        if t.speaker not in mapping:
            mapping[t.speaker] = f"Speaker {len(mapping) + 1}"
        t.speaker = mapping[t.speaker]
    return turns
