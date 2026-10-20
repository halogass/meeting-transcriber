"""Splits a live audio stream into utterance-sized chunks for live captions.

Energy-based with an adaptive noise floor: cheap, no model, good enough to decide
where to cut. The final pass re-transcribes the whole recording anyway.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import SAMPLE_RATE

BLOCK_S = 0.1


@dataclass
class Chunk:
    start: float  # seconds from the start of the recording
    samples: np.ndarray
    energy: float  # RMS of the speech blocks

    @property
    def duration(self) -> float:
        return len(self.samples) / SAMPLE_RATE


class Chunker:
    def __init__(
        self,
        min_chunk_s: float = 1.5,
        max_chunk_s: float = 12.0,
        end_silence_s: float = 0.6,
        pre_roll_s: float = 0.3,
        min_threshold: float = 0.004,
        max_threshold: float = 0.02,
    ):
        self.min_blocks = round(min_chunk_s / BLOCK_S)
        self.max_blocks = round(max_chunk_s / BLOCK_S)
        self.end_silence_blocks = round(end_silence_s / BLOCK_S)
        self.pre_roll = round(pre_roll_s / BLOCK_S)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self._history: deque[float] = deque(maxlen=round(60 / BLOCK_S))  # 60 s of block energies
        self._buf: list[np.ndarray] = []
        self._buf_rms: list[float] = []
        self._buf_start_block = 0
        self._block_index = 0
        self._pending = np.zeros(0, dtype=np.float32)

    def threshold(self) -> float:
        if len(self._history) < 10:
            return max(self.min_threshold, 0.01)
        # noise floor = quietest moments of the last minute; capped so that a long
        # monologue (few pauses) can never push the threshold above normal speech level
        floor = float(np.percentile(self._history, 5))
        return min(self.max_threshold, max(self.min_threshold, floor * 3.0))

    def feed(self, samples: np.ndarray) -> list[Chunk]:
        """Feed any amount of audio; returns finished chunks."""
        self._pending = np.concatenate([self._pending, samples])
        n = int(BLOCK_S * SAMPLE_RATE)
        out: list[Chunk] = []
        while len(self._pending) >= n:
            block, self._pending = self._pending[:n], self._pending[n:]
            c = self._feed_block(block)
            if c:
                out.append(c)
        return out

    def _feed_block(self, block: np.ndarray) -> Chunk | None:
        rms = float(np.sqrt(np.mean(block**2)))
        thr = self.threshold()
        self._history.append(rms)
        if not self._buf:
            self._buf_start_block = self._block_index
        self._buf.append(block)
        self._buf_rms.append(rms)
        self._block_index += 1

        speech = [r > thr for r in self._buf_rms]
        if not any(speech):
            # nothing said yet: keep only a short pre-roll so the first syllable isn't cut
            if len(self._buf) > self.pre_roll:
                drop = len(self._buf) - self.pre_roll
                self._buf = self._buf[drop:]
                self._buf_rms = self._buf_rms[drop:]
                self._buf_start_block += drop
            return None

        trailing_silence = 0
        for s in reversed(speech):
            if s:
                break
            trailing_silence += 1

        if len(self._buf) >= self.min_blocks and trailing_silence >= self.end_silence_blocks:
            return self._emit(len(self._buf), thr)
        if len(self._buf) >= self.max_blocks:
            # cut at the quietest block in the last 3 s to avoid splitting a word
            window = min(30, len(self._buf) - self.min_blocks)
            tail = self._buf_rms[-window:]
            cut = len(self._buf) - window + int(np.argmin(tail)) + 1
            return self._emit(cut, thr)
        return None

    def _emit(self, n_blocks: int, thr: float) -> Chunk:
        blocks, rms = self._buf[:n_blocks], self._buf_rms[:n_blocks]
        start = self._buf_start_block * BLOCK_S
        self._buf, self._buf_rms = self._buf[n_blocks:], self._buf_rms[n_blocks:]
        self._buf_start_block += n_blocks
        speech_rms = [r for r in rms if r > thr] or rms
        return Chunk(start=start, samples=np.concatenate(blocks), energy=float(np.mean(speech_rms)))

    def flush(self) -> Chunk | None:
        if self._buf and any(r > self.threshold() for r in self._buf_rms):
            return self._emit(len(self._buf), self.threshold())
        return None


def split_long_audio(samples: np.ndarray, target_s: float = 300.0, search_s: float = 10.0) -> list[tuple[int, int]]:
    """Split a long recording into ~target_s windows, cutting at the quietest 100 ms
    near each boundary. Returns (start_sample, end_sample) pairs covering everything."""
    n = len(samples)
    target = int(target_s * SAMPLE_RATE)
    if n <= target * 1.2:
        return [(0, n)]
    blk = int(BLOCK_S * SAMPLE_RATE)
    search = int(search_s * SAMPLE_RATE)
    bounds, pos = [], 0
    while n - pos > target * 1.2:
        lo, hi = pos + target - search, min(pos + target + search, n)
        seg = samples[lo:hi]
        nb = len(seg) // blk
        energies = np.sqrt(np.mean(seg[: nb * blk].reshape(nb, blk) ** 2, axis=1))
        cut = lo + int(np.argmin(energies)) * blk + blk // 2
        bounds.append((pos, cut))
        pos = cut
    bounds.append((pos, n))
    return bounds
