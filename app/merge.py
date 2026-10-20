"""Combine transcription and diarization into one speaker-attributed transcript."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from .diarize import Turn
from .whisper import Segment, Word

ME = "Me"  # label for the local microphone stream


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def speaker_at(t0: float, t1: float, turns: list[Turn]) -> str | None:
    """Speaker with the most overlap with [t0, t1]; nearest turn if none overlaps."""
    if not turns:
        return None
    best, best_ov = None, 0.0
    for t in turns:
        if t.start > t1:
            break
        ov = _overlap(t0, t1, t.start, t.end)
        if ov > best_ov:
            best, best_ov = t.speaker, ov
    if best:
        return best
    mid = (t0 + t1) / 2
    nearest = min(turns, key=lambda t: min(abs(mid - t.start), abs(mid - t.end)))
    return nearest.speaker


def _join(words: list[Word]) -> str:
    return re.sub(r"\s+", " ", "".join(w.text for w in words)).strip()


def assign_speakers(segments: list[Segment], turns: list[Turn], default: str = "Speaker 1") -> list[dict]:
    """Label each segment; split a segment where the speaker changes mid-way (word level)."""
    out: list[dict] = []
    for seg in segments:
        if not turns:
            out.append(_item(seg.start, seg.end, default, seg.text))
            continue
        if not seg.words:
            out.append(_item(seg.start, seg.end, speaker_at(seg.start, seg.end, turns) or default, seg.text))
            continue
        run: list[Word] = []
        run_spk = None
        for w in seg.words:
            spk = speaker_at(w.start, max(w.end, w.start + 0.02), turns) or default
            if run and spk != run_spk:
                out.append(_item(run[0].start, run[-1].end, run_spk, _join(run)))
                run = []
            run.append(w)
            run_spk = spk
        if run:
            out.append(_item(run[0].start, run[-1].end, run_spk, _join(run)))
    # drop fragments that ended up empty or punctuation-only after splitting
    return [s for s in out if re.search(r"\w", s["text"])]


def _item(start: float, end: float, speaker: str, text: str) -> dict:
    return {"start": round(start, 2), "end": round(max(end, start), 2), "speaker": speaker, "text": text.strip()}


def _norm(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def drop_echo(mic: list[dict], call: list[dict], window: float = 2.0, threshold: float = 0.6) -> list[dict]:
    """Remove mic segments that are the remote side leaking from speakers into the microphone."""
    kept = []
    for m in mic:
        mt = _norm(m["text"])
        echo = False
        for c in call:
            if c["end"] < m["start"] - window or c["start"] > m["end"] + window:
                continue
            ct = _norm(c["text"])
            if not mt or not ct:
                continue
            ratio = SequenceMatcher(None, mt, ct).ratio()
            if ratio >= threshold or (len(mt) > 10 and mt in ct):
                echo = True
                break
        if not echo:
            kept.append(m)
    return kept


def merge_paragraphs(items: list[dict], max_gap: float = 1.5, max_chars: int = 600) -> list[dict]:
    """Merge consecutive segments by the same speaker into readable paragraphs."""
    out: list[dict] = []
    for it in sorted(items, key=lambda s: (s["start"], s["end"])):
        last = out[-1] if out else None
        if (
            last
            and last["speaker"] == it["speaker"]
            and it["start"] - last["end"] <= max_gap
            and len(last["text"]) + len(it["text"]) < max_chars
        ):
            last["text"] = f'{last["text"]} {it["text"]}'.strip()
            last["end"] = max(last["end"], it["end"])
        else:
            out.append(dict(it))
    return out


def build_transcript(
    mic_segments: list[Segment] | None,
    call_segments: list[Segment],
    call_turns: list[Turn],
    echo_dedup: bool = True,
) -> list[dict]:
    """mic_segments: the local microphone (always the user). call_segments + turns: everyone else."""
    call = assign_speakers(call_segments, call_turns)
    mic = [_item(s.start, s.end, ME, s.text) for s in (mic_segments or [])]
    if echo_dedup and mic and call:
        mic = drop_echo(mic, call)
    return merge_paragraphs(mic + call)


def speakers_in(items: list[dict]) -> list[str]:
    seen: list[str] = []
    for it in items:
        if it["speaker"] not in seen:
            seen.append(it["speaker"])
    # Me first, then Speaker 1, 2, ..., 10 (numeric order)
    def key(s: str):
        m = re.search(r"(\d+)$", s)
        return (s != ME, int(m.group(1)) if m else 0, s)

    return sorted(seen, key=key)
