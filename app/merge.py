"""Combine transcription and diarization into one speaker-attributed transcript."""

from __future__ import annotations

import bisect
import re
from difflib import SequenceMatcher

from .diarize import Turn
from .whisper import Segment, Word

ME = "Me"  # label for the local microphone stream


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


class TurnIndex:
    """Fast "who was speaking at time t" over sorted, mostly non-overlapping speaker turns."""

    def __init__(self, turns: list[Turn]):
        self.turns = sorted(turns, key=lambda t: t.start)
        self.starts = [t.start for t in self.turns]

    def speaker(self, t0: float, t1: float) -> str | None:
        """Speaker with the most overlap with [t0, t1]; the nearest turn if none overlaps."""
        if not self.turns:
            return None
        hi = bisect.bisect_right(self.starts, t1)
        best, best_ov = None, 0.0
        i = hi - 1
        while i >= 0 and i >= hi - 8:  # only a few turns can overlap a short span
            t = self.turns[i]
            ov = _overlap(t0, t1, t.start, t.end)
            if ov > best_ov:
                best, best_ov = t.speaker, ov
            i -= 1
        if best:
            return best
        mid = (t0 + t1) / 2
        near = [self.turns[j] for j in (hi - 1, hi) if 0 <= j < len(self.turns)]
        return min(near, key=lambda t: min(abs(mid - t.start), abs(mid - t.end))).speaker


def speaker_at(t0: float, t1: float, turns: list[Turn]) -> str | None:
    return TurnIndex(turns).speaker(t0, t1)


def _join(words: list[Word]) -> str:
    return re.sub(r"\s+", " ", "".join(w.text for w in words)).strip()


_SENTENCE_END = re.compile(r"[.?!…]['\")]*$")


def smooth_labels(words: list[Word], labels: list[str], max_island_words: int = 4, max_island_s: float = 2.0,
                  snap_words: int = 2) -> list[str]:
    """Clean up word-level speaker labels.

    1. A short island (up to 4 words, 2 s) between two runs of the same speaker that is not a
       complete sentence ("I guess that's" | "all from me.") is a timing error: give it to the
       surrounding speaker. Complete short replies ("Okay, noted.") are kept.
    2. Move a speaker change to a sentence end when one is within `snap_words` words, because
       word timestamps are approximate but people rarely hand over mid-sentence.
    """
    labels = list(labels)
    n = len(labels)

    def runs():
        out, i = [], 0
        while i < n:
            j = i
            while j + 1 < n and labels[j + 1] == labels[i]:
                j += 1
            out.append((i, j))
            i = j + 1
        return out

    for a, b in runs():
        if a == 0 or b == n - 1:
            continue
        if labels[a - 1] != labels[b + 1]:
            continue
        size, dur = b - a + 1, words[b].end - words[a].start
        # a short island that isn't a complete sentence ("I guess that's" | "all for me.")
        broken = not _SENTENCE_END.search(words[b].text.strip()) or not _SENTENCE_END.search(words[a - 1].text.strip())
        if broken and size <= max_island_words and dur <= max_island_s:
            for k in range(a, b + 1):
                labels[k] = labels[a - 1]

    for a, _ in runs():
        if a == 0:
            continue
        if _SENTENCE_END.search(words[a - 1].text.strip()):
            continue  # already at a sentence boundary
        # candidate boundaries b (a sentence ends at word b-1), nearest first
        for d in range(1, snap_words + 1):
            moved = False
            for b in (a - d, a + d):
                if 0 < b < n and _SENTENCE_END.search(words[b - 1].text.strip()):
                    lo, hi = min(a, b), max(a, b)
                    # don't cross another speaker change
                    if len({labels[k] for k in range(lo - 1, hi + 1) if 0 <= k < n}) > 2:
                        continue
                    new = labels[a] if b < a else labels[a - 1]
                    for k in range(lo, hi):
                        labels[k] = new
                    moved = True
                    break
            if moved:
                break
    return labels


def assign_speakers(segments: list[Segment], turns: list[Turn], default: str = "Speaker 1") -> list[dict]:
    """Label every word with the speaker talking at that moment, then group into runs.
    Segments without word timings are labelled as a whole."""
    if not turns:
        return [_item(seg.start, seg.end, default, seg.text) for seg in segments]
    index = TurnIndex(turns)
    words: list[Word] = []
    for seg in segments:
        if seg.words:
            words.extend(seg.words)
        else:
            words.append(Word(" " + seg.text, seg.start, seg.end))
    words.sort(key=lambda w: w.start)
    labels = [index.speaker(w.start, max(w.end, w.start + 0.02)) or default for w in words]
    labels = smooth_labels(words, labels)
    out: list[dict] = []
    run: list[Word] = []
    for w, spk in zip(words, labels):
        # new item when the speaker changes or after a long pause
        if run and (spk != run_spk or w.start - run[-1].end > 2.0):
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


def merge_paragraphs(items: list[dict], max_gap: float = 4.0, max_chars: int = 700) -> list[dict]:
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
