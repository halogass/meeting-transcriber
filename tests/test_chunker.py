import numpy as np

from app.chunker import Chunker, split_long_audio
from app.config import SAMPLE_RATE


def tone(seconds, amp=0.1):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds, amp=0.0005):
    rng = np.random.default_rng(0)
    return (amp * rng.standard_normal(int(seconds * SAMPLE_RATE))).astype(np.float32)


def test_emits_utterance_after_trailing_silence():
    c = Chunker()
    chunks = c.feed(silence(3)) + c.feed(tone(2.0)) + c.feed(silence(1.0))
    assert len(chunks) == 1
    ch = chunks[0]
    assert 2.4 <= ch.start <= 3.0  # includes a short pre-roll before speech
    assert 2.0 <= ch.duration <= 3.2


def test_silence_alone_produces_nothing():
    c = Chunker()
    assert c.feed(silence(20)) == []
    assert c.flush() is None


def test_long_speech_is_cut_at_max_length():
    c = Chunker(max_chunk_s=12)
    chunks = c.feed(silence(2)) + c.feed(tone(30))
    assert chunks and all(ch.duration <= 12.01 for ch in chunks)


def test_timestamps_are_continuous():
    c = Chunker()
    audio = np.concatenate([silence(2), tone(2), silence(1), tone(3), silence(1)])
    chunks = c.feed(audio)
    assert len(chunks) == 2
    assert chunks[1].start >= chunks[0].start + chunks[0].duration - 1e-6


def test_split_long_audio_covers_everything():
    audio = np.concatenate([tone(290), silence(1), tone(300), silence(1), tone(100)])
    bounds = split_long_audio(audio, target_s=300)
    assert bounds[0][0] == 0 and bounds[-1][1] == len(audio)
    for (a0, a1), (b0, _) in zip(bounds, bounds[1:]):
        assert a1 == b0
    assert len(bounds) == 3


def test_split_short_audio_is_single_window():
    audio = tone(60)
    assert split_long_audio(audio) == [(0, len(audio))]

