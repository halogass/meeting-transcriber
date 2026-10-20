from app.diarize import Turn
from app.merge import ME, assign_speakers, build_transcript, drop_echo, merge_paragraphs, speaker_at, speakers_in
from app.whisper import Segment, Word, is_hallucination, parse_segments


def seg(start, end, text, words=None):
    return Segment(start=start, end=end, text=text, words=words or [])


def test_speaker_at_prefers_largest_overlap():
    turns = [Turn(0, 5, "Speaker 1"), Turn(5, 10, "Speaker 2")]
    assert speaker_at(4, 9, turns) == "Speaker 2"
    assert speaker_at(1, 6, turns) == "Speaker 1"


def test_speaker_at_falls_back_to_nearest_turn():
    turns = [Turn(0, 2, "Speaker 1"), Turn(10, 12, "Speaker 2")]
    assert speaker_at(8.5, 9, turns) == "Speaker 2"


def test_assign_speakers_splits_segment_on_speaker_change():
    words = [Word(" Halo", 0.0, 0.4), Word(" semua.", 0.4, 0.9), Word(" Hi", 1.2, 1.5), Word(" there.", 1.5, 2.0)]
    turns = [Turn(0, 1.0, "Speaker 1"), Turn(1.1, 2.2, "Speaker 2")]
    out = assign_speakers([seg(0, 2, "Halo semua. Hi there.", words)], turns)
    assert [(o["speaker"], o["text"]) for o in out] == [("Speaker 1", "Halo semua."), ("Speaker 2", "Hi there.")]


def test_assign_speakers_without_turns_uses_default():
    out = assign_speakers([seg(0, 1, "hello")], [])
    assert out[0]["speaker"] == "Speaker 1"


def test_drop_echo_removes_leaked_call_audio():
    mic = [{"start": 10.2, "end": 12, "speaker": ME, "text": "Can you share the report by Friday?"},
           {"start": 20, "end": 22, "speaker": ME, "text": "Sure, I will send it today."}]
    call = [{"start": 10, "end": 12, "speaker": "Speaker 1", "text": "Can you share the report by Friday"}]
    kept = drop_echo(mic, call)
    assert [k["text"] for k in kept] == ["Sure, I will send it today."]


def test_merge_paragraphs_joins_same_speaker():
    items = [
        {"start": 0, "end": 2, "speaker": "A", "text": "one"},
        {"start": 2.5, "end": 4, "speaker": "A", "text": "two"},
        {"start": 4.2, "end": 5, "speaker": "B", "text": "three"},
        {"start": 10, "end": 11, "speaker": "B", "text": "four"},
    ]
    out = merge_paragraphs(items)
    assert [(o["speaker"], o["text"]) for o in out] == [("A", "one two"), ("B", "three"), ("B", "four")]


def test_build_transcript_orders_mic_and_call():
    mic = [seg(3, 4, "Saya setuju.")]
    call = [seg(0, 2, "Bagaimana menurut Anda?"), seg(5, 6, "Oke, lanjut.")]
    turns = [Turn(0, 2.5, "Speaker 1"), Turn(4.8, 6.5, "Speaker 2")]
    out = build_transcript(mic, call, turns)
    assert [o["speaker"] for o in out] == ["Speaker 1", ME, "Speaker 2"]
    assert speakers_in(out) == [ME, "Speaker 1", "Speaker 2"]


def test_speakers_in_numeric_order():
    items = [{"speaker": f"Speaker {n}"} for n in (10, 2, 1)]
    assert speakers_in(items) == ["Speaker 1", "Speaker 2", "Speaker 10"]


def test_parse_segments_skips_special_tokens():
    data = {"segments": [{"start": 0, "end": 1.5, "text": " Halo", "no_speech_prob": 0.01, "avg_logprob": -0.2,
                          "words": [{"word": "[_BEG_]", "start": 0, "end": 0}, {"word": " Halo", "start": 0.1, "end": 0.6}]}]}
    segs = parse_segments(data)
    assert len(segs) == 1 and segs[0].text == "Halo"
    assert [w.text for w in segs[0].words] == [" Halo"]


def test_hallucination_filter():
    assert is_hallucination(seg(0, 1, "Terima kasih."), energy=0.005)
    assert not is_hallucination(seg(0, 1, "Terima kasih."), energy=0.1)
    assert is_hallucination(seg(0, 5, "ya ya ya ya ya ya ya ya ya"))
    assert not is_hallucination(seg(0, 2, "Kita mulai rapatnya sekarang."))


def _words(sentence, start, step=0.3):
    toks = sentence.split()
    return [Word(" " + t, start + i * step, start + (i + 1) * step) for i, t in enumerate(toks)]


def test_drop_repeats_removes_whisper_loops():
    from app.whisper import drop_repeats

    loop = "Nanti kita bahas lagi jadwal rapat minggu depan."
    w = _words(loop, 10) + _words(loop, 13) + _words("Lalu topik berikutnya.", 16)
    segs = [
        seg(0, 3, "Selamat pagi semuanya.", _words("Selamat pagi semuanya.", 0)),
        seg(10, 18, f"{loop} {loop} Lalu topik berikutnya.", w),
        seg(20, 23, loop, _words(loop, 20)),
    ]
    out = drop_repeats(segs)
    assert [s.text for s in out] == ["Selamat pagi semuanya.", f"{loop} Lalu topik berikutnya."]


def test_drop_repeats_keeps_short_repeated_replies():
    from app.whisper import drop_repeats

    segs = [seg(0, 1, "Ya."), seg(2, 3, "Ya."), seg(4, 5, "Oke, siap.")]
    assert len(drop_repeats(segs)) == 3


def test_vad_word_times_are_mapped_onto_segment():
    data = {"segments": [{"start": 10.0, "end": 12.0, "text": " Hello there.",
                          "words": [{"word": " Hello", "start": 0.5, "end": 1.0, "t_dtw": 60},
                                    {"word": " there", "start": 1.0, "end": 1.4, "t_dtw": 110},
                                    {"word": ".", "start": 1.4, "end": 1.5, "t_dtw": 140}]}]}
    seg = parse_segments(data, vad=True)[0]
    assert [w.text for w in seg.words] == [" Hello", " there."]  # punctuation joined to the word
    assert 10.0 <= seg.words[0].start < seg.words[1].start <= 12.0
    assert abs(seg.words[0].start - 10.2) < 0.05  # DTW time 0.6 mapped from [0.5, 1.5] onto [10, 12]


def test_smooth_labels_absorbs_short_islands_and_snaps_to_sentence_end():
    from app.merge import smooth_labels

    w = _words("I will tell you this manager's name. Okay great.", 0, 0.3)
    # timing noise: "tell you" labelled B, and the change to B happens one word late
    labels = ["A", "A", "B", "B", "A", "A", "A", "A", "B"]
    out = smooth_labels(w, labels)
    assert out == ["A"] * 7 + ["B", "B"]


def test_smooth_labels_keeps_real_interjections():
    from app.merge import smooth_labels

    w = _words("We start Monday. Okay, noted. Then we test.", 0, 0.3)
    labels = ["A", "A", "A", "B", "B", "A", "A", "A"]  # "Okay, noted." is a real, complete reply
    assert smooth_labels(w, labels) == labels


def test_smooth_labels_absorbs_mid_sentence_fragment():
    from app.merge import smooth_labels

    w = _words("Thank you. I guess that's all the question from me.", 0, 0.3)
    labels = ["B", "B", "A", "A", "A", "B", "B", "B", "B", "B"]
    assert smooth_labels(w, labels) == ["B"] * 10
