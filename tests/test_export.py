import io

from app import export

META = {"title": "Weekly sync", "created_at": "2026-10-20T13:00:00+07:00", "duration": 3725, "speakers": {"Me": "Hilmy", "Speaker 1": "Budi"}}
SEGS = [
    {"start": 0.0, "end": 2.5, "speaker": "Speaker 1", "text": "Selamat pagi."},
    {"start": 3661.2, "end": 3663.0, "speaker": "Me", "text": "Good morning."},
]


def test_ts():
    assert export.ts(65.4) == "01:05"
    assert export.ts(3661.2) == "1:01:01"
    assert export.ts(3661.2, srt=True) == "01:01:01,200"


def test_txt_uses_display_names():
    txt = export.to_txt(META, SEGS)
    assert "[00:00] Budi: Selamat pagi." in txt
    assert "[1:01:01] Hilmy: Good morning." in txt


def test_srt():
    srt = export.to_srt(META, SEGS)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,500\nBudi: Selamat pagi.")


def test_md_with_summary():
    md = export.to_md(META, SEGS, "## Summary\nAll good.")
    assert md.startswith("# Weekly sync")
    assert "## Transcript" in md and "**Hilmy**" in md


def test_docx_opens():
    from docx import Document

    data = export.to_docx(META, SEGS, "## Action items\n- [ ] **Budi**: send report")
    doc = Document(io.BytesIO(data))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "Weekly sync" in text and "Selamat pagi." in text and "send report" in text
