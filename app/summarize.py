"""Meeting summary + action items with Claude (Anthropic API) or a local Ollama model."""

from __future__ import annotations

import logging

import httpx

from . import config, storage
from .events import bus
from .export import to_txt

log = logging.getLogger(__name__)

SYSTEM = """You write concise, accurate meeting notes from a transcript.
The transcript comes from automatic speech recognition: expect some misheard words, and the
meeting may mix Indonesian and English. Speaker names are as labelled; do not invent names.
Only state what the transcript supports. If something is unclear, say so briefly."""

INSTRUCTIONS = """Write the meeting notes in {language}, in Markdown, with exactly these sections:

## Summary
3-6 sentences on what the meeting was about and where it ended up.

## Key points
Bullet list of the important points discussed.

## Decisions
Bullet list of decisions made (write "None recorded." if there were none).

## Action items
Bullet list in the form "- [ ] **Owner**: task (deadline if mentioned)". Use "Unassigned" when no owner was named.

## Open questions
Bullet list of unresolved questions (omit the section if there are none).

<transcript>
{transcript}
</transcript>"""

# Ollama models have small context windows: summarize long meetings in parts, then combine.
OLLAMA_CHUNK_CHARS = 12000

FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1"}


class SummaryError(RuntimeError):
    pass


def tidy(md: str) -> str:
    """Drop headings with nothing under them (models often keep an empty optional section)."""
    lines = md.strip().splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            rest = next((l for l in lines[i + 1 :] if l.strip()), "")
            if not rest or rest.lstrip().startswith("#"):
                continue
        out.append(line)
    return "\n".join(out).strip()


def _claude(prompt: str) -> str:
    import anthropic

    key = config.anthropic_api_key()
    if not key:
        raise SummaryError("No Anthropic API key. Add one in Settings or set ANTHROPIC_API_KEY.")
    client = anthropic.Anthropic(api_key=key)
    s = config.load_settings()
    kwargs = {}
    if s.claude_model in FALLBACK_MODELS:
        # if a safety classifier declines, the API re-runs the request on a suitable model
        kwargs = {"betas": ["server-side-fallback-2026-07-01"], "extra_body": {"fallbacks": "default"}}
    try:
        with client.beta.messages.stream(
            model=s.claude_model,
            max_tokens=16000,
            system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        ) as stream:
            msg = stream.get_final_message()
    except anthropic.AuthenticationError as e:
        raise SummaryError("The Anthropic API key was rejected.") from e
    except anthropic.NotFoundError as e:
        raise SummaryError(f"Model {s.claude_model!r} was not found.") from e
    except anthropic.RateLimitError as e:
        raise SummaryError("Anthropic rate limit reached; try again in a minute.") from e
    except anthropic.APIStatusError as e:
        raise SummaryError(f"Anthropic API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise SummaryError("Could not reach the Anthropic API (network).") from e
    if msg.stop_reason == "refusal":
        raise SummaryError("The model declined to summarize this transcript.")
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def _ollama(prompt: str, system: str = SYSTEM) -> str:
    s = config.load_settings()
    if not s.ollama_model:
        raise SummaryError("No Ollama model selected in Settings.")
    try:
        r = httpx.post(
            s.ollama_url.rstrip("/") + "/api/chat",
            json={
                "model": s.ollama_model,
                "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "options": {"num_ctx": 16384},
                "keep_alive": 0,  # unload right after: the GPU is needed for transcription
            },
            timeout=1800,
        )
    except httpx.HTTPError as e:
        raise SummaryError(f"Could not reach Ollama at {s.ollama_url}: {e}") from e
    if r.status_code != 200:
        raise SummaryError(f"Ollama error {r.status_code}: {r.text[:300]}")
    return r.json()["message"]["content"].strip()


def ollama_models() -> list[str]:
    s = config.load_settings()
    try:
        r = httpx.get(s.ollama_url.rstrip("/") + "/api/tags", timeout=3)
        return sorted(m["name"] for m in r.json().get("models", []))
    except (httpx.HTTPError, ValueError, KeyError):
        return []


def _split(text: str, size: int) -> list[str]:
    parts, cur = [], []
    n = 0
    for line in text.splitlines():
        if n + len(line) > size and cur:
            parts.append("\n".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += len(line) + 1
    if cur:
        parts.append("\n".join(cur))
    return parts


def summarize_meeting(mid: str) -> str:
    meta = storage.read_meta(mid)
    tr = storage.read_transcript(mid)
    if not tr or not tr["segments"]:
        raise SummaryError("This meeting has no transcript yet.")
    s = config.load_settings()
    transcript = to_txt(meta, tr["segments"])
    storage.update_meta(mid, summary_status="running", summary_error=None)
    bus.publish({"type": "summary", "meeting": mid, "status": "running"})
    try:
        if s.summary_provider == "claude":
            text = _claude(INSTRUCTIONS.format(language=s.summary_language, transcript=transcript))
        elif s.summary_provider == "ollama":
            parts = _split(transcript, OLLAMA_CHUNK_CHARS)
            if len(parts) == 1:
                text = _ollama(INSTRUCTIONS.format(language=s.summary_language, transcript=transcript))
            else:
                notes = [
                    _ollama(f"Write detailed bullet-point notes (who said what, decisions, tasks, owners) "
                            f"for this part {i + 1}/{len(parts)} of a meeting transcript:\n\n{p}")
                    for i, p in enumerate(parts)
                ]
                combined = "\n\n".join(f"### Part {i + 1}\n{n}" for i, n in enumerate(notes))
                text = _ollama(INSTRUCTIONS.format(language=s.summary_language, transcript=combined)
                               .replace("<transcript>", "<notes from consecutive parts of the meeting>")
                               .replace("</transcript>", "</notes>"))
        else:
            raise SummaryError("No summary provider selected in Settings.")
    except Exception as e:
        storage.update_meta(mid, summary_status="error", summary_error=str(e))
        bus.publish({"type": "summary", "meeting": mid, "status": "error", "error": str(e)})
        raise
    text = tidy(text)
    storage.write_summary(mid, text)
    storage.update_meta(mid, summary_status="done", summary_error=None)
    bus.publish({"type": "summary", "meeting": mid, "status": "done"})
    return text
