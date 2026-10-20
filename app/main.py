"""FastAPI app: REST API + WebSocket events + the web UI."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import secrets
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__, config, export, storage
from .audio import capture_available, convert_to_wav16k, list_sources
from .autodetect import AutoDetector
from .events import bus
from .live import Recording, finalize_raw, recorder
from .pipeline import jobs
from .whisper import pool

log = logging.getLogger("meeting-transcriber")

PASSWORD = os.environ.get("MT_PASSWORD", "")


# ---------------------------------------------------------------- recording control


def start_recording(title: str, mic_source: str | None = None, call_source: str | None = None, use_mic: bool = True) -> str:
    if not capture_available():
        raise HTTPException(400, "Audio capture is not available on this machine (no PulseAudio/PipeWire). Use Upload instead.")
    s = config.load_settings()
    with recorder.lock:
        if recorder.current:
            raise HTTPException(409, "A recording is already running")
        meta = storage.create(title, "live", s.user_name)
        rec = Recording(
            meta,
            s.mic_source if mic_source is None else mic_source,
            s.call_source if call_source is None else call_source,
            use_mic=use_mic,
        )
        rec.start()
        recorder.current = rec
    bus.publish({"type": "status", "meeting": meta["id"], "status": "recording"})
    return meta["id"]


def stop_recording() -> dict:
    with recorder.lock:
        rec = recorder.current
        if not rec:
            raise HTTPException(409, "Nothing is recording")
        recorder.current = None
    meta = rec.stop()
    jobs.submit(rec.id)
    bus.publish({"type": "status", "meeting": rec.id, "status": "queued"})
    return meta


def _stop_quietly() -> None:
    try:
        stop_recording()
    except HTTPException:
        pass


autodetector = AutoDetector(
    start_cb=lambda title: start_recording(title),
    stop_cb=_stop_quietly,
    is_recording=lambda: recorder.current is not None,
    silent_for=lambda: time.time() - recorder.current.last_speech if recorder.current else 0.0,
)


def recover() -> None:
    """After a crash or restart: finish what was interrupted."""
    for meta in storage.list_meetings():
        mid, status = meta["id"], meta["status"]
        if status == "recording":
            log.warning("recovering interrupted recording %s", mid)
            finalize_raw(mid)
            storage.update_meta(mid, status="queued", error=None)
            jobs.submit(mid)
        elif status in ("queued", "processing"):
            jobs.submit(mid)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
    await run_in_threadpool(recover)
    autodetector.start()
    yield
    if recorder.current:
        await run_in_threadpool(_stop_quietly)
    pool.stop_all()


app = FastAPI(title="Meeting Transcriber", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    """Optional password (MT_PASSWORD) for when the app is reachable from other machines."""
    if PASSWORD:
        header = request.headers.get("authorization", "")
        ok = False
        if header.startswith("Basic "):
            try:
                _, _, pw = base64.b64decode(header[6:]).decode().partition(":")
                ok = secrets.compare_digest(pw, PASSWORD)
            except (ValueError, UnicodeDecodeError):
                ok = False
        if not ok:
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Meeting Transcriber"'})
    return await call_next(request)


# ---------------------------------------------------------------- API


@app.get("/api/status")
def status():
    rec = recorder.current
    s = config.load_settings()
    return {
        "version": __version__,
        "capture_available": capture_available(),
        "recording": {
            "id": rec.id, "title": rec.meta["title"], "elapsed": round(rec.elapsed(), 1),
            "streams": list(rec.captures), "errors": rec.errors(),
        } if rec else None,
        "processing": jobs.current,
        "queued": jobs.q.qsize(),
        "whisper": pool.status(),
        "gpus": config.gpu_count(),
        "diarization_device": config.diarization_torch_device(),
        "models": config.available_models(),
        "whisper_ready": (config.BIN_DIR / "whisper-server").exists() and config.model_path(s.whisper_model).exists(),
        "hf_token": bool(config.hf_token()),
        "summary_provider": s.summary_provider,
        "autodetect": autodetector.state if s.auto_detect_calls else "off",
    }


@app.get("/api/devices")
def devices():
    return list_sources()


@app.get("/api/settings")
def get_settings():
    return config.load_settings().public()


@app.put("/api/settings")
def put_settings(update: dict):
    return config.save_settings(update).public()


@app.get("/api/ollama/models")
def get_ollama_models():
    from .summarize import ollama_models

    return ollama_models()


class StartReq(BaseModel):
    title: str = ""
    mic_source: str | None = None
    call_source: str | None = None
    use_mic: bool = True


@app.post("/api/recordings")
def api_start(req: StartReq):
    mid = start_recording(req.title, req.mic_source, req.call_source, req.use_mic)
    return storage.read_meta(mid)


@app.post("/api/recordings/stop")
def api_stop():
    return stop_recording()


@app.get("/api/meetings")
def meetings(q: str = ""):
    return storage.list_meetings(q)


def _meta_or_404(mid: str) -> dict:
    try:
        return storage.read_meta(mid)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "Meeting not found")


@app.get("/api/meetings/{mid}")
def meeting(mid: str):
    meta = _meta_or_404(mid)
    d = storage.meeting_dir(mid)
    return {
        "meta": meta,
        "transcript": storage.read_transcript(mid),
        "live": storage.read_live(mid),
        "summary": storage.read_summary(mid),
        "has_audio": (d / "audio.opus").exists(),
    }


class MeetingPatch(BaseModel):
    title: str | None = None
    speakers: dict[str, str] | None = None


@app.patch("/api/meetings/{mid}")
def patch_meeting(mid: str, patch: MeetingPatch):
    meta = _meta_or_404(mid)
    changes = {}
    if patch.title is not None and patch.title.strip():
        changes["title"] = patch.title.strip()
    if patch.speakers is not None:
        speakers = dict(meta.get("speakers") or {})
        speakers.update({k: v.strip() or k for k, v in patch.speakers.items()})
        changes["speakers"] = speakers
    return storage.update_meta(mid, **changes)


class SegmentPatch(BaseModel):
    index: int
    text: str | None = None
    speaker: str | None = None


@app.patch("/api/meetings/{mid}/segments")
def patch_segment(mid: str, patch: SegmentPatch):
    """Fix a transcription mistake or reassign a paragraph to another speaker."""
    _meta_or_404(mid)
    tr = storage.read_transcript(mid)
    if not tr or not (0 <= patch.index < len(tr["segments"])):
        raise HTTPException(404, "Segment not found")
    seg = tr["segments"][patch.index]
    if patch.text is not None:
        seg["text"] = patch.text.strip()
    if patch.speaker:
        seg["speaker"] = patch.speaker
    storage.write_transcript(mid, tr)
    return seg


@app.delete("/api/meetings/{mid}")
def delete_meeting(mid: str):
    meta = _meta_or_404(mid)
    busy = meta["status"] in ("recording", "processing") or meta.get("summary_status") == "running"
    if busy or jobs.current == mid or (recorder.current and recorder.current.id == mid):
        raise HTTPException(409, "Meeting is still recording or processing")
    storage.delete(mid)
    return {"ok": True}


@app.post("/api/meetings/{mid}/reprocess")
def reprocess(mid: str):
    meta = _meta_or_404(mid)
    if meta["status"] in ("recording", "processing", "queued"):
        raise HTTPException(409, "Meeting is busy")
    storage.update_meta(mid, status="queued", progress=0.0, error=None)
    jobs.submit(mid)
    return {"ok": True}


@app.post("/api/meetings/{mid}/summary")
async def summary(mid: str):
    from .summarize import SummaryError, summarize_meeting

    _meta_or_404(mid)
    try:
        text = await run_in_threadpool(summarize_meeting, mid)
    except SummaryError as e:
        raise HTTPException(400, str(e))
    return {"summary": text}


@app.get("/api/meetings/{mid}/audio")
def audio(mid: str):
    _meta_or_404(mid)
    p = storage.meeting_dir(mid) / "audio.opus"
    if not p.exists():
        raise HTTPException(404, "No audio")
    return FileResponse(p, media_type="audio/ogg")


@app.get("/api/meetings/{mid}/export/{fmt}")
def export_meeting(mid: str, fmt: str, summary: bool = True):
    meta = _meta_or_404(mid)
    tr = storage.read_transcript(mid)
    if not tr:
        raise HTTPException(404, "No transcript yet")
    segs = tr["segments"]
    summ = storage.read_summary(mid) if summary else None
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in meta["title"]).strip() or mid
    if fmt == "txt":
        body = export.to_txt(meta, segs)
        if summ:
            body = summ.strip() + "\n\n---\n\n" + body
        data, mime = body.encode(), "text/plain; charset=utf-8"
    elif fmt == "md":
        data, mime = export.to_md(meta, segs, summ).encode(), "text/markdown; charset=utf-8"
    elif fmt == "srt":
        data, mime = export.to_srt(meta, segs).encode(), "application/x-subrip"
    elif fmt == "docx":
        data = export.to_docx(meta, segs, summ)
        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    else:
        raise HTTPException(400, "Unknown format")
    return Response(data, media_type=mime, headers={"Content-Disposition": f'attachment; filename="{safe}.{fmt}"'})


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), title: str = Form("")):
    """Transcribe an existing recording (all speakers diarized)."""
    s = config.load_settings()
    meta = storage.create(title or Path(file.filename or "Recording").stem, "upload", s.user_name)
    d = storage.meeting_dir(meta["id"])
    with tempfile.NamedTemporaryFile(dir=d, suffix=Path(file.filename or "").suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        src = Path(tmp.name)
    try:
        await run_in_threadpool(convert_to_wav16k, src, d / "upload.wav")
    except RuntimeError as e:
        storage.delete(meta["id"])
        raise HTTPException(400, f"Could not read that file: {e}")
    finally:
        src.unlink(missing_ok=True)
    import soundfile as sf

    info = sf.info(str(d / "upload.wav"))
    meta = storage.update_meta(meta["id"], duration=round(info.duration, 1))
    jobs.submit(meta["id"])
    return meta


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    if PASSWORD:
        # browsers send the Basic auth credentials they already used for the page
        header = websocket.headers.get("authorization", "")
        try:
            pw = base64.b64decode(header[6:]).decode().partition(":")[2] if header.startswith("Basic ") else ""
        except (ValueError, UnicodeDecodeError):
            pw = ""
        if not secrets.compare_digest(pw, PASSWORD):
            await websocket.close(code=4401)
            return
    await websocket.accept()
    q = bus.subscribe()

    async def pump():
        while True:
            await websocket.send_json(await q.get())

    async def watch():  # returns when the client goes away
        while True:
            msg = await websocket.receive()
            if msg["type"] == "websocket.disconnect":
                return

    tasks = [asyncio.create_task(pump()), asyncio.create_task(watch())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        for t in tasks:
            t.cancel()
        bus.unsubscribe(q)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse({"detail": str(exc)}, status_code=500)


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")
