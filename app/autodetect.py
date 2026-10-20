"""Optional: start recording when a call app opens the microphone, stop when it ends.

A call app (Zoom, a browser on Meet/Teams, ...) records from the microphone for the whole
call. We watch PulseAudio/PipeWire "source outputs" (apps capturing audio) for those apps.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from datetime import datetime

from . import config
from .audio import CLIENT_NAME, capture_available

log = logging.getLogger(__name__)

POLL_S = 3
START_AFTER_S = 6  # app must hold the mic this long (ignores quick mic tests)
STOP_AFTER_S = 10  # app released the mic this long ago


def call_apps() -> list[str]:
    """Names of matching apps that currently capture audio (excluding ourselves)."""
    try:
        out = subprocess.run(["pactl", "-f", "json", "list", "source-outputs"], capture_output=True, text=True, timeout=3)
        outputs = json.loads(out.stdout or "[]")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    wanted = [a.lower() for a in config.load_settings().auto_detect_apps]
    found = []
    for o in outputs:
        props = o.get("properties", {})
        name = (props.get("application.name") or "").lower()
        binary = (props.get("application.process.binary") or "").lower()
        if CLIENT_NAME in name or not (name or binary):
            continue
        if props.get("media.name", "").lower() == "peak detect":  # pavucontrol meters
            continue
        if any(w in name or w in binary for w in wanted):
            found.append(props.get("application.name") or binary)
    return found


class AutoDetector:
    def __init__(self, start_cb, stop_cb, is_recording, silent_for):
        self.start_cb = start_cb  # (title) -> meeting id
        self.stop_cb = stop_cb  # () -> None
        self.is_recording = is_recording  # () -> bool
        self.silent_for = silent_for  # () -> seconds since anyone last spoke
        self.auto_started: str | None = None
        self.suppressed = False  # stopped during a call (manually or by silence): wait for the call to end
        self.state = "idle"
        self._thread = threading.Thread(target=self._run, name="autodetect", daemon=True)

    def start(self) -> None:
        if capture_available():
            self._thread.start()

    def _run(self) -> None:
        seen_since = None
        gone_since = None
        while True:
            time.sleep(POLL_S)
            s = config.load_settings()
            if not s.auto_detect_calls:
                self.state = "off"
                seen_since = gone_since = None
                continue
            apps = call_apps()
            now = time.time()
            try:
                if self.auto_started and self.is_recording() and self.silent_for() >= s.auto_stop_silence_s:
                    log.info("no speech for %ss: auto-stopping", s.auto_stop_silence_s)
                    self.auto_started = None
                    self.suppressed = True
                    self.stop_cb()
                    continue
                if apps:
                    gone_since = None
                    seen_since = seen_since or now
                    self.state = f"call detected ({apps[0]})"
                    if self.auto_started and not self.is_recording():
                        self.auto_started = None  # the user pressed Stop
                        self.suppressed = True
                    if not self.suppressed and not self.is_recording() and now - seen_since >= START_AFTER_S:
                        title = f"{apps[0].title()} call {datetime.now():%d %b %H:%M}"
                        self.auto_started = self.start_cb(title)
                        log.info("auto-started recording for %s", apps[0])
                else:
                    seen_since = None
                    self.suppressed = False
                    self.state = "watching"
                    if self.auto_started and self.is_recording():
                        gone_since = gone_since or now
                        if now - gone_since >= STOP_AFTER_S:
                            log.info("call ended: auto-stopping")
                            self.auto_started = None
                            self.stop_cb()
                    elif not self.is_recording():
                        self.auto_started = None
            except Exception:
                log.exception("auto-detect step failed")
