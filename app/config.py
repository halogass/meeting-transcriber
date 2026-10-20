"""Paths, persisted settings and hardware detection."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MT_DATA_DIR", ROOT / "data")).resolve()
MEETINGS_DIR = DATA_DIR / "meetings"
SETTINGS_FILE = DATA_DIR / "settings.json"
MODELS_DIR = Path(os.environ.get("MT_MODELS_DIR", ROOT / "models")).resolve()
BIN_DIR = Path(os.environ.get("MT_BIN_DIR", ROOT / "bin")).resolve()
STATIC_DIR = ROOT / "app" / "static"

SAMPLE_RATE = 16000  # whisper.cpp and pyannote both work at 16 kHz mono

SECRET_FIELDS = ("hf_token", "anthropic_api_key")


@dataclass
class Settings:
    # Identity
    user_name: str = "Me"

    # Audio sources (PulseAudio/PipeWire source names; "" = system default)
    mic_source: str = ""
    call_source: str = ""  # "" = monitor of the default output device

    # Transcription
    language: str = "auto"
    whisper_model: str = "ggml-large-v3-turbo-q5_0.bin"
    live_model: str = ""  # "" = same as whisper_model (one server)
    whisper_device: str = "auto"  # "auto" | "cpu" | GPU index such as "0"
    whisper_threads: int = 4
    vad_model: str = "ggml-silero-v5.1.2.bin"

    # Diarization
    hf_token: str = ""
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    diarization_device: str = "auto"  # "auto" | "cpu" | GPU index
    min_speakers: int = 0  # 0 = let pyannote decide
    max_speakers: int = 0
    echo_dedup: bool = True  # drop mic segments that are just the call audio leaking into the mic

    # Summary
    summary_provider: str = "none"  # "claude" | "ollama" | "none"
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-5-5"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = ""
    summary_language: str = "English"
    auto_summary: bool = True  # summarize when a meeting finishes processing (if a provider is set)

    # Auto-detect calls
    auto_detect_calls: bool = False
    auto_detect_apps: list[str] = field(
        default_factory=lambda: ["zoom", "chrome", "chromium", "firefox", "teams", "slack", "discord", "skype", "webex", "brave", "edge"]
    )
    auto_stop_silence_s: int = 60

    def public(self) -> dict:
        """Settings safe to send to the browser: secrets masked."""
        d = asdict(self)
        for k in SECRET_FIELDS:
            v = d[k]
            d[k] = ("•" * 8 + v[-4:]) if v else ""
            d[k + "_set"] = bool(v)
        return d


_lock = threading.Lock()
_settings: Settings | None = None


def load_settings() -> Settings:
    global _settings
    with _lock:
        if _settings is None:
            data = {}
            if SETTINGS_FILE.exists():
                try:
                    data = json.loads(SETTINGS_FILE.read_text())
                except json.JSONDecodeError:
                    data = {}
            known = {f.name for f in fields(Settings)}
            _settings = Settings(**{k: v for k, v in data.items() if k in known})
        return _settings


def save_settings(update: dict) -> Settings:
    """Merge `update` into the settings. Masked secret values (unchanged in the UI) are ignored."""
    global _settings
    current = asdict(load_settings())
    known = {f.name: f for f in fields(Settings)}
    for k, v in update.items():
        if k not in known:
            continue
        if k in SECRET_FIELDS and isinstance(v, str) and v.startswith("•"):
            continue
        default = current[k]
        if isinstance(default, bool):
            v = bool(v)
        elif isinstance(default, int):
            v = int(v or 0)
        elif isinstance(default, list) and isinstance(v, str):
            v = [s.strip() for s in v.split(",") if s.strip()]
        current[k] = v
    with _lock:
        _settings = Settings(**current)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(current, indent=2))
        os.chmod(tmp, 0o600)  # holds API keys
        tmp.replace(SETTINGS_FILE)
    return _settings


_WATCHDOG = r"""
"$@" &
child=$!
trap 'kill $child 2>/dev/null; wait $child; exit' TERM INT
while kill -0 "$child" 2>/dev/null; do
  kill -0 "$MT_PARENT_PID" 2>/dev/null || { kill "$child"; break; }
  sleep 1
done
wait "$child"
"""


def supervised(cmd: list[str]) -> list[str]:
    """Wrap a child command so it is stopped when the app process goes away, even after
    `kill -9` or a crash: a GPU-holding whisper-server must never outlive the app.
    Terminating the wrapper (Popen.terminate) also stops the child. Needs MT_PARENT_PID."""
    os.environ["MT_PARENT_PID"] = str(os.getpid())
    return ["bash", "-c", _WATCHDOG, "watchdog", *cmd]


def hf_token() -> str | None:
    s = load_settings()
    if s.hf_token:
        return s.hf_token
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    p = Path.home() / ".cache" / "huggingface" / "token"
    return p.read_text().strip() if p.exists() else None


def anthropic_api_key() -> str | None:
    return load_settings().anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY") or None


@lru_cache(maxsize=1)
def gpu_count() -> int:
    if os.environ.get("MT_FORCE_CPU"):
        return 0
    if not shutil.which("nvidia-smi"):
        return 0
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5).stdout
        return sum(1 for line in out.splitlines() if line.startswith("GPU "))
    except (OSError, subprocess.SubprocessError):
        return 0


def whisper_gpu() -> int | None:
    """GPU index for whisper.cpp, or None for CPU."""
    dev = load_settings().whisper_device
    if dev == "cpu" or gpu_count() == 0:
        return None
    if dev.isdigit():
        return int(dev)
    return 0


def diarization_torch_device() -> str:
    """torch device string for pyannote. With two GPUs, pyannote gets the one whisper isn't using."""
    dev = load_settings().diarization_device
    n = gpu_count()
    if dev == "cpu" or n == 0:
        return "cpu"
    if dev.isdigit():
        return f"cuda:{dev}"
    w = whisper_gpu()
    if n >= 2:
        return f"cuda:{1 if w in (0, None) else 0}"
    return "cuda:0"


def model_path(name: str) -> Path:
    return MODELS_DIR / name


def available_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(p.name for p in MODELS_DIR.glob("ggml-*.bin") if "silero" not in p.name)
