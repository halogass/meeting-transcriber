#!/usr/bin/env bash
# Meeting Transcriber installer for Linux (Ubuntu/Debian; other distros: install the
# packages listed below with your package manager first).
#
#   ./install.sh            interactive
#   ./install.sh --yes      accept defaults (no questions)
#   ./install.sh --cpu      force a CPU-only build even if an NVIDIA GPU is present
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
YES=0; FORCE_CPU=""
for a in "$@"; do
  case "$a" in
    --yes|-y) YES=1 ;;
    --cpu) FORCE_CPU=1 ;;
    -h|--help) sed -n 2,9p "$0"; exit 0 ;;
  esac
done

bold() { printf '\n\033[1m%s\033[0m\n' "$*"; }
ask() { # ask "question" default(y/n)
  local q="$1" def="${2:-y}" ans
  if [ "$YES" = 1 ]; then [ "$def" = y ]; return; fi
  read -r -p "$q [$( [ "$def" = y ] && echo Y/n || echo y/N )] " ans || true
  ans="${ans:-$def}"; [[ "$ans" =~ ^[Yy] ]]
}

[ "$(uname -s)" = Linux ] || { echo "This installer supports Linux only."; exit 1; }

# ---------------------------------------------------------------- system packages
bold "1/5  System packages"
need=()
command -v git >/dev/null || need+=(git)
command -v cmake >/dev/null || need+=(cmake)
command -v g++ >/dev/null || need+=(build-essential)
command -v ffmpeg >/dev/null || need+=(ffmpeg)
command -v parec >/dev/null || need+=(pulseaudio-utils)
command -v curl >/dev/null || need+=(curl)
ldconfig -p 2>/dev/null | grep -q libsndfile || need+=(libsndfile1)
if [ ${#need[@]} -gt 0 ]; then
  echo "Missing: ${need[*]}"
  if command -v apt-get >/dev/null && ask "Install them with apt (needs sudo)?"; then
    sudo apt-get update && sudo apt-get install -y "${need[@]}"
  else
    echo "Please install: ${need[*]}  and run this script again."; exit 1
  fi
else
  echo "All present (git, cmake, compiler, ffmpeg, parec, libsndfile)."
fi
if ! pactl info >/dev/null 2>&1; then
  echo "Note: no PulseAudio/PipeWire session found. Live recording needs one; uploads still work."
fi

# ---------------------------------------------------------------- GPU
GPU=0
if [ -z "$FORCE_CPU" ] && command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  GPU=1
  echo "NVIDIA GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd', ')"
  if ! command -v nvcc >/dev/null; then
    echo "The CUDA toolkit (nvcc) is needed to build whisper.cpp for the GPU."
    if command -v apt-get >/dev/null && ask "Install nvidia-cuda-toolkit with apt (needs sudo, ~3 GB)?"; then
      sudo apt-get install -y nvidia-cuda-toolkit
    else
      echo "Continuing with a CPU build of whisper.cpp (pyannote will still use the GPU)."
    fi
  fi
fi

# ---------------------------------------------------------------- whisper.cpp
bold "2/5  Building whisper.cpp"
if [ -x bin/whisper-server ] && ! ask "whisper.cpp is already built. Rebuild?" n; then
  echo "Keeping existing build."
else
  FORCE_CPU="$FORCE_CPU" scripts/build_whisper.sh
fi

# ---------------------------------------------------------------- models
bold "3/5  Downloading Whisper models"
# large-v3-turbo quantized to 5 bits: same transcripts as the full model in our tests,
# ~2.5x faster and a third of the memory (the full model can still be downloaded and picked in Settings)
FINAL=large-v3-turbo-q5_0
if [ "$GPU" = 1 ] && command -v nvcc >/dev/null; then
  LIVE=""
else
  LIVE=small-q5_1  # CPU: a small model keeps live captions fast
fi
scripts/download_models.sh "$FINAL" ${LIVE:+"$LIVE"}

# ---------------------------------------------------------------- python
bold "4/5  Python environment"
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  if ask "Install uv (Python package manager from astral.sh)?"; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  else
    echo "uv is required."; exit 1
  fi
fi
export PATH="$HOME/.local/bin:$PATH"
uv sync --no-dev

# ---------------------------------------------------------------- settings
bold "5/5  Settings"
mkdir -p data
if [ ! -f data/settings.json ]; then
  NAME="$(getent passwd "$USER" | cut -d: -f5 | cut -d, -f1)"; NAME="${NAME:-$USER}"
  HF=""
  if [ -z "${HF_TOKEN:-}" ] && [ "$YES" = 0 ]; then
    echo "Speaker identification uses pyannote, which needs a free Hugging Face token:"
    echo "  1. https://huggingface.co/settings/tokens  (create a token of type 'Read')"
    echo "  2. https://huggingface.co/pyannote/speaker-diarization-community-1  (accept the terms)"
    read -r -p "Paste the token (or press Enter to add it later in Settings): " HF || true
  fi
  uv run python - "$NAME" "$FINAL" "${LIVE:+ggml-$LIVE.bin}" "$HF" <<'EOF'
import sys
from app.config import save_settings
name, final, live, hf = sys.argv[1:5]
update = {"user_name": name, "whisper_model": f"ggml-{final}.bin", "live_model": live}
if hf.strip():
    update["hf_token"] = hf.strip()
save_settings(update)
print("Saved data/settings.json")
EOF
else
  echo "Keeping existing data/settings.json"
fi

# desktop launcher
if [ -d "$HOME/.local/share/applications" ] || mkdir -p "$HOME/.local/share/applications" 2>/dev/null; then
  cat > "$HOME/.local/share/applications/meeting-transcriber.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Meeting Transcriber
Comment=Record and transcribe meetings with speaker names
Exec=$ROOT/run.sh --open
Icon=audio-input-microphone
Terminal=false
Categories=AudioVideo;Office;
EOF
fi

bold "Done."
echo "Start it with:   ./run.sh --open      (or 'Meeting Transcriber' in your app menu)"
