#!/usr/bin/env bash
# Download whisper.cpp GGML models and the Silero VAD model.
#
#   scripts/download_models.sh                       # large-v3-turbo-q5_0 (default)
#   scripts/download_models.sh large-v3-turbo-q5_0 small
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/models"
mkdir -p "$DIR"

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=(large-v3-turbo-q5_0)

fetch() {
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then echo ">> $(basename "$dest") already present"; return; fi
  echo ">> Downloading $(basename "$dest")"
  curl -fL --retry 3 -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

for m in "${MODELS[@]}"; do
  fetch "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$m.bin" "$DIR/ggml-$m.bin"
done

# Silero VAD for whisper.cpp (skips silence, reduces hallucinations on long recordings)
fetch "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin" "$DIR/ggml-silero-v5.1.2.bin"
