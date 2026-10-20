#!/usr/bin/env bash
# Download the models on first start (they live in the /app/models volume), then run the app.
set -euo pipefail
cd /app

FINAL="${MT_WHISPER_MODEL:-large-v3-turbo}"
LIVE="${MT_LIVE_MODEL:-}"
scripts/download_models.sh "$FINAL" ${LIVE:+"$LIVE"}

if [ ! -f data/settings.json ]; then
  mkdir -p data
  .venv/bin/python - "$FINAL" "$LIVE" <<'EOF'
import sys
from app.config import save_settings
final, live = sys.argv[1:3]
save_settings({"whisper_model": f"ggml-{final}.bin", "live_model": f"ggml-{live}.bin" if live else ""})
EOF
fi

exec .venv/bin/python -m app --host "${MT_HOST:-0.0.0.0}" --port "${MT_PORT:-8765}"
