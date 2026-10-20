#!/usr/bin/env bash
# Start Meeting Transcriber.   ./run.sh [--open] [--host 127.0.0.1] [--port 8765]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PATH="$HOME/.local/bin:$PATH"
# optional .env (HF_TOKEN, ANTHROPIC_API_KEY, MT_PASSWORD, MT_PORT, ...)
if [ -f .env ]; then set -a; . ./.env; set +a; fi

OPEN=0; ARGS=()
for a in "$@"; do
  if [ "$a" = "--open" ]; then OPEN=1; else ARGS+=("$a"); fi
done

PORT="${MT_PORT:-8765}"
for ((i = 0; i < ${#ARGS[@]}; i++)); do
  [ "${ARGS[$i]}" = "--port" ] && PORT="${ARGS[$((i + 1))]}"
done

# already running? just open it
if curl -fs "http://127.0.0.1:$PORT/api/status" >/dev/null 2>&1; then
  echo "Already running on port $PORT"
  [ "$OPEN" = 1 ] && xdg-open "http://localhost:$PORT" >/dev/null 2>&1
  exit 0
fi

if [ "$OPEN" = 1 ]; then
  ( for _ in $(seq 1 60); do
      sleep 1
      curl -fs "http://127.0.0.1:$PORT/api/status" >/dev/null 2>&1 && { xdg-open "http://localhost:$PORT" >/dev/null 2>&1; break; }
    done ) &
fi
exec uv run --no-dev python -m app "${ARGS[@]}"
