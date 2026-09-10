#!/usr/bin/env bash
# Start the web app with exactly the configuration .env describes.
#
# `set -a; . ./.env` alone is not enough. It can only *set* variables, so
# anything already exported in the shell survives - and a leftover
# OPENAI_API_KEY or ANTHROPIC_WORKSPACE_ID from an earlier session then
# silently overrides the file, which is how a commented-out workspace id kept
# reaching the API long after it was removed. Every variable the app reads is
# therefore cleared first, so the file is the only source of truth.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "error: no .env file. Copy .env.example to .env and add your key." >&2
    exit 1
fi

unset OPENAI_API_KEY ANTHROPIC_API_KEY ANTHROPIC_WORKSPACE_ID \
      OPENROUTER_API_KEY OPENROUTER_MODEL OPENROUTER_BASE_URL \
      TRANSLATION_PROVIDER TRANSLATION_MODEL VISION_LAYOUT \
      VISION_LAYOUT_PROVIDER VISION_LAYOUT_MODEL \
      GOOGLE_TRANSLATE_API_KEY DEEPL_API_KEY

set -a
# shellcheck disable=SC1091
. ./.env
set +a

# Stop an older server first: it holds the environment it was started with,
# so leaving it running means the edit you just made has no effect.
if pgrep -f 'uvicorn app.web.main' >/dev/null; then
    echo "Stopping the running server..."
    pkill -f 'uvicorn app.web.main' || true
    for _ in $(seq 20); do
        pgrep -f 'uvicorn app.web.main' >/dev/null || break
        sleep 0.25
    done
fi

.venv/bin/python - <<'PY'
"""Report what the app will actually use, before it starts serving."""
from app.core import translate, vision_layout

print(f"  translation : {translate.provider_name()}")
print(f"  layout      : {vision_layout.provider_name()}"
      f" ({'on' if vision_layout.is_available() else 'unavailable'})")
PY

echo "Starting on http://127.0.0.1:8000"
exec .venv/bin/uvicorn app.web.main:app --port "${PORT:-8000}"
