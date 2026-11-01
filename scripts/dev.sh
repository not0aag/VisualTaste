#!/usr/bin/env bash
# Start the API (127.0.0.1:8000) and the frontend (127.0.0.1:5173) together.
# Linux and macOS counterpart of scripts/dev.ps1. Ctrl+C stops both.
#
#   ./scripts/dev.sh              # API + frontend
#   ./scripts/dev.sh --backend    # API only
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
python="$repo/.venv/bin/python"

if [[ ! -x "$python" ]]; then
  echo "No venv at $python -- see the README's Quick start." >&2
  exit 1
fi

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "Starting the API on http://127.0.0.1:8000 ..."
(cd "$repo" && "$python" -m uvicorn visualtaste.api.main:app --host 127.0.0.1 --port 8000) &
pids+=($!)

if [[ "${1:-}" != "--backend" ]]; then
  if [[ ! -d "$repo/frontend/node_modules" ]]; then
    echo "Installing frontend dependencies (first run only) ..."
    (cd "$repo/frontend" && pnpm install)
  fi
  echo "Starting the frontend on http://127.0.0.1:5173 ..."
  (cd "$repo/frontend" && pnpm run dev) &
  pids+=($!)
fi

wait
