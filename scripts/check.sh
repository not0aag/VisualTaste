#!/usr/bin/env bash
# Every check the build gates on: lint, backend tests, frontend types, unit
# tests and build, then the browser tests. Counterpart of scripts/check.ps1.
#
#   ./scripts/check.sh            # everything
#   ./scripts/check.sh --fast     # skip the Playwright run
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
python="$repo/.venv/bin/python"
cd "$repo"

step() { printf '\n== %s\n' "$1"; }

step "ruff check";          "$python" -m ruff check .
step "ruff format --check"; "$python" -m ruff format --check .
step "pytest";              "$python" -m pytest -q -m "not gpu"

cd frontend
step "typecheck";           pnpm typecheck
step "vitest";              pnpm test
step "build";               pnpm build
if [[ "${1:-}" != "--fast" ]]; then
  step "playwright";        pnpm e2e
fi

printf '\nAll checks passed.\n'
