# Shortcuts for every platform: https://github.com/casey/just
# On Windows `just` runs the PowerShell scripts; elsewhere the shell ones.

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

python := if os_family() == "windows" { ".venv\\Scripts\\python.exe" } else { ".venv/bin/python" }

# List the recipes.
default:
    @just --list

# API on 127.0.0.1:8000 and the app on 127.0.0.1:5173.
[unix]
dev:
    ./scripts/dev.sh

[windows]
dev:
    .\scripts\dev.ps1

# Every check the build gates on.
[unix]
check:
    ./scripts/check.sh

[windows]
check:
    .\scripts\check.ps1

# Backend tests only.
test:
    {{python}} -m pytest -q -m "not gpu"

# Run one simulator study, e.g. `just study phase10_h3 20`.
study name seeds="20":
    {{python}} -m visualtaste.jobs.evaluate --study {{name}} --seeds {{seeds}}

# Re-render the README's charts and photo composites from the caches.
figures:
    {{python}} scripts/readme_charts.py
    {{python}} scripts/readme_strips.py
