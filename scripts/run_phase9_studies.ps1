# Phase 9 simulator studies, in dependency order (H2 replays phase9_rungs' R0 histories).
# Each is cached per seed under data/eval/<study>/, so a re-run resumes.
$ErrorActionPreference = "Continue"
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
foreach ($study in "phase9_rungs", "phase9_h2", "phase9_skip", "phase9_ablation") {
    Write-Output "=== $study started $(Get-Date -Format o) ==="
    & $python -m visualtaste.jobs.evaluate --study $study --seeds 20 --swipes 1000
    Write-Output "=== $study finished $(Get-Date -Format o) exit $LASTEXITCODE ==="
}
