# Phase 10 simulator studies, on their own database copy so they can run
# alongside the Phase 9 ones (VT_DB_FILE; docs/DECISIONS.md #22).
$ErrorActionPreference = "Continue"
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$env:VT_DB_FILE = (Join-Path $PSScriptRoot "..\data\db\sim_studies_b.db")
foreach ($run in @(@("phase10_h3", 20), @("phase10_checks", 20), @("phase10_sensitivity", 10))) {
    Write-Output "=== $($run[0]) started $(Get-Date -Format o) ==="
    & $python -m visualtaste.jobs.evaluate --study $run[0] --seeds $run[1] --swipes 1000
    Write-Output "=== $($run[0]) finished $(Get-Date -Format o) exit $LASTEXITCODE ==="
}
