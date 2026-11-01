# Fill the phase14_final cache for some variants, on its own database copy
# (docs/DECISIONS.md #22). A run without -Variants afterwards writes the report.
#
#   .\scripts\run_final_split.ps1 -Database sim_f1 -Variants "S1_semantic,S2_appearance"
param(
    [Parameter(Mandatory)] [string] $Database,
    [Parameter(Mandatory)] [string] $Variants,
    [int] $Seeds = 20,
    [int] $Swipes = 1000
)
$ErrorActionPreference = "Continue"
$root = Join-Path $PSScriptRoot ".."
$python = Join-Path $root ".venv\Scripts\python.exe"
$copy = Join-Path $root "data\db\$Database.db"
if (-not (Test-Path $copy)) {
    & $python -c "import sqlite3,sys; s=sqlite3.connect(sys.argv[1]); d=sqlite3.connect(sys.argv[2]); s.backup(d)" (Join-Path $root "data\db\visualtaste.db") $copy
}
$env:VT_DB_FILE = $copy
Write-Output "=== phase14_final [$Variants] started $(Get-Date -Format o) ==="
& $python -m visualtaste.jobs.evaluate --study phase14_final --variants $Variants --seeds $Seeds --swipes $Swipes
Write-Output "=== phase14_final [$Variants] finished $(Get-Date -Format o) exit $LASTEXITCODE ==="
