# Run simulator studies in sequence against their own database copy, so several
# runners can work in parallel without contending for SQLite's write lock
# (docs/DECISIONS.md #22). Results are cached under data/eval/; re-running resumes.
#
#   .\scripts\run_studies.ps1 -Database sim_c -Studies "phase11_h4:20:1000", "phase12_logs:10:5000"
param(
    [Parameter(Mandatory)] [string] $Database,
    [Parameter(Mandatory)] [string[]] $Studies
)
$ErrorActionPreference = "Continue"
$root = Join-Path $PSScriptRoot ".."
$python = Join-Path $root ".venv\Scripts\python.exe"
$copy = Join-Path $root "data\db\$Database.db"
if (-not (Test-Path $copy)) {
    & $python -c "import sqlite3,sys; s=sqlite3.connect(sys.argv[1]); d=sqlite3.connect(sys.argv[2]); s.backup(d)" (Join-Path $root "data\db\visualtaste.db") $copy
}
$env:VT_DB_FILE = $copy
foreach ($item in $Studies) {
    $name, $seeds, $swipes = $item.Split(":")
    Write-Output "=== $name started $(Get-Date -Format o) ==="
    & $python -m visualtaste.jobs.evaluate --study $name --seeds $seeds --swipes $swipes
    Write-Output "=== $name finished $(Get-Date -Format o) exit $LASTEXITCODE ==="
}
