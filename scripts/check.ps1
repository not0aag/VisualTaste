# Run every check the build gates on. Must be green at every phase commit.
#
#   .\scripts\check.ps1            everything that exists so far
#   .\scripts\check.ps1 -Python    Python only (fast loop while writing backend code)
#
# Exits non-zero if anything fails, so it works as a pre-commit gate.

param(
    [switch]$Python,
    [switch]$Frontend,
    [switch]$SkipPlaywright
)

$ErrorActionPreference = 'Continue'
$repo = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repo '.venv\Scripts\python.exe'
$failures = @()

function Invoke-Check {
    param([string]$Name, [scriptblock]$Body)

    Write-Host ''
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) {
        $script:failures += $Name
        Write-Host "FAILED: $Name" -ForegroundColor Red
    }
}

$runPython = $Python -or (-not $Frontend)
$runFrontend = $Frontend -or (-not $Python)

if ($runPython) {
    if (-not (Test-Path $venvPython)) {
        Write-Host "No venv at $venvPython -- see README setup." -ForegroundColor Red
        exit 1
    }

    Invoke-Check 'ruff check' { & $venvPython -m ruff check $repo }
    Invoke-Check 'ruff format --check' { & $venvPython -m ruff format --check $repo }
    Invoke-Check 'pytest' { & $venvPython -m pytest -q }
}

if ($runFrontend) {
    $frontendDir = Join-Path $repo 'frontend'
    # The frontend does not exist until Phase 6; skip quietly before then.
    if (Test-Path (Join-Path $frontendDir 'package.json')) {
        Push-Location $frontendDir
        try {
            Invoke-Check 'tsc --noEmit' { & pnpm exec tsc --noEmit }
            Invoke-Check 'vitest' { & pnpm exec vitest run }
            Invoke-Check 'vite build' { & pnpm run build }
            if (-not $SkipPlaywright) {
                Invoke-Check 'playwright' { & pnpm exec playwright test }
            }
        }
        finally {
            Pop-Location
        }
    }
    else {
        Write-Host ''
        Write-Host 'Frontend not created yet (Phase 6) -- skipping.' -ForegroundColor DarkGray
    }
}

Write-Host ''
if ($failures.Count -gt 0) {
    Write-Host "RED: $($failures.Count) check(s) failed:" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}

Write-Host 'GREEN: all checks passed.' -ForegroundColor Green
exit 0
