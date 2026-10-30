# Start the backend and the frontend for a swiping session.
#
#   .\scripts\dev.ps1
#
# Opens two PowerShell windows (API on 127.0.0.1:8000, Vite on 127.0.0.1:5173)
# and leaves them running. Close the windows to stop. Nothing binds to anything
# but loopback, by design.

param(
    [switch]$BackendOnly,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repo '.venv\Scripts\python.exe'

if (-not (Test-Path $venvPython)) {
    Write-Host "No venv at $venvPython -- see README setup." -ForegroundColor Red
    exit 1
}

$apiMain = Join-Path $repo 'backend\visualtaste\api\main.py'
if (-not (Test-Path $apiMain)) {
    Write-Host 'The API does not exist yet (built in Phase 6).' -ForegroundColor Yellow
    exit 1
}

Write-Host 'Starting the API on http://127.0.0.1:8000 ...' -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    '-NoExit', '-Command',
    "Set-Location '$repo'; & '$venvPython' -m uvicorn visualtaste.api.main:app --host 127.0.0.1 --port 8000 --reload"
)

if (-not $BackendOnly) {
    $frontendDir = Join-Path $repo 'frontend'
    if (Test-Path (Join-Path $frontendDir 'package.json')) {
        if (-not (Test-Path (Join-Path $frontendDir 'node_modules'))) {
            Write-Host 'Installing frontend dependencies (first run only) ...' -ForegroundColor Cyan
            Push-Location $frontendDir
            try { & pnpm install } finally { Pop-Location }
        }

        Write-Host 'Starting the frontend on http://127.0.0.1:5173 ...' -ForegroundColor Cyan
        Start-Process powershell -ArgumentList @(
            '-NoExit', '-Command',
            "Set-Location '$frontendDir'; & pnpm run dev"
        )

        if (-not $NoBrowser) {
            # Give Vite a moment to bind before the browser asks for the page.
            Start-Sleep -Seconds 4
            Start-Process 'http://127.0.0.1:5173'
        }
    }
    else {
        Write-Host 'Frontend not created yet (Phase 6) -- backend only.' -ForegroundColor Yellow
    }
}

Write-Host ''
Write-Host 'Both windows stay open. Close them to stop the servers.' -ForegroundColor Green
