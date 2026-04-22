# hdgl_selfprovision.ps1
# Conscious 2.0 -- Self-Provisioning wrapper
#
# Runs hdgl_bench.exe (if present) and appends the Base4096 session token.
# Satisfies roadmap Ev1 Long: "running hdgl_bench.exe produces a Base4096 state token automatically"
#
# Usage:
#   .\hdgl_selfprovision.ps1             # bench + token
#   .\hdgl_selfprovision.ps1 -TokenOnly  # token only (skip bench)
#   .\hdgl_selfprovision.ps1 -Decode <token>  # decode a token

param(
    [switch]$TokenOnly,
    [string]$Decode
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($Decode) {
    py "$ScriptDir\hdgl_session_handoff.py" --decode $Decode
    exit $LASTEXITCODE
}

if (-not $TokenOnly) {
    $bench = Join-Path $ScriptDir "hdgl_bench.exe"
    if (Test-Path $bench) {
        Write-Host "=== hdgl_bench ===" -ForegroundColor Cyan
        & $bench
        Write-Host ""
    } else {
        Write-Host "[selfprovision] hdgl_bench.exe not found in $ScriptDir -- skipping bench run"
    }
}

Write-Host "=== Base4096 Session Token ===" -ForegroundColor Cyan
py "$ScriptDir\hdgl_session_handoff.py" --stats
py "$ScriptDir\hdgl_session_handoff.py"
