<#
.SYNOPSIS
    One-shot bring-up of this machine for boxing-comps: survey it (read-only),
    then build and check the .venv. Safe to re-run.

.DESCRIPTION
    1. inventory-machine.ps1 -- read-only. Printed, and saved to
       ~\boxing-comps-inventory-<machine>-<time>.txt (outside the repo and the
       vault) so it can be pasted into the assistant session.
    2. setup-env.ps1 -- builds or repairs <repo>\.venv for this machine (CUDA when
       nvidia-smi sees a GPU) with the pose step, and ends with check_env.py.
    Each runs as its own process, so a failure in one is reported, not fatal to
    the other. Nothing is moved or deleted: the old clones and environments the
    inventory finds are decided on afterwards, by a person.

.PARAMETER NoSetup
    Only the inventory.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
#>
[CmdletBinding()]
param([switch]$NoSetup)

$repo = Split-Path -Parent $PSScriptRoot
$stamp = Get-Date -Format 'yyyyMMdd-HHmm'
$inv = Join-Path $env:USERPROFILE "boxing-comps-inventory-$env:COMPUTERNAME-$stamp.txt"
$ps = Join-Path $PSHOME 'powershell.exe'

Write-Host "=== 1/2 inventory (read-only) ===" -ForegroundColor Cyan
$out = & $ps -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'inventory-machine.ps1')
$invCode = $LASTEXITCODE
$out | Out-File -LiteralPath $inv -Encoding utf8
$out | ForEach-Object { Write-Host $_ }

$envCode = $null
if (-not $NoSetup) {
    Write-Host "`n=== 2/2 environment (setup-env.ps1) ===" -ForegroundColor Cyan
    & $ps -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'setup-env.ps1')
    $envCode = $LASTEXITCODE
}

Write-Host "`n=== summary ===" -ForegroundColor Cyan
# Pasted into chat like the inventory, so the profile path (the username) is shortened here too.
$home_ = $env:USERPROFILE.TrimEnd('\')
$short = { param($p) $p -replace [regex]::Escape($home_), '~' }
Write-Host "repo:        $(& $short $repo) (branch $(git -C $repo branch --show-current))"
Write-Host ("inventory:   {0}  -> {1}" -f $(if ($invCode -eq 0) { 'done' } else { "FAILED (exit $invCode)" }), (& $short $inv))
if ($NoSetup) { Write-Host 'environment: skipped (-NoSetup)' }
elseif ($envCode -eq 0) { Write-Host 'environment: ENVIRONMENT OK' -ForegroundColor Green }
else { Write-Host "environment: BROKEN (exit $envCode) -- the lines above say what" -ForegroundColor Red }
Write-Host 'next:        paste the inventory file into the assistant session; nothing has been moved or deleted.'
if ($invCode -ne 0 -or ($envCode -ne $null -and $envCode -ne 0)) { exit 1 }
exit 0
