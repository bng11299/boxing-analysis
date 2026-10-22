<#
.SYNOPSIS
    Build (or check) this repo's .venv for THIS machine, then prove it works.

.DESCRIPTION
    A .venv is never copied between machines: torch is a different build on each
    (CPU here, CUDA on the GPU laptop), and the venv records absolute paths. This
    script rebuilds it in place, the same way on every machine.

    Torch (and, with the pose step, torchvision) is installed explicitly from the
    PyTorch index as ONE build for this machine. Everything else is installed
    without those lines and under a constraints file that pins them: requirements.txt
    pins torch==2.13.0+cpu, and ultralytics left to PyPI pulls torchvision 0.29,
    which requires torch>=2.14 -- either would quietly replace the CUDA build.

    It ends by running scripts\check_env.py, which imports every package and, on
    CUDA, computes a tensor on the GPU. An install that exits 0 proves nothing
    until the import is checked (ISSUES-LOG 2026-08-30).

.PARAMETER Torch
    auto (default): CUDA if nvidia-smi lists a GPU, else CPU. Or cpu / cuda.

.PARAMETER CudaTag
    The PyTorch CUDA wheel tag. cu130 is what the RTX 3060 laptop runs (driver
    617.14 supports up to CUDA 13.4; torch 2.13 ships cu126, cu130, cu132).

.PARAMETER NoPose
    Skip requirements-pose.txt (ultralytics). The pose step then cannot run here.

.PARAMETER CheckOnly
    Install nothing; report whether the existing .venv is good. Exit 1 if not.

.PARAMETER Recreate
    Delete this repo's .venv first and build it from scratch.

.EXAMPLE
    .\scripts\setup-env.ps1 -CheckOnly
    .\scripts\setup-env.ps1                 # on the laptop: CUDA, with the pose step
    .\scripts\setup-env.ps1 -NoPose         # here: CPU, no ultralytics
#>
[CmdletBinding()]
param(
    [ValidateSet('auto', 'cpu', 'cuda')][string]$Torch = 'auto',
    [ValidatePattern('^cu\d{3}$')][string]$CudaTag = 'cu130',
    [switch]$NoPose,
    [switch]$CheckOnly,
    [switch]$Recreate
)

$TorchIndex = 'https://download.pytorch.org/whl/'

function Get-Pin([string]$RequirementsPath, [string]$Name) {
    # The base version pinned for $Name, e.g. "2.13.0" from torch==2.13.0+cpu.
    $n = [regex]::Escape($Name)
    $hits = @(Get-Content -LiteralPath $RequirementsPath |
        Where-Object { $_ -match "^\s*$n\s*==\s*([0-9][0-9A-Za-z.]*)(\+[0-9A-Za-z.]+)?\s*(#.*)?$" })
    if ($hits.Count -ne 1) { throw "expected exactly one $Name== pin in $RequirementsPath, found $($hits.Count)" }
    $null = $hits[0] -match '==\s*([0-9][0-9A-Za-z.]*)'
    return $Matches[1]
}

function Get-RequirementsWithout([string]$RequirementsPath, [string[]]$Names) {
    # Every line except the pins for $Names (exact names: torchaudio is kept for torch).
    $alt = ($Names | ForEach-Object { [regex]::Escape($_) }) -join '|'
    return @(Get-Content -LiteralPath $RequirementsPath | Where-Object { $_ -notmatch "^\s*($alt)\s*==" })
}

function Resolve-TorchBuild([string]$Mode, [string]$Tag, [scriptblock]$HasNvidiaGpu) {
    if ($Mode -eq 'cpu') { return 'cpu' }
    if ($Mode -eq 'cuda') { return $Tag }
    if (& $HasNvidiaGpu) { return $Tag }
    return 'cpu'
}

function Test-NvidiaGpu {
    # Under the caller's 'Stop', PS 5.1 turns nvidia-smi's first stderr line into a
    # terminating error even with 2>$null; a probe must not abort the script.
    $ErrorActionPreference = 'Continue'
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) { return $false }
    try { $out = & $smi.Source -L 2>$null } catch { return $false }
    return ($LASTEXITCODE -eq 0) -and ([string]$out -match 'GPU \d')
}

function Assert-VenvPath([string]$Repo, [string]$Venv) {
    # The only directory -Recreate may delete: <repo>\.venv, a real venv, not a link.
    $expected = [IO.Path]::GetFullPath((Join-Path $Repo '.venv'))
    if ([IO.Path]::GetFullPath($Venv) -ne $expected) { throw "refusing to delete ${Venv}: not $expected" }
    $item = Get-Item -LiteralPath $Venv -Force -ErrorAction SilentlyContinue
    if (-not $item) { throw "refusing to delete ${Venv}: does not exist" }
    # A junction: some Remove-Item versions follow it and empty the TARGET.
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "refusing to delete ${Venv}: a junction or link" }
    if (-not (Test-Path -LiteralPath (Join-Path $Venv 'pyvenv.cfg'))) { throw "refusing to delete ${Venv}: no pyvenv.cfg, not a venv" }
}

function Find-PyLauncher {
    $p = Join-Path $env:LOCALAPPDATA 'Programs\Python\Launcher\py.exe'
    if (Test-Path -LiteralPath $p) { return $p }
    $cmd = Get-Command py -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notmatch 'WindowsApps') { return $cmd.Source }
    throw 'no Python launcher (py.exe) found; install Python 3.13 from python.org'
}

function Invoke-Checked([string]$Exe, [string[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$([IO.Path]::GetFileName($Exe)) $($Arguments -join ' ') exited $LASTEXITCODE" }
}

function Get-InstalledVersion([string]$Python, [string]$Package) {
    $v = & $Python -c "import importlib.metadata as m, sys`ntry: print(m.version(sys.argv[1]))`nexcept m.PackageNotFoundError: pass" $Package
    if ($LASTEXITCODE -ne 0) { return $null }
    return [string]$v
}

function Invoke-Main {
    $ErrorActionPreference = 'Stop'
    $repo = Split-Path -Parent $PSScriptRoot
    $venv = Join-Path $repo '.venv'
    $py = Join-Path $venv 'Scripts\python.exe'
    $req = Join-Path $repo 'requirements.txt'
    $build = Resolve-TorchBuild $Torch $CudaTag ${function:Test-NvidiaGpu}
    $checkArgs = @((Join-Path $repo 'scripts\check_env.py'), '--torch', $build)
    if (-not $NoPose) { $checkArgs += '--pose' }
    Write-Host "repo $repo; torch build: $build; pose step: $(-not $NoPose)"

    if ($CheckOnly) {
        if (-not (Test-Path -LiteralPath $py)) { Write-Host "NO .venv at $venv -- run this script without -CheckOnly"; exit 1 }
        & $py @checkArgs
        exit $LASTEXITCODE
    }
    if ($Recreate -and (Test-Path -LiteralPath $venv)) {
        Assert-VenvPath $repo $venv
        Remove-Item -LiteralPath $venv -Recurse -Force
    }
    if (-not (Test-Path -LiteralPath $py)) {
        Invoke-Checked (Find-PyLauncher) @('-3.13', '-m', 'venv', $venv)
    }
    $poseReq = Join-Path $repo 'requirements-pose.txt'
    # torch and torchvision: ONE build, from the PyTorch index only (--index-url
    # replaces PyPI for this command, so nothing same-named can be mixed in).
    $pins = [ordered]@{ torch = "$(Get-Pin $req 'torch')+$build" }
    if (-not $NoPose) { $pins['torchvision'] = "$(Get-Pin $poseReq 'torchvision')+$build" }
    $stale = @($pins.Keys | Where-Object { (Get-InstalledVersion $py $_) -ne $pins[$_] })
    if ($stale.Count) {
        $specs = @($pins.Keys | ForEach-Object { "$_==$($pins[$_])" })
        Invoke-Checked $py (@('-m', 'pip', 'install') + $specs + @('--index-url', "$TorchIndex$build"))
    }
    # Everything else, under a constraints file that holds those builds: if any
    # dependency wants another torch, pip fails here instead of swapping it.
    $tmp = [IO.Path]::GetTempPath()
    $con = Join-Path $tmp "boxing-comps-constraints-$PID.txt"
    $rest = Join-Path $tmp "boxing-comps-req-$PID.txt"
    $restPose = Join-Path $tmp "boxing-comps-pose-$PID.txt"
    try {
        Set-Content -LiteralPath $con -Value @($pins.Keys | ForEach-Object { "$_==$($pins[$_])" }) -Encoding ASCII
        Set-Content -LiteralPath $rest -Value (Get-RequirementsWithout $req @('torch', 'torchvision')) -Encoding ASCII
        Invoke-Checked $py @('-m', 'pip', 'install', '-r', $rest, '-c', $con)
        if (-not $NoPose) {
            Set-Content -LiteralPath $restPose -Value (Get-RequirementsWithout $poseReq @('torch', 'torchvision')) -Encoding ASCII
            Invoke-Checked $py @('-m', 'pip', 'install', '-r', $restPose, '-c', $con)
        }
    } finally {
        Remove-Item -LiteralPath $con, $rest, $restPose -ErrorAction SilentlyContinue
    }
    & $py @checkArgs
    exit $LASTEXITCODE
}

# Dot-sourcing (the tests) loads the functions without running anything.
if ($MyInvocation.InvocationName -ne '.') { Invoke-Main }
