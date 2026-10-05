<#
.SYNOPSIS
    READ-ONLY survey of a machine for the two-machine setup: where the git clones,
    Python environments, videos and model weights are, and whether the GPU, ffmpeg
    and disk encryption are in place. Changes nothing.

.DESCRIPTION
    Read-only by construction: git runs with --no-optional-locks and only reads
    (no fetch, no index refresh), Python environments are read from pyvenv.cfg and
    their dist-info folder names (nothing is executed inside them), and files are
    listed, never opened -- except weights (.pt/.pth/.onnx under 1 GB), which are
    hashed so both machines can be checked against one pinned sha256.

    Paths are printed relative to your profile (~). Credentials embedded in a git
    remote URL are redacted. Paste the output back into the assistant session.

.PARAMETER Roots
    Where to search. Defaults to your profile; add other drives if you use them,
    e.g. -Roots $env:USERPROFILE, D:\

.PARAMETER MaxDepth
    How many folder levels below each root to search (default 6).

.EXAMPLE
    .\scripts\inventory-machine.ps1
    powershell -ExecutionPolicy Bypass -File .\inventory-machine.ps1 -Roots $env:USERPROFILE, D:\
#>
[CmdletBinding()]
param(
    [string[]]$Roots = @($env:USERPROFILE),
    [ValidateRange(1, 12)][int]$MaxDepth = 6
)

$SkipDirs = @('AppData', 'node_modules', '.venv', 'venv', 'site-packages', '__pycache__',
              '$Recycle.Bin', 'Windows', 'Program Files', 'Program Files (x86)', 'ProgramData',
              '.cache', '.git', 'System Volume Information')
$MediaExt = @('.mp4', '.mov', '.m4v', '.mkv', '.avi')
$WeightExt = @('.pt', '.pth', '.onnx', '.ckpt')
$MaxHashBytes = 1GB

function Hide-Credentials([string]$Url) {
    # https://user:token@host/... -> https://<redacted>@host/... ; token query
    # parameters -> <redacted>; ssh remotes (git@host:...) carry no secret and are kept.
    $u = $Url -replace '(?<=://)[^/@\s]+@', '<redacted>@'
    return ($u -replace '(?i)([?&](access_token|private_token|oauth_token|token|auth|key|sig|password)=)[^&#\s]+', '$1<redacted>')
}

function Format-Home([string]$Text) {
    # Every occurrence, either slash style: `py -0p` lines carry the path mid-line,
    # and file:/// URLs use forward slashes.
    $h = $env:USERPROFILE.TrimEnd('\')
    return ($Text -replace [regex]::Escape($h), '~' -replace [regex]::Escape($h.Replace('\', '/')), '~')
}

function Test-CloudOnly([IO.FileInfo]$File) {
    # OneDrive and similar placeholders: reading the file would DOWNLOAD it, which
    # a read-only survey must not do. Offline 0x1000, RecallOnOpen 0x40000,
    # RecallOnDataAccess 0x400000, or any reparse point.
    return (([int]$File.Attributes) -band (0x1000 -bor 0x40000 -bor 0x400000 -bor 0x400)) -ne 0
}

function Find-Items([string[]]$Start, [int]$Depth, [string[]]$Skip) {
    # Breadth-first walk. Returns git work trees, venvs/conda envs and files of interest.
    # Reparse points (junctions, OneDrive placeholders' links) are not followed.
    $found = @{ Repos = @(); Envs = @(); Files = @() }
    $queue = New-Object System.Collections.Queue
    foreach ($s in $Start) { if (Test-Path -LiteralPath $s) { $queue.Enqueue(@((Get-Item -LiteralPath $s -Force), 0)) } }
    $interesting = $MediaExt + $WeightExt
    while ($queue.Count -gt 0) {
        $dir, $d = $queue.Dequeue()
        $kids = @(Get-ChildItem -LiteralPath $dir.FullName -Force -ErrorAction SilentlyContinue)
        foreach ($k in $kids) {
            if ($k.PSIsContainer) {
                if ($k.Name -eq '.git') { $found.Repos += $dir.FullName; continue }
                if ($k.Name -eq 'conda-meta') { $found.Envs += $dir.FullName; continue }
                if ($Skip -contains $k.Name) {
                    if ($k.Name -in @('.venv', 'venv') -and (Test-Path -LiteralPath (Join-Path $k.FullName 'pyvenv.cfg'))) { $found.Envs += $k.FullName }
                    continue
                }
                if ($k.Attributes -band [IO.FileAttributes]::ReparsePoint) { continue }
                if ($d + 1 -le $Depth) { $queue.Enqueue(@($k, ($d + 1))) }
            } elseif ($k.Name -eq 'pyvenv.cfg') {
                $found.Envs += $dir.FullName
            } elseif ($interesting -contains $k.Extension.ToLower() -and $k.Length -ge 1MB) {
                $found.Files += $k
            }
        }
    }
    $found.Envs = @($found.Envs | Sort-Object -Unique)
    return $found
}

function Get-WeightHash([IO.FileInfo]$File) {
    # The one place a file's contents are read. Never a cloud-only placeholder.
    if (Test-CloudOnly $File) { return '(cloud-only, not hashed: hashing would download it)' }
    if ($File.Length -gt $MaxHashBytes) { return '(not hashed, > 1 GB)' }
    $fh = Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256 -ErrorAction SilentlyContinue
    if ($fh) { return $fh.Hash.ToLower() }
    return '(unreadable: in use or no permission)'
}

function Get-EnvPackages([string]$EnvDir) {
    # Package versions from dist-info folder names -- nothing inside the env is run.
    $site = @((Join-Path $EnvDir 'Lib\site-packages'), (Join-Path $EnvDir 'site-packages')) |
        Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    $out = [ordered]@{}
    foreach ($pkg in 'torch', 'ultralytics', 'numpy') {
        $hit = if ($site) { Get-ChildItem -LiteralPath $site -Directory -Filter "$pkg-*.dist-info" -ErrorAction SilentlyContinue | Select-Object -First 1 }
        $out[$pkg] = if ($hit) { $hit.Name -replace "^$pkg-", '' -replace '\.dist-info$', '' } else { '-' }
    }
    $cfg = Join-Path $EnvDir 'pyvenv.cfg'
    $ver = '?'
    if (Test-Path -LiteralPath $cfg) {
        $line = Select-String -LiteralPath $cfg -Pattern '^\s*version(_info)?\s*=\s*(.+)$' | Select-Object -First 1
        if ($line) { $ver = $line.Matches[0].Groups[2].Value.Trim() }
    }
    return "python $ver | torch $($out.torch) | ultralytics $($out.ultralytics) | numpy $($out.numpy)"
}

function Get-RepoLine([string]$Repo) {
    # --no-optional-locks: no index refresh written. core.fsmonitor off: a repo
    # configured with one would otherwise start a daemon or run a hook.
    $g = @('-c', 'core.fsmonitor=false', '--no-optional-locks', '-C', $Repo)
    $remote = (& git @g remote get-url origin 2>$null)
    $branch = (& git @g branch --show-current 2>$null)
    $head = (& git @g log -1 --format='%h %ad' --date=short 2>$null)
    $dirty = @(& git @g status --porcelain --untracked-files=no 2>$null).Count
    $remote = if ($remote) { Hide-Credentials ([string]$remote) } else { '(no origin)' }
    return "{0}`n      origin {1} | branch {2} | head {3} | {4} modified tracked file(s)" -f (Format-Home $Repo), $remote, $branch, $head, $dirty
}

function Invoke-Main {
    Write-Output "=== machine: $env:COMPUTERNAME, $((Get-Date).ToString('yyyy-MM-dd HH:mm')), read-only inventory"
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    # Quoted: a bare comma list is an array in PowerShell and would split the argument.
    $gpu = if ($smi) { (& $smi.Source '--query-gpu=name,memory.total,driver_version' '--format=csv,noheader' 2>$null) -join '; ' } else { 'no nvidia-smi' }
    Write-Output "gpu: $gpu"
    $ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
    Write-Output "ffmpeg: $(if ($ff) { Format-Home $ff.Source } else { 'NOT on PATH' })"
    try {
        $bl = Get-BitLockerVolume -MountPoint 'C:' -ErrorAction Stop
        Write-Output "bitlocker C: $($bl.ProtectionStatus) ($($bl.VolumeStatus))"
    } catch { Write-Output 'bitlocker C: unknown (needs an admin shell to read)' }
    Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Used -ne $null } | ForEach-Object {
        Write-Output ("disk {0}: {1:N0} GB free of {2:N0} GB" -f $_.Name, ($_.Free / 1GB), (($_.Used + $_.Free) / 1GB)) }

    Write-Output "`n=== python installs (py -0p)"
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        foreach ($l in @(& $launcher.Source -0p 2>$null)) {
            $exe = ($l -split '\s+') | Where-Object { $_ -like '*.exe' } | Select-Object -First 1
            $pk = if ($exe) { Get-EnvPackages (Split-Path -Parent $exe) } else { '' }
            Write-Output "  $(Format-Home ([string]$l).Trim())`n      $pk"
        }
    } else { Write-Output '  no py launcher' }
    # `pip install` outside a venv without admin rights lands in the USER site, not
    # the install above -- a common place for a stray torch build to hide.
    foreach ($u in @(Get-ChildItem -Path (Join-Path $env:APPDATA 'Python') -Directory -Filter 'Python3*' -ErrorAction SilentlyContinue)) {
        if (Test-Path -LiteralPath (Join-Path $u.FullName 'site-packages')) {
            Write-Output "  user site $(Format-Home $u.FullName)`n      $(Get-EnvPackages $u.FullName)" }
    }

    $f = Find-Items $Roots $MaxDepth $SkipDirs
    Write-Output "`n=== git clones ($($f.Repos.Count))"
    foreach ($r in $f.Repos) { Write-Output "  $(Get-RepoLine $r)" }
    Write-Output "`n=== python environments ($($f.Envs.Count))"
    foreach ($e in $f.Envs) { Write-Output "  $(Format-Home $e)`n      $(Get-EnvPackages $e)" }
    $media = @($f.Files | Where-Object { $MediaExt -contains $_.Extension.ToLower() })
    $weights = @($f.Files | Where-Object { $WeightExt -contains $_.Extension.ToLower() })
    Write-Output "`n=== videos >= 1 MB ($($media.Count))"
    foreach ($m in $media | Sort-Object FullName) {
        Write-Output ("  {0}  {1:N0} MB  {2}" -f (Format-Home $m.FullName), ($m.Length / 1MB), $m.LastWriteTime.ToString('yyyy-MM-dd')) }
    Write-Output "`n=== model weights ($($weights.Count))"
    foreach ($w in $weights | Sort-Object FullName) {
        $h = Get-WeightHash $w
        Write-Output ("  {0}  {1:N1} MB`n      sha256 {2}" -f (Format-Home $w.FullName), ($w.Length / 1MB), $h) }
}

if ($MyInvocation.InvocationName -ne '.') { Invoke-Main }
