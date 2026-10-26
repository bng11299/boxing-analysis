# Tests for scripts\setup-env.ps1 and scripts\inventory-machine.ps1.
# Plain asserts, no Pester (Windows PowerShell ships only Pester 3.4).
#   powershell -ExecutionPolicy Bypass -File tests\test_machine_scripts.ps1

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$script:pass = 0; $script:fail = 0

function Check([string]$Name, [bool]$Cond, [string]$Detail = '') {
    if ($Cond) { $script:pass++; Write-Output "  PASS  $Name" }
    else { $script:fail++; Write-Output "  FAIL  $Name  $Detail" }
}

function Throws([string]$Name, [scriptblock]$Block, [string]$Pattern) {
    # The message must match: "it threw something" also passes on a typo or a binding error.
    try { & $Block; Check $Name $false 'did not throw' }
    catch { Check $Name ($_.Exception.Message -match $Pattern) "threw: $($_.Exception.Message)" }
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("bc-scripts-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null

. (Join-Path $root 'scripts\setup-env.ps1')
Write-Output 'setup-env: requirements handling'
Check 'the torch pin base version is read from the real requirements.txt' ((Get-Pin (Join-Path $root 'requirements.txt') 'torch') -eq '2.13.0')
Check 'the torchvision pin is read from the real requirements-pose.txt' ((Get-Pin (Join-Path $root 'requirements-pose.txt') 'torchvision') -eq '0.28.0')
$r = Join-Path $tmp 'r.txt'
Set-Content -LiteralPath $r -Value @('numpy==2.2.6', 'torchvision==0.28.0') -Encoding ASCII
Throws 'no torch pin -> refused (torchvision is not mistaken for it)' { Get-Pin $r 'torch' } 'found 0'
Set-Content -LiteralPath $r -Value @('torch==2.13.0+cpu', 'torch==2.12.0') -Encoding ASCII
Throws 'two torch pins -> refused' { Get-Pin $r 'torch' } 'found 2'
Set-Content -LiteralPath $r -Value @('# c', 'torch==2.13.0+cpu', 'torchvision==0.28.0', 'torchaudio==2.13.0', 'numpy==2.2.6') -Encoding ASCII
$rest = Get-RequirementsWithout $r @('torch', 'torchvision')
Check 'torch and torchvision pins dropped, and only they (torchaudio kept)' (($rest -join '|') -eq '# c|torchaudio==2.13.0|numpy==2.2.6') ($rest -join '|')
$real = Get-RequirementsWithout (Join-Path $root 'requirements.txt') @('torch', 'torchvision')
Check 'the real requirements.txt loses exactly one line' ($real.Count -eq (@(Get-Content (Join-Path $root 'requirements.txt')).Count - 1))
$realPose = Get-RequirementsWithout (Join-Path $root 'requirements-pose.txt') @('torch', 'torchvision')
Check 'the real requirements-pose.txt keeps ultralytics and loses torchvision' ((($realPose -join '|') -match 'ultralytics==') -and -not (($realPose -join '|') -match 'torchvision=='))

Write-Output 'setup-env: torch build'
Check 'cpu -> cpu' ((Resolve-TorchBuild 'cpu' 'cu130' { $true }) -eq 'cpu')
Check 'cuda -> the tag, GPU or not' ((Resolve-TorchBuild 'cuda' 'cu130' { $false }) -eq 'cu130')
Check 'auto with an NVIDIA GPU -> the tag' ((Resolve-TorchBuild 'auto' 'cu130' { $true }) -eq 'cu130')
Check 'auto without one -> cpu' ((Resolve-TorchBuild 'auto' 'cu130' { $false }) -eq 'cpu')

Write-Output 'setup-env: the -Recreate delete guard'
$fakeRepo = Join-Path $tmp 'repo'
New-Item -ItemType Directory -Path (Join-Path $fakeRepo '.venv'), (Join-Path $fakeRepo 'data') | Out-Null
Throws 'refuses a .venv that is not a venv (no pyvenv.cfg)' { Assert-VenvPath $fakeRepo (Join-Path $fakeRepo '.venv') } 'no pyvenv.cfg'
# Every refusal below is set up so ONLY the path rule can refuse it: each target holds a pyvenv.cfg.
foreach ($d in $fakeRepo, (Join-Path $fakeRepo 'data'), (Join-Path $fakeRepo '.venv')) {
    Set-Content -LiteralPath (Join-Path $d 'pyvenv.cfg') -Value 'version = 3.13.15' -Encoding ASCII }
Throws 'refuses the repo itself' { Assert-VenvPath $fakeRepo $fakeRepo } 'not .*\\\.venv'
Throws 'refuses another folder in the repo' { Assert-VenvPath $fakeRepo (Join-Path $fakeRepo 'data') } 'not .*\\\.venv'
Throws 'refuses a path that only resolves past .venv' { Assert-VenvPath $fakeRepo (Join-Path $fakeRepo '.venv\..\data') } 'not .*\\\.venv'
try { Assert-VenvPath $fakeRepo (Join-Path $fakeRepo '.venv'); Check 'accepts <repo>\.venv with pyvenv.cfg' $true }
catch { Check 'accepts <repo>\.venv with pyvenv.cfg' $false $_.Exception.Message }
$junRepo = Join-Path $tmp 'junrepo'; $outside = Join-Path $tmp 'outside-venv'
New-Item -ItemType Directory -Path $junRepo, $outside | Out-Null
Set-Content -LiteralPath (Join-Path $outside 'pyvenv.cfg') -Value 'version = 3.13.15' -Encoding ASCII
New-Item -ItemType Junction -Path (Join-Path $junRepo '.venv') -Target $outside | Out-Null
Throws 'refuses a .venv that is a junction (it holds a pyvenv.cfg through the link)' { Assert-VenvPath $junRepo (Join-Path $junRepo '.venv') } 'junction'
Throws 'refuses a .venv that does not exist' { Assert-VenvPath (Join-Path $tmp 'nope') (Join-Path $tmp 'nope\.venv') } 'does not exist'
[IO.Directory]::Delete((Join-Path $junRepo '.venv'))  # removes the link only

. (Join-Path $root 'scripts\inventory-machine.ps1')
Write-Output 'inventory: credentials'
$h = Hide-Credentials 'https://bng11299:ghp_SECRETtoken123@github.com/bng11299/boxing-comps.git'
Check 'a token in an https remote is redacted' (($h -notmatch 'ghp_SECRET') -and ($h -match '://<redacted>@github\.com/bng11299/boxing-comps')) $h
Check 'a token-only https remote is redacted' ((Hide-Credentials 'https://ghp_x@github.com/a/b') -eq 'https://<redacted>@github.com/a/b')
Check 'a plain https remote is unchanged' ((Hide-Credentials 'https://github.com/a/b.git') -eq 'https://github.com/a/b.git')
Check 'an ssh remote is unchanged' ((Hide-Credentials 'git@github.com:a/b.git') -eq 'git@github.com:a/b.git')
$q = Hide-Credentials 'https://host/a/b.git?access_token=s3cret&ref=main'
Check 'a token in a query string is redacted, other parameters kept' (($q -notmatch 's3cret') -and ($q -match 'ref=main')) $q
$fwd = Format-Home ('file:///' + $env:USERPROFILE.Replace('\', '/') + '/repos/x')
Check 'a forward-slash profile path is shortened too' (($fwd -eq 'file:///~/repos/x')) $fwd

$line = " -V:3.13 *        $env:USERPROFILE\AppData\Local\Programs\Python\Python313\python.exe"
Check 'the profile path is shortened to ~ mid-line too (py -0p)' (((Format-Home $line) -notmatch [regex]::Escape($env:USERPROFILE)) -and ((Format-Home $line) -match '~\\AppData')) (Format-Home $line)
Check 'a path outside the profile is unchanged' ((Format-Home 'D:\data\x.mov') -eq 'D:\data\x.mov')

Write-Output 'inventory: the walk'
$t = Join-Path $tmp 'tree'
$dirs = 'a\.git', 'a\data\own', 'node_modules\b\.git', 'env1', 'proj\.venv\Lib\site-packages\torch-2.13.0+cu130.dist-info',
        'deep\x\y\z\.git', 'AppData\c\.git'
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path (Join-Path $t $d) | Out-Null }
Set-Content -LiteralPath (Join-Path $t 'env1\pyvenv.cfg') -Value 'version = 3.12.1' -Encoding ASCII
Set-Content -LiteralPath (Join-Path $t 'proj\.venv\pyvenv.cfg') -Value 'version = 3.13.15' -Encoding ASCII
$fs = [IO.File]::Create((Join-Path $t 'a\data\own\big.mov')); $fs.SetLength(2MB); $fs.Close()
$fs = [IO.File]::Create((Join-Path $t 'a\data\own\small.mov')); $fs.SetLength(10KB); $fs.Close()
$fs = [IO.File]::Create((Join-Path $t 'a\m.pt')); $fs.SetLength(1MB); $fs.Close()
$found = Find-Items @($t) 3 $SkipDirs
$repos = @($found.Repos | ForEach-Object { $_.Substring($t.Length + 1) })
Check 'a clone is found' ($repos -contains 'a') ($repos -join ',')
Check 'clones under node_modules and AppData are not walked into' (-not ($repos | Where-Object { $_ -match 'node_modules|AppData' })) ($repos -join ',')
Check 'a clone deeper than MaxDepth is not reported' (-not ($repos -contains 'deep\x\y\z')) ($repos -join ',')
$envs = @($found.Envs | ForEach-Object { $_.Substring($t.Length + 1) })
Check 'a venv by pyvenv.cfg, and a .venv inside a project, are both found' (($envs -contains 'env1') -and ($envs -contains 'proj\.venv')) ($envs -join ',')
$files = @($found.Files | ForEach-Object { $_.Name })
Check 'videos and weights >= 1 MB are listed, smaller ones are not' (($files -contains 'big.mov') -and ($files -contains 'm.pt') -and -not ($files -contains 'small.mov')) ($files -join ',')
$pk = Get-EnvPackages (Join-Path $t 'proj\.venv')
Check 'env packages are read from dist-info names, nothing executed' (($pk -match 'python 3\.13\.15') -and ($pk -match 'torch 2\.13\.0\+cu130') -and ($pk -match 'ultralytics -')) $pk

Write-Output 'inventory: cloud-only files are never read'
$cloud = Join-Path $tmp 'cloud.pt'
$fs = [IO.File]::Create($cloud); $fs.SetLength(1MB); $fs.Close()
Check 'control: an ordinary file is not cloud-only' (-not (Test-CloudOnly (Get-Item -LiteralPath $cloud)))
$plain = Get-WeightHash (Get-Item -LiteralPath $cloud)
Check 'control: an ordinary weights file is hashed' ($plain -match '^[0-9a-f]{64}$') $plain
[IO.File]::SetAttributes($cloud, [IO.FileAttributes]::Offline)
Check 'a file marked Offline (a placeholder) is cloud-only' (Test-CloudOnly (Get-Item -LiteralPath $cloud))
$h = Get-WeightHash (Get-Item -LiteralPath $cloud)
Check '... and the inventory''s hashing step reports it instead of reading it' ($h -match 'cloud-only') $h
[IO.File]::SetAttributes($cloud, [IO.FileAttributes]::Normal)

Write-Output 'inventory: reading a repo writes nothing to it'
$ErrorActionPreference = 'Continue'  # git's CRLF warnings on stderr must not abort the setup below
$gr = Join-Path $tmp 'gitrepo'
& git init -q $gr
Set-Content -LiteralPath (Join-Path $gr 'a.txt') -Value 'a' -Encoding ASCII
& git -C $gr add a.txt
& git -C $gr -c user.name=t -c user.email=t@t commit -qm a
& git -C $gr config core.fsmonitor true   # a configured fsmonitor starts a daemon on `git status`
& git -C $gr remote add origin 'https://user:ghp_TOKEN@github.com/a/b.git'
Set-Content -LiteralPath (Join-Path $gr 'a.txt') -Value 'changed' -Encoding ASCII  # a stat-dirty file invites an index refresh
$snap = { Get-ChildItem -LiteralPath (Join-Path $gr '.git') -Force | ForEach-Object { "$($_.Name)|$($_.LastWriteTimeUtc.Ticks)" } }
$before = & $snap
Start-Sleep -Milliseconds 50
$line = Get-RepoLine $gr
$after = & $snap
Check 'nothing in .git is created or touched (index, fsmonitor daemon)' (@(Compare-Object $before $after).Count -eq 0) ((Compare-Object $before $after | Out-String).Trim())
Check '... and the line reports the change, with the token redacted' (($line -match '1 modified') -and ($line -notmatch 'ghp_TOKEN')) $line
& git -C $gr fsmonitor--daemon stop 2>$null | Out-Null

Write-Output 'install-pose-watcher: what the task runs'
. (Join-Path $root 'scripts\install-pose-watcher.ps1')
$wr = Join-Path $tmp 'watch repo'
$spec = Get-WatcherSpec $wr (Join-Path $tmp 'w.log')
Check 'the task runs the repo venv python, not a bare python' ($spec.Execute -eq (Join-Path $wr '.venv\Scripts\python.exe')) $spec.Execute
Check 'the script and log paths are quoted (the repo path has a space)' ($spec.Argument -eq "`"$(Join-Path $wr 'pose_watcher.py')`" --log `"$(Join-Path $tmp 'w.log')`"") $spec.Argument
Check 'it runs from the repo root' ($spec.WorkingDirectory -eq $wr)
Throws 'refuses to register when the venv, script and weights are missing' { Assert-WatcherSpec $spec } 'not found: .*python\.exe.*pose_watcher\.py.*yolo11m-pose\.pt'
New-Item -ItemType Directory -Path (Join-Path $wr '.venv\Scripts'), (Join-Path $wr 'weights') | Out-Null
foreach ($f in '.venv\Scripts\python.exe', 'pose_watcher.py') { Set-Content -LiteralPath (Join-Path $wr $f) -Value 'x' -Encoding ASCII }
Throws 'refuses when only the weights are missing, naming them' { Assert-WatcherSpec $spec } 'not found: [^;]*yolo11m-pose\.pt$|not found: [^;]*yolo11m-pose\.pt\.'
Set-Content -LiteralPath (Join-Path $wr 'weights\yolo11m-pose.pt') -Value 'x' -Encoding ASCII
try { Assert-WatcherSpec $spec; Check 'accepts a repo with all three' $true } catch { Check 'accepts a repo with all three' $false $_.Exception.Message }
$dry = & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root 'scripts\install-pose-watcher.ps1') -DryRun -Uninstall
Check '-DryRun -Uninstall changes nothing and says so' (($dry -join ' ') -match '^would remove') ($dry -join ' ')

Remove-Item -LiteralPath $tmp -Recurse -Force
Write-Output ''
Write-Output "=== $($script:pass) passed, $($script:fail) failed ==="
if ($script:fail) { exit 1 } else { exit 0 }
