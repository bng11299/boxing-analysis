<#
.SYNOPSIS
    On the GPU laptop: start cli\pose_watcher.py at every logon, as a scheduled task for this user.

.DESCRIPTION
    Registers the task "boxing-comps pose watcher" (current user, at logon, normal privileges, no
    stored password). It runs <repo>\.venv\Scripts\python.exe cli/cli\pose_watcher.py from the repo root, appending
    to ~\boxing-comps-pose-watcher.log. It is restarted if it stops, with no time limit, and runs on battery.
    Checks first that the venv, the pinned pose weights and cli\pose_watcher.py are where the task will look.
    Changes nothing else. See docs\MACHINE-SYNC.md section 5.

.PARAMETER DryRun
    Print what would be registered and change nothing.

.PARAMETER Uninstall
    Stop and remove the task.

.EXAMPLE
    .\scripts\install-pose-watcher.ps1 -DryRun
    .\scripts\install-pose-watcher.ps1
    .\scripts\install-pose-watcher.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$Uninstall
)

$ErrorActionPreference = 'Stop'
$TaskName = 'boxing-comps pose watcher'

function Get-WatcherSpec([string]$Repo, [string]$LogPath) {
    # What the task runs. Every path is absolute and quoted; nothing comes from outside this repo and $HOME.
    $python = Join-Path $Repo '.venv\Scripts\python.exe'
    $script = Join-Path $Repo 'cli\pose_watcher.py'
    [pscustomobject]@{
        Execute          = $python
        Argument         = "`"$script`" --log `"$LogPath`""
        WorkingDirectory = $Repo
        Weights          = Join-Path $Repo 'weights\yolo11m-pose.pt'
        Script           = $script
    }
}

function Assert-WatcherSpec($Spec) {
    $missing = @()
    foreach ($p in $Spec.Execute, $Spec.Script, $Spec.Weights) {
        if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { $missing += $p }
    }
    if ($missing.Count) {
        throw ("not found: " + ($missing -join '; ') +
               ". Run .\scripts\setup-env.ps1 (with the pose step) and let Syncthing bring weights\ first.")
    }
}

function Invoke-Main {
    $repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
    $log = Join-Path $env:USERPROFILE 'boxing-comps-pose-watcher.log'
    $spec = Get-WatcherSpec $repo $log
    if ($Uninstall) {
        if ($DryRun) { Write-Output "would remove the task '$TaskName'"; return }
        if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
            Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
            Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
            Write-Output "removed the task '$TaskName'"
        } else { Write-Output "no task '$TaskName' to remove" }
        return
    }
    Write-Output "task:     $TaskName (at logon of $env:USERNAME, normal privileges)"
    Write-Output "runs:     $($spec.Execute) $($spec.Argument)"
    Write-Output "in:       $($spec.WorkingDirectory)"
    Write-Output "log:      $log"
    Assert-WatcherSpec $spec
    if ($DryRun) { Write-Output 'dry run: nothing registered'; return }
    $action = New-ScheduledTaskAction -Execute $spec.Execute -Argument $spec.Argument -WorkingDirectory $spec.WorkingDirectory
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 10 -RestartInterval (New-TimeSpan -Minutes 1) `
        -MultipleInstances IgnoreNew
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
        -Principal $principal -Force | Out-Null
    Start-ScheduledTask -TaskName $TaskName
    Write-Output "registered and started. Check it: Get-Content `"$log`" -Tail 5"
}

if ($MyInvocation.InvocationName -ne '.') { Invoke-Main }
