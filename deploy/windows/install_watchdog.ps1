param(
    [string]$AppTaskName = "DeepSky Web App",
    [string]$WatchdogTaskName = "DeepSky Web Watchdog",
    [string]$HealthUrl = "http://127.0.0.1:8000/process",
    [string]$LogPath = "C:\Apps\deepskyv2\deepsky-watchdog.log"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$watchdogScript = Join-Path $repoRoot "deploy\windows\watchdog.ps1"
if (-not (Test-Path -LiteralPath $watchdogScript)) {
    throw "Missing watchdog script: $watchdogScript"
}
if (-not (Get-ScheduledTask -TaskName $AppTaskName -ErrorAction SilentlyContinue)) {
    throw "Application task '$AppTaskName' does not exist."
}

$arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$watchdogScript`" -AppTaskName `"$AppTaskName`" -HealthUrl `"$HealthUrl`" -LogPath `"$LogPath`""
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arguments -WorkingDirectory $repoRoot
$startupTrigger = New-ScheduledTaskTrigger -AtStartup
$intervalTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 2)
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 2)

Register-ScheduledTask `
    -TaskName $WatchdogTaskName `
    -Action $action `
    -Trigger @($startupTrigger, $intervalTrigger) `
    -Settings $settings `
    -Description "Checks DeepSky every two minutes and restarts its web task when unhealthy" `
    -User "SYSTEM" `
    -RunLevel Highest `
    -Force | Out-Null

Write-Output "Installed watchdog task '$WatchdogTaskName' for '$AppTaskName'."
