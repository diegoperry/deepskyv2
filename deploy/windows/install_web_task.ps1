param(
    [string]$TaskName = "DeepSkyWeb",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$runScript = Join-Path $repoRoot "deploy\windows\run_web.ps1"

if (-not (Test-Path -LiteralPath $runScript)) {
    throw "Missing run script: $runScript"
}

$argument = "-NoProfile -ExecutionPolicy Bypass -File `"$runScript`" -HostName $HostName -Port $Port"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argument -WorkingDirectory $repoRoot
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit 0

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "DeepSky FastAPI web server" `
    -User "SYSTEM" `
    -RunLevel Highest `
    -Force | Out-Null

Start-ScheduledTask -TaskName $TaskName

Write-Output "Installed and started scheduled task: $TaskName"
