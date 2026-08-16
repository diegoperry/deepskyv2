param(
    [string]$AppTaskName = "DeepSky Web App",
    [string]$HealthUrl = "http://127.0.0.1:8000/process",
    [string]$LogPath = "C:\Apps\deepskyv2\deepsky-watchdog.log"
)

$ErrorActionPreference = "Stop"

function Write-WatchdogLog([string]$Message) {
    $directory = Split-Path -Parent $LogPath
    if ($directory) {
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
    }
    "$(Get-Date -Format o) $Message" | Add-Content -LiteralPath $LogPath -Encoding UTF8
}

try {
    $response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 20
    if ($response.StatusCode -eq 200) {
        exit 0
    }
    throw "Health check returned HTTP $($response.StatusCode)."
} catch {
    Write-WatchdogLog "Health check failed: $($_.Exception.Message)"
}

try {
    $task = Get-ScheduledTask -TaskName $AppTaskName -ErrorAction Stop
    if ($task.State -eq "Running") {
        Stop-ScheduledTask -TaskName $AppTaskName -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 3
    }

    Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object {
            Write-WatchdogLog "Stopping unhealthy listener PID $($_.OwningProcess)."
            Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
        }

    Start-ScheduledTask -TaskName $AppTaskName
    Start-Sleep -Seconds 12
    $response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 20
    if ($response.StatusCode -ne 200) {
        throw "Restarted task returned HTTP $($response.StatusCode)."
    }
    Write-WatchdogLog "Recovered DeepSky successfully."
} catch {
    Write-WatchdogLog "Recovery failed: $($_.Exception.Message)"
    exit 1
}
