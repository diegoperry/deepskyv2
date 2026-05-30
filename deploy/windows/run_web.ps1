param(
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$pythonExe = Join-Path $repoRoot ".venv-web\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Missing web virtualenv. Run deploy\windows\install.ps1 first."
}

Set-Location (Join-Path $repoRoot "DeepSky")
& $pythonExe -m uvicorn app.web_app:app --host $HostName --port $Port
