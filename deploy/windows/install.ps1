param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repoRoot

$venv = Join-Path $repoRoot ".venv-web"
$pythonExe = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    & $Python -m venv $venv
}

& $pythonExe -m pip install --upgrade pip
& $pythonExe -m pip install -r .\DeepSky\requirements.txt

New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "outputs") | Out-Null

Write-Output "DeepSky web environment installed."
Write-Output "Run: powershell -ExecutionPolicy Bypass -File .\deploy\windows\run_web.ps1"
