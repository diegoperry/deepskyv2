param(
    [switch]$SkipPyInstaller,
    [switch]$SkipVenv
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$venv = Join-Path $root ".build-venv"
$python = Join-Path $venv "Scripts\python.exe"

if (-not $SkipVenv) {
    if (-not (Test-Path -LiteralPath $python)) {
        python -m venv $venv
    }
    & $python -m pip install --upgrade pip
    & $python -m pip install -r .\DeepSky\requirements.txt pyinstaller
}

if (-not $SkipPyInstaller) {
    & $python -m PyInstaller --noconfirm .\DeepSky.spec
}

$distRoot = Join-Path $root "dist\DeepSky"
if (-not (Test-Path -LiteralPath $distRoot)) {
    throw "Build output not found: $distRoot"
}

$copyItems = @(
    "tools",
    "siril-1.4.3-ucrt64_win",
    "deepsnr_win_1.2.0-0111_ORT_x64_cli",
    "starnet2_win_2.5.0-0204_ORT_x64_cli",
    "samples"
)

foreach ($item in $copyItems) {
    $source = Join-Path $root $item
    if (Test-Path -LiteralPath $source) {
        $dest = Join-Path $distRoot $item
        if (Test-Path -LiteralPath $dest) {
            Remove-Item -LiteralPath $dest -Recurse -Force
        }
        Copy-Item -LiteralPath $source -Destination $dest -Recurse -Force
    }
}

$outputs = Join-Path $distRoot "outputs"
New-Item -ItemType Directory -Force -Path $outputs | Out-Null

$settings = Join-Path $distRoot "settings.json"
if (Test-Path -LiteralPath $settings) {
    Remove-Item -LiteralPath $settings -Force
}

Write-Output "DeepSky build ready: $distRoot"
Write-Output "Run: $distRoot\DeepSky.exe"
