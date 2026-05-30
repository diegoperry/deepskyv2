param(
    [Parameter(Mandatory = $true)]
    [string]$TunnelToken,

    [string]$CloudflaredPath = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$toolsDir = Join-Path $repoRoot "tools\cloudflared"

if (-not $CloudflaredPath) {
    New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null
    $CloudflaredPath = Join-Path $toolsDir "cloudflared.exe"
}

if (-not (Test-Path -LiteralPath $CloudflaredPath)) {
    $url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    Invoke-WebRequest -Uri $url -OutFile $CloudflaredPath
}

& $CloudflaredPath service install $TunnelToken

Write-Output "Cloudflare Tunnel service installed."
Write-Output "Make sure the DeepSky web app is running on the origin configured in Cloudflare."
