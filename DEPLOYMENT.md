# DeepSky Production Deployment

DeepSky currently depends on Windows CLI binaries for Siril, DeepSNR, and StarNet, so the production host should be a Windows VPS or Windows Server VM.

## What You Need

- Windows Server VPS or VM
- Python 3.11 or 3.12
- Git
- Git LFS
- A Cloudflare account and domain for HTTPS

## Server Setup

Open PowerShell on the Windows server:

```powershell
git lfs install
git clone https://github.com/diegoperry/deepskyv2.git
cd deepskyv2
git lfs pull
powershell -ExecutionPolicy Bypass -File .\deploy\windows\install.ps1
```

Start the web app:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\windows\run_web.ps1
```

The app listens on `127.0.0.1:8000` by default.

To keep it running after reboot, install it as a Windows Scheduled Task:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\windows\install_web_task.ps1
```

## Cloudflare Tunnel

Create a named Cloudflare Tunnel in the Cloudflare dashboard and route your domain to:

```text
http://127.0.0.1:8000
```

Then install the tunnel as a Windows service using the token Cloudflare gives you:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\windows\install_cloudflare_tunnel_service.ps1 -TunnelToken "YOUR_TOKEN_HERE"
```

## Production Notes

- Keep `outputs/` on a disk with enough space for FITS/TIFF uploads and intermediate files.
- This first deployment runs one processing worker at a time to avoid exhausting RAM/CPU.
- For public launch, put upload size and cleanup policies in place before sharing widely.
- The bundled `tools/` folder is large and uses Git LFS, so server clones must run `git lfs pull`.
