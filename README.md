# DeepSky V2

DeepSky is an astrophotography processing app for FITS and TIFF files.

It includes:

- A PySide6 desktop app
- A FastAPI web app prototype
- FITS/TIFF loading and 16-bit TIFF output
- Siril, DeepSNR, and StarNet CLI integration
- Before/after previews
- Galaxy background refinement and color-preserving processing

The Windows CLI tool folders are bundled under `tools/` and tracked with Git LFS:

- `tools/siril`
- `tools/deepsnr`
- `tools/starnet`

Clone with Git LFS enabled so the actual binaries download instead of pointer files.

## Local Web Preview

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\windows\install.ps1
powershell -ExecutionPolicy Bypass -File .\deploy\windows\run_web.ps1
```

Then open:

```text
http://127.0.0.1:8000
```

## Web Auth

The web app uses Supabase Auth. Set these environment variables before starting the FastAPI server:

```powershell
$env:SUPABASE_URL="https://your-project.supabase.co"
$env:SUPABASE_ANON_KEY="your-supabase-anon-key"
```

Unsigned users can view the process page, but previews, processing jobs, job status, and downloads require a valid Supabase session.

## Desktop App

```powershell
cd DeepSky
python -m app.main
```

## Windows Build

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
```

## Production Deployment

See `DEPLOYMENT.md` for the Windows VPS + Cloudflare Tunnel deployment path.
