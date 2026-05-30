# DeepSky V2

DeepSky is an astrophotography processing app for FITS and TIFF files.

It includes:

- A PySide6 desktop app
- A FastAPI web app prototype
- FITS/TIFF loading and 16-bit TIFF output
- Siril, DeepSNR, and StarNet CLI integration
- Before/after previews
- Galaxy background refinement and color-preserving processing

The local Windows CLI tools are not committed to this repository. Place them beside the project or configure their paths in local settings.

## Local Web Preview

```powershell
cd DeepSky
python -m uvicorn app.web_app:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

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
