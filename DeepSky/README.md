# DeepSky

DeepSky is a Windows desktop app for a first-pass astrophotography processing pipeline using local CLI tools:

- DeepSNR
- StarNet

The app loads FITS or TIFF files, creates a timestamped job folder, saves intermediate images, runs the configured tools, and previews the before/after result.

## Setup

```powershell
cd "C:\Users\diego\Desktop\DeepSky V2\DeepSky"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.main
```

## Supported Inputs

- `.fits`
- `.fit`
- `.fts`
- `.tif`
- `.tiff`

## Outputs

Each run creates a timestamped folder under `outputs/` containing:

- original copied input
- `working.tif`
- `stretched.tif`
- `siril_input.tif`
- `siril_output.fit`
- `calibrated.tif`
- `denoised.tif`
- `starless.tif`
- `stars.tif`
- `final.tif`
- `processing_log.txt`
- preview PNGs

Some files are only produced by modes that need them.

## Tool Paths

DeepSky stores settings in `DeepSky/settings.json`. You can change paths in the UI.

By default the app looks for executables in:

- `tools/starnet/`
- `tools/deepsnr/`
- `tools/siril/`
- extracted folders in the project whose names start with `starnet`, `deepsnr`, or `siril`

## Adjusting CLI Commands

The exact StarNet and DeepSNR command-line flags may vary by release. Edit these constants in `app/cli_tools.py`:

```python
DEEPSNR_COMMAND = ["{exe}", "-i", "{input}", "-o", "{output}"]
STARNET_COMMAND = ["{exe}", "-i", "{input}", "-o", "{output}"]
```

Use `{exe}`, `{input}`, and `{output}` placeholders. For example:

```python
STARNET_COMMAND = ["{exe}", "--input", "{input}", "--output", "{output}"]
```

## Pipeline

1. Select a FITS or TIFF file.
2. Create a timestamped job folder in `outputs/`.
3. Copy the original input into the job folder.
4. Create `working.tif`.
5. Apply the app stretch and save `stretched.tif`.
6. Run Siril color calibration from `siril_input.tif`.
7. Save Siril output as `siril_output.fit`.
8. Export `calibrated.tif` with DeepSky's TIFF writer.
9. Run DeepSNR on `calibrated.tif`.
10. Run StarNet on `denoised.tif`.
11. Create `stars.tif` by subtracting `starless.tif` from `denoised.tif`.
12. Create `final.tif` by recombining `starless.tif` and `stars.tif`.
13. Save logs and previews.

## Color Calibration

Color calibration runs before DeepSNR and StarNet. Siril scripts are saved in the job folder for testing.

Defaults:

- Mode: Basic
- Apply SCNR / Remove Green: off
- Color Saturation: 15
- Debug mode: off

Modes:

- Off
- Basic
- Siril Photometric

Photometric mode can use optional object name, RA/Dec, focal length, and pixel size fields for Siril `pcc`.

The calibration stage writes `calibrated.tif`, saves the generated `.ssf` script in the job folder, captures Siril output in `siril_output.log`, and records settings plus image shape/dtype/min/max diagnostics in `processing_log.txt`.
