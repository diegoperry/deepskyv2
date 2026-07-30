# Experimental AstroSharp integration

DeepSky can optionally use AstroSharp as a structure donor in the dedicated
narrowband pipeline. It runs after the linked HOO display finish and before
StarNet/star recomposition. The normal DeepSky luminance and color remain the
base image; only bounded luminance detail is accepted through continuous
nebula, star, background, and edge-protection masks.

The feature is disabled by default and fails open. If AstroSharp is missing or
inference fails, DeepSky continues with its normal narrowband result.

## External runtime

AstroSharp's Electron/R application and PSF model files are not copied into
this repository. Stage the upstream `DualPSF` application files separately:

<https://github.com/deepskydetail/AstroSharp/tree/DualPSF>

The configured directory must contain:

- `GetMatrixFun9x9.R`
- `PSF/81_1_FWHM_4.RDS`
- `R-Portable-Win/bin/x64/Rscript.exe`

Set these environment variables before starting DeepSky:

```powershell
[Environment]::SetEnvironmentVariable(
    'DEEPSKY_ASTROSHARP_APP_DIR',
    'C:\path\to\AstroSharp\resources\app',
    'Machine'
)
[Environment]::SetEnvironmentVariable(
    'DEEPSKY_ASTROSHARP_ENABLED',
    '1',
    'Machine'
)
```

Restart the DeepSky process after changing machine environment variables.

## Experimental controls

The validated defaults are conservative:

- `DEEPSKY_ASTROSHARP_PSF=4.0`
- `DEEPSKY_ASTROSHARP_MODEL_STRENGTH=0.50`
- `DEEPSKY_ASTROSHARP_MIX=0.76`
- `DEEPSKY_ASTROSHARP_CHUNK_SIZE=256`

`DEEPSKY_ASTROSHARP_RSCRIPT` can override the bundled Rscript location.

The PSF value is the measured stellar FWHM divided by 2.35, following
AstroSharp's instructions. Do not raise the model strength or final mix
globally without testing multiple representative images for ringing, brittle
filaments, and amplified noise.

## Validation result

On the NGC 6992 validation frame, the reinforced protected blend increased the
final image's gradient/detail metric by 16.82% and its fine-structure metric by
6.33%. Compact-star peaks were not amplified. AstroSharp inference added about 32
seconds for a 1296 x 2304 image on the development machine.

The raw unprotected donor is retained as `astrosharp_raw.tif` in the job
folder for auditing. It is never used directly as the final image.
