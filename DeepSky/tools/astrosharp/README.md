# AstroSharp DualPSF integration

DeepSky uses the official AstroSharp DualPSF neural-network weights as the
primary structure stage in its dedicated narrowband pipeline. The integration
is enabled by default, runs after linked stretching/color preparation with
DeepSky's own multiscale sharpening disabled, and replaces masked
Richardson-Lucy deconvolution instead of stacking two sharpeners.

## Production runtime

Production inference is native NumPy and does not require R, Electron, or a
separately installed AstroSharp application. The bundled model archive contains
all official quarter-step PSF models from 1.0 through 8.0. It is included in
the Windows application build by `DeepSky.spec`.

For each frame, DeepSky measures compact-star FWHM and applies AstroSharp's
published FWHM / 2.35 rule. It selects separate DSO and stellar PSF models,
runs both at the upstream default aggressiveness of 1.0, then combines them
with continuous stellar, nebula-signal, background, and edge-protection masks.

Controls:

- `DEEPSKY_ASTROSHARP_ENABLED=0` explicitly uses the legacy Richardson-Lucy fallback.
- `DEEPSKY_ASTROSHARP_MIX` sets the maximum protected contribution (default `0.90`).
- `DEEPSKY_ASTROSHARP_CHUNK_SIZE` sets native inference tiling (default `325`).

## Verifiable activation

Every successful run logs `ASTROSHARP_NATIVE_START`,
`ASTROSHARP_NATIVE_COMPLETE`, and `ASTROSHARP_PIPELINE_VERIFIED`. The job folder
also contains:

- `astrosharp_manifest.json`
- `astrosharp_dso_donor.tif`
- `astrosharp_star_donor.tif`

The manifest records the model-bundle hash, input hash, separate donor hashes,
accepted-output hash, measured FWHM, selected DSO/star PSFs, inference deltas,
accepted pipeline delta, elapsed time, and `actually_executed: true`.

## Official-R validation oracle

`headless_astrosharp_dual.R` remains as a development-only oracle. It runs both
upstream R models so native inference can be compared against the official
implementation. On the NGC 6992 validation crop, native-versus-R RGB
correlation was greater than 0.999998 for both donors, with mean absolute RGB
error below 0.000058.

Upstream: <https://github.com/deepskydetail/AstroSharp/tree/DualPSF>

AstroSharp is MIT-licensed. Model attribution is recorded next to the bundled
weights in `app/models/ASTROSHARP_MODEL_ATTRIBUTION.md`.