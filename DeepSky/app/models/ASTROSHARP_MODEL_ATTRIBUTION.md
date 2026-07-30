# AstroSharp model attribution

`astrosharp_dualpsf_weights.npz` contains the neural-network weights from the
DualPSF branch of AstroSharp by Deep Sky Detail:
https://github.com/deepskydetail/AstroSharp/tree/DualPSF

AstroSharp is distributed under the MIT License. The bundled weights are used
by DeepSky's native, headless inference adapter without changing their learned
parameters. DeepSky independently estimates the stellar FWHM, selects the
matching upstream quarter-step PSF models, runs both the DSO and stellar models,
and blends their luminance donors through continuous signal masks.
