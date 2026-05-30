DeepSNR CLI for Windows - ONNX Runtime Backend
==============================================

DeepSNR is a deep-learning-based tool for noise reduction in astronomical
images. This package uses the ONNX Runtime backend and includes the executable,
model weights, and package-local runtime DLLs together.

Keep deepsnr.exe, DeepSNR_weights_v1.onnx, DeepSNR_weights_v2.onnx, and
the bundled DLLs together. The executable uses the bundled weights next to
itself by default. Use --model 1 or --model 2 to select a bundled model;
use --weights only to override the bundled model file. Do not mix files
from different DeepSNR packages.

Output paths are resolved relative to the current working directory unless
absolute.

Install-Style Layout
--------------------

Portable archive:
  Extract the archive and run from that directory. Keep the executable,
  model weights, README, license, and bundled runtime DLLs as shipped.

Manual install:
  To run deepsnr.exe from anywhere, place the files like this:

    C:\Program Files\DeepSNR\bin\deepsnr.exe
    C:\Program Files\DeepSNR\lib\deepsnr\<model weights>

  Runtime DLLs must stay next to deepsnr.exe.


Quick Start
-----------

Run the executable from the extracted package directory, or invoke it by
path from another working directory:

  deepsnr.exe --input input.tif --output denoised.tif

Model 2 is the default and supports RGB and grayscale images. Model 1 is the
older model line and is intended for RGB images.


Options
-------

  -i, --input <file>
      Input image filename. Required for processing. Recommended: TIFF/TIF or PNG.

  -o, --output <file>
      Denoised output image filename. Default: denoised.jpg.

  -m, --model <1|2>
      Select bundled model version 1 or 2 from the package directory.
      Default: 2. Model 2 supports RGB and grayscale images. Model 1 is
      the older model line and is intended for RGB images. This option is
      ignored when --weights is provided.

  -w, --weights <file>
      Override the bundled model file. Normally omit this option and use
      --model 1 or --model 2 to select DeepSNR_weights_v1.onnx or
      DeepSNR_weights_v2.onnx from the package directory.

  -s, --stride <int>
      Tile stride. Default: 480. The value must be even and no larger than
      the 512 pixel processing window.

  -q, --quiet
      Suppress progress output.

  -e, --eight
      Write TIFF and PNG output as 8-bit instead of the default 16-bit.


Short Option Clustering
-----------------------

Short boolean switches can be combined. For example, -eq is equivalent to
-e -q. Only switches without values can be clustered; options that take values,
such as -i, -o, -m, -w, and -s, must be provided separately.


Inputs And Outputs
------------------

Tested input formats are TIFF/TIF and PNG. TIFF inputs are tested in
uncompressed, LZW, and Deflate variants. JPEG/JPG and BMP might work through
OpenCV, but these formats were not tested for this release. JPEG is lossy and
not recommended for scientific or archival data.

Supported input sample depths are 8-bit and 16-bit integer images.
Unsupported input depths, including 32-bit floating-point images, are rejected.
Convert those images to 16-bit integer before running the CLI tool.

Model 1 accepts RGB/color images only. Model 2 accepts RGB/color and true
grayscale/monochrome images. Images with alpha channels or other channel counts
are rejected. Images must be at least 512x512 pixels.

Both models work on images from monochrome CCD cameras. Drizzle-integrated
images from one-shot color cameras might work as well. To expect good results,
your noise should be uncorrelated high-frequency noise; correlated noise, such
as walking noise, will yield poor results.

TIFF and PNG outputs are saved as 16-bit or 8-bit with --eight. TIFF outputs
are always saved with LZW compression. PNG output uses OpenCV default encoding.
Other output formats are written as 8-bit images.


Legal
-----

The DeepSNR neural-network architecture is based on the NAFNet repository:

  https://github.com/megvii-research/NAFNet

See LICENSE.txt for the DeepSNR license. ONNX Runtime license and third-party notices are included with the bundled runtime files.
