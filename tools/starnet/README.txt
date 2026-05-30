StarNet2 CLI for Windows - ONNX Runtime Backend
===============================================

StarNet2 removes stars from astronomical images. This package uses the ONNX Runtime
backend and includes the executable, model weights, and package-local runtime DLLs together.

Keep starnet2.exe, StarNet2_weights.onnx, and the bundled DLLs together.
The executable uses the bundled weights next to itself by default. Use
--weights only to override the bundled model file. Do not mix files from
different StarNet2 packages.

Output paths are resolved relative to the current working directory unless
absolute.

Install-Style Layout
--------------------

Portable archive:
  Extract the archive and run from that directory. Keep the executable,
  model weights, README, license, and bundled runtime DLLs as shipped.

Manual install:
  To run starnet2.exe from anywhere, place the files like this:

    C:\Program Files\StarNet2\bin\starnet2.exe
    C:\Program Files\StarNet2\lib\starnet2\<model weights>

  Runtime DLLs must stay next to starnet2.exe.


Quick Start
-----------

Run the executable from the extracted package directory, or invoke it by
path from another working directory:

  starnet2.exe --input input.tif --output starless.tif

By default StarNet2 writes only the starless image. To also write a star
mask, provide a mask filename:

  starnet2.exe --input input.tif --output starless.tif --mask starmask.tif

Optional unscreen star-layer output (disabled unless --unscreen is provided):

  starnet2.exe --input input.tif --output starless.tif --unscreen stars.tif


Options
-------

  -i, --input <file>
      Input image filename. Required for processing. Recommended: TIFF/TIF or PNG.

  -o, --output <file>
      Starless output image filename. Default: starless.jpg.

  -m, --mask <file>
      Optional star mask output filename. Disabled unless provided.

  -n, --unscreen <file>
      Optional unscreen star-layer output filename. Disabled unless provided.

  -w, --weights <file>
      Override the bundled model file. Normally omit this option; the
      executable loads StarNet2_weights.onnx from its package directory.

  -s, --stride <int>
      Tile stride. Default: 256. The value must be even and no larger than
      the 512 pixel processing window.

  -u, --upsample
      Use intermediate 2x upsampling before inference, then downsample outputs.
      This is slower and uses more memory.

  -q, --quiet
      Suppress progress output.

  -e, --eight
      Write TIFF and PNG starless, mask, and unscreen outputs as 8-bit
      instead of the default 16-bit.


Short Option Clustering
-----------------------

Short boolean switches can be combined. For example, -equ is equivalent to
-e -q -u. Only switches without values can be clustered; options that take
values, such as -i, -o, -m, -n, -w, and -s, must be provided separately.


Inputs And Outputs
------------------

Tested input formats are TIFF/TIF and PNG. TIFF inputs are tested in
uncompressed, LZW, and Deflate variants. JPEG/JPG and BMP might work through
OpenCV, but these formats were not tested for this release. JPEG is lossy and
not recommended for scientific or archival data.

Supported input sample depths are 8-bit and 16-bit integer images.
Unsupported input depths, including 32-bit floating-point images, are rejected.
Convert those images to 16-bit integer before running the CLI tool.

Supported inputs are grayscale or RGB/color images. Images with alpha channels
or other channel counts are rejected. Images must be at least 512x512 pixels in
normal mode.

TIFF and PNG starless, mask, and unscreen outputs are saved as 16-bit or
8-bit with --eight. TIFF outputs are always saved with LZW compression. PNG
output uses OpenCV default encoding. Other output formats are written as 8-bit
images.


Legal
-----

See LICENSE.txt for the StarNet2 license. ONNX Runtime license and third-party notices are included with the bundled runtime files.
