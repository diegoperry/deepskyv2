from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import tifffile
from astropy.io import fits

from app.image_io import crop_image_file


def test_crop_fits_preserves_planes_pixels_and_wcs_reference() -> None:
    with TemporaryDirectory() as folder:
        root = Path(folder)
        source = root / "source.fit"
        output = root / "cropped.fit"
        data = np.arange(3 * 160 * 200, dtype=np.uint16).reshape(3, 160, 200)
        hdu = fits.PrimaryHDU(data)
        hdu.header["CRPIX1"] = 100.0
        hdu.header["CRPIX2"] = 80.0
        hdu.writeto(source)

        size = crop_image_file(source, output, 0.25, 0.20, 0.50, 0.50)

        with fits.open(output, memmap=False) as hdul:
            cropped = hdul[0].data
            assert cropped.shape == (3, 80, 100)
            assert np.array_equal(cropped, data[:, 32:112, 50:150])
            assert float(hdul[0].header["CRPIX1"]) == pytest.approx(50.0)
            assert float(hdul[0].header["CRPIX2"]) == pytest.approx(48.0)
        assert size == (100, 80)


def test_crop_tiff_preserves_rgb_bit_depth_and_pixels() -> None:
    with TemporaryDirectory() as folder:
        root = Path(folder)
        source = root / "source.tif"
        output = root / "cropped.tif"
        data = np.arange(160 * 200 * 3, dtype=np.uint16).reshape(160, 200, 3)
        tifffile.imwrite(source, data, photometric="rgb")

        size = crop_image_file(source, output, 0.10, 0.25, 0.60, 0.50)
        cropped = tifffile.imread(output)

        assert cropped.dtype == np.uint16
        assert cropped.shape == (80, 120, 3)
        assert np.array_equal(cropped, data[40:120, 20:140, :])
        assert size == (120, 80)


def test_crop_rejects_tiny_or_out_of_bounds_regions() -> None:
    with TemporaryDirectory() as folder:
        root = Path(folder)
        source = root / "source.tif"
        output = root / "cropped.tif"
        tifffile.imwrite(source, np.zeros((100, 100, 3), dtype=np.uint16), photometric="rgb")

        with pytest.raises(ValueError, match="at least 64 pixels"):
            crop_image_file(source, output, 0.0, 0.0, 0.20, 0.20)
        with pytest.raises(ValueError, match="outside"):
            crop_image_file(source, output, 0.80, 0.20, 0.30, 0.50)