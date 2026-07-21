from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.web_app import _build_png_export_with_footer


class PngExportTests(unittest.TestCase):
    def test_logo_is_small_and_fully_inside_bottom_right_corner(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source.png"
            logo = root / "logo.png"
            Image.new("RGB", (2048, 1187), (0, 0, 0)).save(source)
            Image.new("RGBA", (412, 141), (0, 255, 0, 255)).save(logo)

            with patch("app.web_app.EXPORT_LOGO_PATH", logo):
                exported = _build_png_export_with_footer(
                    source,
                    telescope="Dwarf 3",
                    target="Nebula",
                    capture_time="4hr 50 mins",
                    date_captured="2026.07.18",
                )

            result = np.asarray(Image.open(BytesIO(exported)).convert("RGB"))
            logo_pixels = (result[..., 1] > 200) & (result[..., 0] < 40) & (result[..., 2] < 40)
            ys, xs = np.where(logo_pixels)
            self.assertGreater(xs.size, 0)
            self.assertGreater(int(xs.min()), int(result.shape[1] * 0.75))
            self.assertGreater(int(ys.min()), int(result.shape[0] * 0.88))
            self.assertLess(int(xs.max()), result.shape[1] - 8)
            self.assertLess(int(ys.max()), result.shape[0] - 8)
            self.assertLess(int(xs.max() - xs.min() + 1), int(result.shape[1] * 0.16))


if __name__ == "__main__":
    unittest.main()
