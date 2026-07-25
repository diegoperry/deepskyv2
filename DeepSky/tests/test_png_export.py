from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app.web_app import _build_png_export_with_footer, _export_footer_layout


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



    def test_narrow_cropped_export_uses_two_by_two_footer_without_overlap(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "cropped.png"
            logo = root / "logo.png"
            width, height = 756, 1164
            Image.new("RGB", (width, height), (0, 0, 0)).save(source)
            Image.new("RGBA", (412, 141), (0, 255, 0, 255)).save(logo)

            with patch("app.web_app.EXPORT_LOGO_PATH", logo):
                exported = _build_png_export_with_footer(
                    source,
                    telescope="Seestar S50 Smart Telescope",
                    target="Veil Nebula NGC 6992",
                    capture_time="3 hours 32 minutes 20 seconds",
                    date_captured="2026.07.25",
                )

            result = np.asarray(Image.open(BytesIO(exported)).convert("RGB"))
            layout = _export_footer_layout(width, height)
            self.assertEqual(layout["columns"], 2)
            self.assertEqual(layout["rows"], 2)
            self.assertEqual(result.shape[:2], (height, width))

            panel_top = height - layout["overlay_height"]
            content_top = panel_top + layout["top_fade"] + layout["content_pad"]
            text_pixels = np.all(result > 150, axis=2)
            for row in range(2):
                row_top = content_top + row * layout["row_height"]
                row_bottom = min(height, row_top + layout["row_height"])
                for column in range(2):
                    left = layout["left_pad"] + column * (layout["col_width"] + layout["gutter"])
                    right = left + layout["col_width"]
                    self.assertGreater(np.count_nonzero(text_pixels[row_top:row_bottom, left:right]), 0)
                gutter_left = layout["left_pad"] + layout["col_width"]
                gutter_right = gutter_left + layout["gutter"]
                self.assertEqual(np.count_nonzero(text_pixels[row_top:row_bottom, gutter_left:gutter_right]), 0)

            info_right = layout["left_pad"] + layout["info_width"]
            self.assertEqual(np.count_nonzero(text_pixels[content_top:, info_right:]), 0)
            logo_pixels = (result[..., 1] > 200) & (result[..., 0] < 40) & (result[..., 2] < 40)
            ys, xs = np.where(logo_pixels)
            self.assertGreater(xs.size, 0)
            self.assertGreaterEqual(int(xs.min()), info_right)
            self.assertLess(int(xs.max()), width)
            self.assertGreaterEqual(int(ys.min()), panel_top)
            self.assertLess(int(ys.max()), height)

if __name__ == "__main__":
    unittest.main()
