from __future__ import annotations

import unittest

import numpy as np

from app.stretch import astrophotography_stretch


class WeakNebulaStretchTests(unittest.TestCase):
    def _synthetic_stack(self) -> np.ndarray:
        rng = np.random.default_rng(7)
        height, width = 320, 240
        yy, xx = np.mgrid[:height, :width]
        sky = 0.024 + rng.normal(0.0, 0.006, (height, width)).astype(np.float32)
        nebula = np.exp(-(((xx - 122) / 42.0) ** 2 + ((yy - 155) / 67.0) ** 2)).astype(np.float32)
        rgb = np.stack(
            [sky + nebula * 0.055, sky + nebula * 0.025, sky + nebula * 0.014],
            axis=-1,
        )
        rgb[rgb < 0.012] = 0.0
        return (np.clip(rgb, 0.0, 1.0) * 65535.0).astype(np.uint16)

    def test_weak_preset_uses_capped_micro_stretch_and_keeps_sky_dark(self) -> None:
        image = self._synthetic_stack()
        weak_u16 = astrophotography_stretch(image, "seestar_weak_nebula")
        weak = weak_u16.astype(np.float32) / 65535.0
        ordinary = astrophotography_stretch(image, "seestar_slight").astype(np.float32) / 65535.0

        original = image.astype(np.float32) / 65535.0
        positive = np.mean(original, axis=2) > 0.0
        lift = np.mean(weak, axis=2)[positive] / np.mean(original, axis=2)[positive]
        self.assertGreater(float(np.median(lift)), 1.0)
        self.assertLessEqual(float(np.max(lift)), 1.121)
        weak_luminance = np.mean(weak, axis=2)
        ordinary_luminance = np.mean(ordinary, axis=2)
        self.assertLess(np.median(weak_luminance), np.median(ordinary_luminance) * 0.65)
        self.assertLess(np.percentile(weak_luminance, 90), np.percentile(ordinary_luminance, 90) * 0.78)

    def test_weak_preset_preserves_measured_red_brown_direction(self) -> None:
        image = self._synthetic_stack()
        weak = astrophotography_stretch(image, "seestar_weak_nebula").astype(np.float32)
        center = weak[120:190, 95:150]
        channel_medians = np.median(center, axis=(0, 1))

        self.assertGreater(channel_medians[0], channel_medians[1])
        self.assertGreater(channel_medians[1], channel_medians[2])


if __name__ == "__main__":
    unittest.main()
