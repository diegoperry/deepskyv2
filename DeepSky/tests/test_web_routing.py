from __future__ import annotations

import unittest

import numpy as np

from app.pipeline import _looks_like_low_confidence_high_pedestal_nebula
from app.settings import default_settings
from app.web_app import _configure_web_pipeline_settings


class WebPipelineRoutingTests(unittest.TestCase):
    def test_high_pedestal_low_contrast_nebula_ignores_bright_stars(self) -> None:
        rng = np.random.default_rng(42)
        image = np.full((320, 240, 3), 0.043, dtype=np.float32)
        image += rng.normal(0.0, 0.00014, image.shape).astype(np.float32)
        image[40:44, 50:54] = 1.0
        image[180:185, 130:135] = 0.92
        image = (np.clip(image, 0.0, 1.0) * 65535.0).astype(np.uint16)

        detected, metrics = _looks_like_low_confidence_high_pedestal_nebula(image)
        self.assertTrue(detected)
        self.assertGreater(metrics["pedestal"], 0.025)
        self.assertLess(metrics["star_masked_relative_contrast"], 0.15)

    def test_high_contrast_nebula_does_not_use_low_confidence_guard(self) -> None:
        yy, xx = np.mgrid[:320, :240]
        body = np.exp(-(((xx - 120) / 42.0) ** 2 + ((yy - 160) / 65.0) ** 2)).astype(np.float32)
        lum = 0.012 + body * 0.16
        image = np.repeat(lum[..., None], 3, axis=2)
        image = (np.clip(image, 0.0, 1.0) * 65535.0).astype(np.uint16)

        detected, metrics = _looks_like_low_confidence_high_pedestal_nebula(image)
        self.assertFalse(detected)
        self.assertGreater(metrics["star_masked_relative_contrast"], 0.15)

    def test_nebula_web_job_uses_single_validated_route(self) -> None:
        settings = _configure_web_pipeline_settings(
            default_settings(),
            object_type="Nebula",
            input_mode="Auto",
            pre_stretched=False,
            stretch_level="Standard",
            siril_deconvolution=False,
            star_setting="Starless",
            pcc_failure_policy="continue",
        )

        self.assertEqual(settings.object_type, "Nebula")
        self.assertEqual(settings.pcc_failure_policy, "continue_without_pcc")
        self.assertEqual(settings.star_handling_mode, "Slight Star Reduction")
        self.assertFalse(settings.starless_test_enabled)
        self.assertEqual(settings.color_calibration_mode, "Basic")
        self.assertEqual(settings.nebula_color_separation, "Strong")

    def test_non_nebula_settings_do_not_inherit_nebula_override(self) -> None:
        settings = _configure_web_pipeline_settings(
            default_settings(),
            object_type="Galaxy",
            input_mode="Linear",
            pre_stretched=False,
            stretch_level="Subtle",
            siril_deconvolution=True,
            star_setting="Slight Star Reduction",
            pcc_failure_policy="pause",
        )

        self.assertEqual(settings.object_type, "Galaxy")
        self.assertEqual(settings.pcc_failure_policy, "pause")
        self.assertEqual(settings.star_handling_mode, "Slight Star Reduction")
        self.assertFalse(settings.starless_test_enabled)
        self.assertTrue(settings.siril_deconvolution_enabled)


if __name__ == "__main__":
    unittest.main()
