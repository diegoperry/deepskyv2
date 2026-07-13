from __future__ import annotations

import unittest

from app.settings import default_settings
from app.web_app import _configure_web_pipeline_settings


class WebPipelineRoutingTests(unittest.TestCase):
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
