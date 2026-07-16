from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from astropy.io import fits

from app.goal_look import (
    apply_additive_pedestal_duoband_finish,
    apply_measured_nebula_background_neutralization,
    apply_universal_nebula_cosmetic_cleanup,
)
from app.input_analysis import analyze_input_stretch
from app.pipeline import (
    CANONICAL_NEBULA_STAGES,
    _flatten_low_contrast_nebula_gradient,
    _looks_like_low_confidence_high_pedestal_nebula,
)
from app.siril_cli import create_background_extraction_script
from app.settings import default_settings
from app.web_app import (
    _configure_web_pipeline_settings,
    _run_job,
    run_web_legacy_150_pipeline,
)
from app.web_legacy_150_pipeline import (
    PipelineMode as WebLegacyPipelineMode,
    _should_run_early_nebula_deepsnr,
    run_pipeline as expected_web_legacy_150_pipeline,
)
from app.web_legacy_150_goal_look import clean_starless_nebula_background


class WebPipelineRoutingTests(unittest.TestCase):
    def test_web_worker_is_pinned_to_nebula_filament_commit_150(self) -> None:
        self.assertIs(run_web_legacy_150_pipeline, expected_web_legacy_150_pipeline)
        self.assertIn("run_web_legacy_150_pipeline(", inspect.getsource(_run_job))

    def test_early_deepsnr_is_restricted_to_linear_nebula_inputs(self) -> None:
        linear = SimpleNamespace(recommended_mode="linear")
        stretched = SimpleNamespace(recommended_mode="pre_stretched")
        self.assertTrue(
            _should_run_early_nebula_deepsnr(
                object_type="nebula",
                mode=WebLegacyPipelineMode.FULL,
                input_mode="auto",
                use_prestretched=False,
                analysis=linear,
            )
        )
        self.assertFalse(
            _should_run_early_nebula_deepsnr(
                object_type="nebula",
                mode=WebLegacyPipelineMode.FULL,
                input_mode="auto",
                use_prestretched=True,
                analysis=stretched,
            )
        )
        self.assertFalse(
            _should_run_early_nebula_deepsnr(
                object_type="galaxy",
                mode=WebLegacyPipelineMode.FULL,
                input_mode="linear",
                use_prestretched=False,
                analysis=linear,
            )
        )

    def test_starless_background_cleanup_flattens_sky_and_preserves_colored_core(self) -> None:
        height, width = 256, 384
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        gradient = 0.05 + 0.12 * (xx / width) + 0.06 * np.sin(yy / 27.0)
        image = np.stack([gradient * 0.95, gradient * 1.08, gradient], axis=2)
        core = np.exp(-(((xx - width * 0.53) / 35.0) ** 2 + ((yy - height * 0.47) / 28.0) ** 2))
        image[..., 0] += core * 0.42
        image[..., 1] += core * 0.12
        image[..., 2] += core * 0.06
        source = np.clip(image * 65535.0, 0.0, 65535.0).astype(np.uint16)

        cleaned = clean_starless_nebula_background(source).astype(np.float32) / 65535.0
        source_float = source.astype(np.float32) / 65535.0
        sky = core < 0.03
        source_lum = np.mean(source_float, axis=2)
        cleaned_lum = np.mean(cleaned, axis=2)
        source_spread = np.percentile(source_lum[sky], 90) - np.percentile(source_lum[sky], 10)
        cleaned_spread = np.percentile(cleaned_lum[sky], 90) - np.percentile(cleaned_lum[sky], 10)
        self.assertLess(cleaned_spread, source_spread * 0.45)
        self.assertGreater(float(np.max(cleaned[..., 0] - cleaned[..., 1])), 0.18)

    def test_canonical_nebula_stage_order_is_complete(self) -> None:
        self.assertEqual(len(CANONICAL_NEBULA_STAGES), 19)
        self.assertEqual(CANONICAL_NEBULA_STAGES[0], "full-resolution load")
        self.assertEqual(CANONICAL_NEBULA_STAGES[8], "masked RL/deconvolution")
        self.assertEqual(CANONICAL_NEBULA_STAGES[-2], "universal cosmetic cleanup")
        self.assertEqual(CANONICAL_NEBULA_STAGES[-1], "export")

    def test_siril_background_script_is_independent_from_pcc(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = create_background_extraction_script(root / "input.fit", root / "background.fit", root)
            contents = script.read_text(encoding="utf-8").lower()
        self.assertIn("subsky 2", contents)
        self.assertNotIn("pcc ", contents)
        self.assertNotIn("spcc ", contents)

    def test_universal_cosmetic_cleanup_removes_isolated_color_defects(self) -> None:
        height, width = 128, 160
        yy, xx = np.mgrid[:height, :width]
        nebula = 0.04 + np.exp(-(((xx - 82) / 34.0) ** 2 + ((yy - 67) / 28.0) ** 2)) * 0.18
        image = np.repeat(nebula[..., None], 3, axis=2).astype(np.float32)
        image[..., 0] += nebula * 0.20
        image[15, 19] = (1.0, 0.02, 0.01)
        image[101, 137] = (0.01, 1.0, 0.02)
        image[34, 141] = (0.01, 0.02, 1.0)
        source = (np.clip(image, 0.0, 1.0) * 65535.0).astype(np.uint16)
        cleaned = apply_universal_nebula_cosmetic_cleanup(source).astype(np.float32) / 65535.0
        self.assertLess(float(np.max(cleaned[15, 19]) - np.min(cleaned[15, 19])), 0.20)
        self.assertLess(float(np.max(cleaned[101, 137]) - np.min(cleaned[101, 137])), 0.20)
        source_lum = np.mean(source[48:86, 58:108].astype(np.float32) / 65535.0, axis=2)
        clean_lum = np.mean(cleaned[48:86, 58:108], axis=2)
        self.assertLess(float(np.mean(np.abs(clean_lum - source_lum))), 0.01)

    def test_background_neutralization_does_not_gray_nebula_core(self) -> None:
        height, width = 120, 160
        yy, xx = np.mgrid[:height, :width]
        body = np.exp(-(((xx - 84) / 24.0) ** 2 + ((yy - 62) / 20.0) ** 2)).astype(np.float32)
        image = np.zeros((height, width, 3), dtype=np.float32)
        image[:] = (0.035, 0.052, 0.066)
        image[..., 0] += body * 0.32
        image[..., 1] += body * 0.08
        image[..., 2] += body * 0.04
        result = apply_measured_nebula_background_neutralization((image * 65535).astype(np.uint16))
        result = result.astype(np.float32) / 65535.0
        self.assertGreater(float(result[62, 84, 0] - result[62, 84, 2]), 0.20)

    def test_additive_pedestal_finish_compresses_sky_and_keeps_warm_signal(self) -> None:
        rng = np.random.default_rng(84)
        height, width = 240, 360
        yy, xx = np.mgrid[:height, :width]
        body = np.exp(-(((xx - 190) / 30.0) ** 2 + ((yy - 112) / 24.0) ** 2)).astype(np.float32)
        working = np.empty((height, width, 3), dtype=np.float32)
        for channel, pedestal in enumerate((0.26, 0.32, 0.34)):
            working[..., channel] = pedestal + rng.normal(0.0, 0.003, (height, width))
        working[..., 0] += body * 0.20
        working[..., 1] += body * 0.06
        working[..., 2] += body * 0.03
        calibrated = np.clip(working * np.asarray([1.12, 0.91, 0.84], dtype=np.float32), 0.0, 1.0)
        filament = body * np.sin(xx / 3.5).astype(np.float32) * 0.012
        starless = np.repeat((0.14 + body * 0.24 + filament)[..., None], 3, axis=2)

        diagnostics: dict[str, np.ndarray] = {}
        result = apply_additive_pedestal_duoband_finish(
            (starless * 65535.0).astype(np.uint16),
            (np.clip(working, 0.0, 1.0) * 65535.0).astype(np.uint16),
            (calibrated * 65535.0).astype(np.uint16),
            lambda _message: None,
            include_stars=False,
            diagnostics=diagnostics,
        ).astype(np.float32) / 65535.0
        luminance = 0.2126 * result[..., 0] + 0.7152 * result[..., 1] + 0.0722 * result[..., 2]
        core = result[98:126, 174:206]
        sky = result[20:70, 30:100]

        self.assertLess(float(np.median(luminance)), 0.05)
        self.assertGreater(float(np.percentile(luminance, 99.5)), 0.20)
        self.assertGreater(float(np.mean(core[..., 0] - core[..., 2])), 0.08)
        core_luminance = 0.2126 * core[..., 0] + 0.7152 * core[..., 1] + 0.0722 * core[..., 2]
        self.assertGreater(float(np.std(core_luminance[:, 2:] - core_luminance[:, :-2])), 0.001)
        self.assertLess(float(np.std(sky)), 0.035)
        self.assertEqual(
            set(diagnostics),
            {
                "background_confidence",
                "masked_nebula_canvas",
                "nebula_confidence",
                "noise_map",
                "snr_confidence",
                "star_artifact_reject",
                "star_cavity_repair",
                "star_footprint",
            },
        )
        for diagnostic in diagnostics.values():
            self.assertEqual(diagnostic.shape, result.shape)
            self.assertEqual(diagnostic.dtype, np.uint16)

    def test_high_additive_fits_pedestal_is_not_mistaken_for_stretch(self) -> None:
        rng = np.random.default_rng(33)
        height, width = 180, 240
        base = np.empty((3, height, width), dtype=np.float32)
        for channel, pedestal in enumerate((0.257, 0.325, 0.341)):
            base[channel] = pedestal + rng.normal(0.0, 0.0032, (height, width)).astype(np.float32)
        base[:, 80:84, 115:119] = 0.94
        unsigned = np.clip(base * 65535.0, 0.0, 65535.0).astype(np.uint16)

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pedestal_stack.fits"
            fits.PrimaryHDU(unsigned).writeto(path)
            analysis = analyze_input_stretch(path)

        self.assertEqual(analysis.recommended_mode, "linear")
        self.assertEqual(analysis.metrics["additive_pedestal_linear"], 1.0)

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

    def test_borderline_seestar_pedestal_uses_low_confidence_guard(self) -> None:
        rng = np.random.default_rng(7023)
        image = np.full((320, 240, 3), 0.0243, dtype=np.float32)
        image += rng.normal(0.0, 0.00022, image.shape).astype(np.float32)
        image[145:150, 112:117] = 0.80
        image = (np.clip(image, 0.0, 1.0) * 65535.0).astype(np.uint16)

        detected, metrics = _looks_like_low_confidence_high_pedestal_nebula(image)
        self.assertTrue(detected)
        self.assertGreater(metrics["pedestal"], 0.020)
        self.assertLess(metrics["star_masked_relative_contrast"], 0.15)

    def test_low_contrast_gradient_flattening_preserves_compact_signal(self) -> None:
        height, width = 300, 220
        yy, xx = np.mgrid[:height, :width]
        gradient = 0.022 + yy.astype(np.float32) / height * 0.018
        compact = np.exp(-(((xx - 112) / 11.0) ** 2 + ((yy - 145) / 14.0) ** 2)).astype(np.float32) * 0.028
        rgb = np.stack([gradient + compact * 0.82, gradient + compact * 0.95, gradient + compact * 1.15], axis=-1)
        image = (np.clip(rgb, 0.0, 1.0) * 65535.0).astype(np.uint16)

        flattened = _flatten_low_contrast_nebula_gradient(image, lambda _message: None).astype(np.float32) / 65535.0
        before = np.mean(image.astype(np.float32) / 65535.0, axis=2)
        after = np.mean(flattened, axis=2)
        before_gradient = abs(float(np.median(before[-45:]))) - abs(float(np.median(before[:45])))
        after_gradient = abs(float(np.median(after[-45:]))) - abs(float(np.median(after[:45])))
        center_excess = float(np.median(after[138:153, 105:120]) - np.median(after[100:120, 80:100]))

        self.assertLess(abs(after_gradient), abs(before_gradient) * 0.35)
        self.assertGreater(center_excess, 0.010)

    def test_nebula_web_job_uses_single_validated_route(self) -> None:
        settings = _configure_web_pipeline_settings(
            default_settings(),
            object_type="Nebula",
            input_mode="Auto",
            pre_stretched=False,
            stretch_level="Standard",
            siril_deconvolution=False,
            star_setting="Standard",
            pcc_failure_policy="continue",
        )

        self.assertEqual(settings.object_type, "Nebula")
        self.assertEqual(settings.pcc_failure_policy, "continue_without_pcc")
        self.assertEqual(settings.star_handling_mode, "Standard")
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
