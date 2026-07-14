from __future__ import annotations

import unittest

import cv2
import numpy as np

from app.goal_look import (
    apply_color_preserving_nebula_arcsinh,
    apply_conservative_measured_chroma,
    apply_masked_richardson_lucy_nebula,
    apply_multiscale_starless_nebula_detail,
    _apply_nebula_crispness,
    _apply_structured_nebula_visibility_lift,
    apply_measured_color_to_nebula_detail,
    blend_masked_nebula_denoise,
)
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

    def test_low_confidence_preset_is_linked_and_capped_between_weak_and_full(self) -> None:
        image = self._synthetic_stack()
        source = image.astype(np.float32) / 65535.0
        weak = astrophotography_stretch(image, "seestar_weak_nebula").astype(np.float32) / 65535.0
        guarded = astrophotography_stretch(image, "seestar_low_confidence_nebula").astype(np.float32) / 65535.0
        positive = np.mean(source, axis=2) > 0.0
        lift = np.mean(guarded, axis=2)[positive] / np.mean(source, axis=2)[positive]
        self.assertGreater(float(np.median(lift)), 1.12)
        self.assertLessEqual(float(np.max(lift)), 1.721)
        self.assertGreater(float(np.median(guarded)), float(np.median(weak)))

        center = guarded[120:190, 95:150]
        channel_medians = np.median(center, axis=(0, 1))
        self.assertGreater(channel_medians[0], channel_medians[1])
        self.assertGreater(channel_medians[1], channel_medians[2])

    def test_coherent_nebula_preset_preserves_rgb_ratios_at_stronger_lift(self) -> None:
        image = self._synthetic_stack()
        source = image.astype(np.float32) / 65535.0
        stretched = astrophotography_stretch(image, "seestar_coherent_nebula").astype(np.float32) / 65535.0
        valid = (np.mean(source, axis=2) > 0.02) & (source[..., 1] > 1e-4) & (stretched[..., 1] > 1e-4)
        lift = np.mean(stretched, axis=2)[valid] / np.mean(source, axis=2)[valid]
        self.assertGreater(float(np.median(lift)), 1.70)
        self.assertLessEqual(float(np.max(lift)), 4.501)
        before_ratio = np.median(source[..., 0][valid] / source[..., 1][valid])
        after_ratio = np.median(stretched[..., 0][valid] / stretched[..., 1][valid])
        self.assertAlmostEqual(float(after_ratio), float(before_ratio), delta=0.004)

    def test_measured_color_finish_lifts_nebula_without_lifting_empty_sky(self) -> None:
        height, width = 320, 240
        yy, xx = np.mgrid[:height, :width]
        body = np.exp(-(((xx - 120) / 40.0) ** 2 + ((yy - 160) / 62.0) ** 2)).astype(np.float32)
        detail = np.stack(
            [0.012 + body * 0.080, 0.012 + body * 0.038, 0.012 + body * 0.022],
            axis=-1,
        )
        image = (np.clip(detail, 0.0, 1.0) * 65535.0).astype(np.uint16)

        finished = apply_measured_color_to_nebula_detail(image, image).astype(np.float32) / 65535.0
        output_lum = np.mean(finished, axis=2)

        center = output_lum[140:180, 100:140]
        corner = output_lum[45:85, 35:75]
        self.assertGreater(float(np.median(center)), 0.075)
        self.assertLess(float(np.median(corner)), 0.020)

        center_rgb = np.median(finished[140:180, 100:140], axis=(0, 1))
        self.assertGreater(center_rgb[0], center_rgb[1])
        self.assertGreater(center_rgb[1], center_rgb[2])

    def test_extended_chroma_guard_keeps_broad_red_emission(self) -> None:
        height, width = 180, 220
        yy, xx = np.mgrid[:height, :width]
        emission = np.exp(-((xx - 145.0) / 54.0) ** 2).astype(np.float32)
        detail_lum = 0.012 + emission * 0.075
        detail = np.repeat(detail_lum[..., None], 3, axis=2)
        reference = np.stack(
            [detail_lum * (1.0 + emission * 0.75), detail_lum * (1.0 - emission * 0.12), detail_lum * (1.0 - emission * 0.22)],
            axis=-1,
        )
        image = (np.clip(detail, 0.0, 1.0) * 65535.0).astype(np.uint16)
        color = (np.clip(reference, 0.0, 1.0) * 65535.0).astype(np.uint16)

        guarded = apply_conservative_measured_chroma(
            image,
            color,
            preserve_extended_chroma=True,
        ).astype(np.float32) / 65535.0
        band = emission > 0.65
        sky = emission < 0.02
        red_excess = guarded[..., 0] - (guarded[..., 1] + guarded[..., 2]) * 0.5

        self.assertGreater(float(np.median(red_excess[band])), 0.010)
        self.assertGreater(float(np.median(np.mean(guarded[band], axis=1))), float(np.median(detail_lum[band])))
        self.assertLess(float(np.median(np.mean(guarded[sky], axis=1))), 0.020)

    def test_nebula_crispness_restores_object_structure_not_sky(self) -> None:
        height, width = 180, 220
        yy, xx = np.mgrid[:height, :width]
        object_mask = np.exp(-(((xx - 112) / 48.0) ** 2 + ((yy - 90) / 40.0) ** 2)).astype(np.float32)
        ridges = np.sin(xx * 0.34).astype(np.float32) * object_mask * 0.018
        source_lum = np.clip(0.025 + object_mask * 0.13 + ridges, 0.0, 1.0)
        smooth_lum = cv2.GaussianBlur(source_lum, (0, 0), 2.4)
        smooth_rgb = np.repeat(smooth_lum[..., None], 3, axis=2)
        structure = np.clip(np.abs(ridges) / 0.018, 0.0, 1.0)
        star_halo = np.zeros_like(source_lum)

        crisp = _apply_nebula_crispness(smooth_rgb, source_lum, object_mask, structure, star_halo)
        crisp_lum = np.mean(crisp, axis=2)
        object_region = object_mask > 0.45
        sky_region = object_mask < 0.02

        smooth_detail = smooth_lum - cv2.GaussianBlur(smooth_lum, (0, 0), 1.0)
        crisp_detail = crisp_lum - cv2.GaussianBlur(crisp_lum, (0, 0), 1.0)
        self.assertGreater(float(np.std(crisp_detail[object_region])), float(np.std(smooth_detail[object_region])) * 1.08)
        self.assertLess(float(np.max(np.abs(crisp_lum[sky_region] - smooth_lum[sky_region]))), 0.0005)

    def test_structured_visibility_lift_is_generic_and_sky_safe(self) -> None:
        height, width = 160, 200
        yy, xx = np.mgrid[:height, :width]
        signal = np.exp(-(((xx - 105) / 38.0) ** 2 + ((yy - 82) / 46.0) ** 2)).astype(np.float32)
        structure = np.clip(np.abs(np.sin(xx * 0.25)) * signal, 0.0, 1.0).astype(np.float32)
        star_halo = np.zeros_like(signal)
        lum = np.full((height, width), 0.025, dtype=np.float32)
        lum += signal * 0.075
        image = np.repeat(lum[..., None], 3, axis=2)

        lifted = _apply_structured_nebula_visibility_lift(image, signal, structure, star_halo)
        lifted_lum = np.mean(lifted, axis=2)
        object_region = (signal > 0.55) & (structure > 0.35)
        sky_region = signal < 0.01

        self.assertGreater(float(np.median(lifted_lum[object_region])), float(np.median(lum[object_region])) + 0.012)
        self.assertLess(float(np.max(np.abs(lifted_lum[sky_region] - lum[sky_region]))), 0.0008)

    def test_masked_richardson_lucy_improves_nebula_edges_and_preserves_rgb(self) -> None:
        height, width = 180, 220
        yy, xx = np.mgrid[:height, :width]
        body = np.exp(-(((xx - 112) / 45.0) ** 2 + ((yy - 90) / 55.0) ** 2)).astype(np.float32)
        filament = np.exp(-((xx - (105 + 10 * np.sin(yy * 0.09))) / 2.2) ** 2).astype(np.float32) * body
        lum = 0.018 + body * 0.045 + filament * 0.055
        lum = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 1.4)
        rgb = np.stack([lum * 1.28, lum * 0.92, lum * 0.70], axis=-1)
        image = (np.clip(rgb, 0.0, 1.0) * 65535.0).astype(np.uint16)

        restored = apply_masked_richardson_lucy_nebula(image, iterations=5).astype(np.float32) / 65535.0
        before_lum = np.mean(image.astype(np.float32) / 65535.0, axis=2)
        after_lum = np.mean(restored, axis=2)
        object_region = body > 0.35
        before_edge = cv2.Laplacian(before_lum, cv2.CV_32F)
        after_edge = cv2.Laplacian(after_lum, cv2.CV_32F)
        self.assertGreater(float(np.std(after_edge[object_region])), float(np.std(before_edge[object_region])) * 1.01)

        valid = object_region & (restored[..., 1] > 1e-4)
        ratio_before = np.median((image[..., 0].astype(np.float32) + 1.0)[valid] / (image[..., 1].astype(np.float32) + 1.0)[valid])
        ratio_after = np.median((restored[..., 0] + 1e-5)[valid] / (restored[..., 1] + 1e-5)[valid])
        self.assertAlmostEqual(float(ratio_after), float(ratio_before), delta=0.02)

    def test_masked_denoise_blend_recovers_filaments_without_changing_sky(self) -> None:
        source = self._synthetic_stack().astype(np.float32) / 65535.0
        denoised = cv2.GaussianBlur(source, (0, 0), 2.2)
        blended = blend_masked_nebula_denoise(
            (source * 65535.0).astype(np.uint16),
            (denoised * 65535.0).astype(np.uint16),
        ).astype(np.float32) / 65535.0
        source_lum = np.mean(source, axis=2)
        denoised_lum = np.mean(denoised, axis=2)
        blended_lum = np.mean(blended, axis=2)
        object_region = source_lum > np.percentile(source_lum, 82.0)
        sky_region = source_lum < np.percentile(source_lum, 28.0)
        source_band = source_lum - cv2.GaussianBlur(source_lum, (0, 0), 1.2)
        denoised_band = denoised_lum - cv2.GaussianBlur(denoised_lum, (0, 0), 1.2)
        blended_band = blended_lum - cv2.GaussianBlur(blended_lum, (0, 0), 1.2)
        self.assertGreater(float(np.std(blended_band[object_region])), float(np.std(denoised_band[object_region])))
        self.assertLess(float(np.mean(np.abs(blended_lum[sky_region] - denoised_lum[sky_region]))), 0.0015)

    def test_starless_multiscale_contrast_increases_filament_separation(self) -> None:
        height, width = 180, 220
        yy, xx = np.mgrid[:height, :width]
        body = np.exp(-(((xx - 110) / 50.0) ** 2 + ((yy - 92) / 58.0) ** 2)).astype(np.float32)
        texture = (np.sin(xx * 0.22) * 0.008 + np.sin(xx * 0.62) * 0.003).astype(np.float32) * body
        lum = np.clip(0.018 + body * 0.085 + texture, 0.0, 1.0)
        rgb = np.stack([lum * 1.18, lum * 0.96, lum * 0.82], axis=-1)
        image = (np.clip(rgb, 0.0, 1.0) * 65535.0).astype(np.uint16)
        enhanced = apply_multiscale_starless_nebula_detail(image).astype(np.float32) / 65535.0
        before_lum = np.mean(image.astype(np.float32) / 65535.0, axis=2)
        after_lum = np.mean(enhanced, axis=2)
        region = body > 0.40
        before_band = before_lum - cv2.GaussianBlur(before_lum, (0, 0), 2.0)
        after_band = after_lum - cv2.GaussianBlur(after_lum, (0, 0), 2.0)
        self.assertGreater(float(np.std(after_band[region])), float(np.std(before_band[region])) * 1.06)
    def test_linked_arcsinh_lifts_faint_signal_without_shifting_rgb_ratios(self) -> None:
        height, width = 180, 220
        yy, xx = np.mgrid[:height, :width]
        body = np.exp(-(((xx - 112) / 52.0) ** 2 + ((yy - 92) / 58.0) ** 2)).astype(np.float32)
        lum = 0.008 + body * 0.070
        rgb = np.stack([lum * 1.34, lum * 0.96, lum * 0.78], axis=-1)
        image = (np.clip(rgb, 0.0, 1.0) * 65535.0).astype(np.uint16)

        stretched = apply_color_preserving_nebula_arcsinh(image).astype(np.float32) / 65535.0
        source = image.astype(np.float32) / 65535.0
        faint = (body > 0.12) & (body < 0.35)
        self.assertGreater(float(np.median(np.mean(stretched, axis=2)[faint])), float(np.median(np.mean(source, axis=2)[faint])))

        valid = faint & (source[..., 1] > 1e-4) & (stretched[..., 1] > 1e-4)
        before_ratio = np.median(source[..., 0][valid] / source[..., 1][valid])
        after_ratio = np.median(stretched[..., 0][valid] / stretched[..., 1][valid])
        self.assertAlmostEqual(float(after_ratio), float(before_ratio), delta=0.004)

    def test_measured_color_boosts_continuous_red_cyan_separation_only_in_nebula(self) -> None:
        height, width = 220, 260
        yy, xx = np.mgrid[:height, :width]
        warm = np.exp(-(((xx - 105) / 42.0) ** 2 + ((yy - 112) / 60.0) ** 2)).astype(np.float32)
        cyan = np.exp(-(((xx - 158) / 32.0) ** 2 + ((yy - 108) / 55.0) ** 2)).astype(np.float32)
        signal = np.clip(warm + cyan, 0.0, 1.0)
        detail_lum = 0.010 + signal * 0.080
        detail = np.repeat(detail_lum[..., None], 3, axis=2)
        reference = np.stack(
            [0.010 + warm * 0.115 + cyan * 0.035,
             0.010 + warm * 0.052 + cyan * 0.090,
             0.010 + warm * 0.035 + cyan * 0.120],
            axis=-1,
        )
        finished = apply_measured_color_to_nebula_detail(
            (detail * 65535.0).astype(np.uint16),
            (reference * 65535.0).astype(np.uint16),
        ).astype(np.float32) / 65535.0

        warm_region = (warm > 0.65) & (cyan < 0.20)
        cyan_region = (cyan > 0.65) & (warm < 0.20)
        warm_ratio = np.median((finished[..., 0] + 0.005)[warm_region] / (((finished[..., 1] + finished[..., 2]) * 0.5 + 0.005)[warm_region]))
        cyan_ratio = np.median((finished[..., 0] + 0.005)[cyan_region] / (((finished[..., 1] + finished[..., 2]) * 0.5 + 0.005)[cyan_region]))
        self.assertGreater(float(warm_ratio), 1.18)
        self.assertLess(float(cyan_ratio), 0.82)

        sky = signal < 0.01
        sky_chroma = np.max(finished, axis=2) - np.min(finished, axis=2)
        self.assertLess(float(np.percentile(sky_chroma[sky], 95.0)), 0.010)


if __name__ == "__main__":
    unittest.main()
