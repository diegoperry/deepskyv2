from __future__ import annotations

import inspect
import json
import re
import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from astropy.io import fits

from app.goal_look import (
    apply_additive_pedestal_duoband_finish,
    apply_measured_nebula_background_neutralization,
    apply_universal_nebula_cosmetic_cleanup,
)
from app.input_analysis import analyze_input_stretch
from app.narrowband_finish import apply_starnet_guided_narrowband_polish
from app.pipeline import (
    CANONICAL_NEBULA_STAGES,
    _flatten_low_contrast_nebula_gradient,
    _looks_like_low_confidence_high_pedestal_nebula,
    _needs_low_signal_galaxy_safety,
    _orient_like_reference as orient_current_pipeline_like_reference,
)
from app.siril_cli import create_background_extraction_script, create_stacked_rgb_narrowband_script
from app.settings import default_settings
from app.web_app import (
    app,
    WebJob,
    _configure_web_pipeline_settings,
    _job_response,
    blog_article_page,
    blog_page,
    create_job,
    docs_page,
    index,
    process_page,
    robots_txt,
    sitemap_xml,
    _realesrgan_error_message,
    _run_job,
    run_web_legacy_150_pipeline,
)
from app.cli_tools import ToolExecutionError
from app.web_legacy_150_pipeline import (
    PipelineMode as WebLegacyPipelineMode,
    _orient_like_reference as orient_web_pipeline_like_reference,
    _prepare_narrowband_starnet_input,
    _needs_low_signal_galaxy_safety as _needs_web_low_signal_galaxy_safety,
    _run_dedicated_narrowband_pipeline,
    _should_run_early_nebula_deepsnr,
    run_pipeline as expected_web_legacy_150_pipeline,
)
from app.web_legacy_150_goal_look import (
    _apply_reference_nebula_tone_grade,
    _neutralize_nebula_sky_field,
    clean_starless_nebula_background,
)


class WebPipelineRoutingTests(unittest.TestCase):
    def test_low_signal_galaxy_safety_only_rejects_short_compact_stacks(self) -> None:
        compact = SimpleNamespace(metrics={"raw_p999": 0.0217, "bright_fraction": 0.00012})
        strong = SimpleNamespace(metrics={"raw_p999": 0.0800, "bright_fraction": 0.00200})
        with TemporaryDirectory() as folder:
            root = Path(folder)
            short = root / "short.fit"
            long = root / "long.fit"
            for path, count in ((short, 20), (long, 180)):
                hdu = fits.PrimaryHDU(np.zeros((3, 8, 8), dtype=np.uint16))
                hdu.header["EXPTIME"] = 10.0
                hdu.header["STACKCNT"] = count
                hdu.writeto(path)

            self.assertEqual(_needs_low_signal_galaxy_safety(short, compact, "galaxy"), (True, 200.0))
            self.assertEqual(_needs_low_signal_galaxy_safety(long, compact, "galaxy"), (False, 1800.0))
            self.assertEqual(_needs_low_signal_galaxy_safety(short, strong, "galaxy"), (False, None))
            self.assertEqual(_needs_low_signal_galaxy_safety(short, compact, "nebula"), (False, None))

    def test_production_web_galaxy_route_protects_only_short_weak_stacks(self) -> None:
        compact = SimpleNamespace(metrics={"raw_p999": 0.0217, "bright_fraction": 0.00012})
        strong = SimpleNamespace(metrics={"raw_p999": 0.0800, "bright_fraction": 0.00200})
        with TemporaryDirectory() as folder:
            root = Path(folder)
            short = root / "short.fit"
            long = root / "long.fit"
            for path, count in ((short, 20), (long, 180)):
                hdu = fits.PrimaryHDU(np.zeros((3, 8, 8), dtype=np.uint16))
                hdu.header["EXPTIME"] = 10.0
                hdu.header["STACKCNT"] = count
                hdu.writeto(path)

            self.assertEqual(_needs_web_low_signal_galaxy_safety(short, compact, "galaxy"), (True, 200.0))
            self.assertEqual(_needs_web_low_signal_galaxy_safety(long, compact, "galaxy"), (False, 1800.0))
            self.assertEqual(_needs_web_low_signal_galaxy_safety(short, strong, "galaxy"), (False, None))

        route = inspect.getsource(run_web_legacy_150_pipeline)
        self.assertNotIn("protected Siril deconvolution forced on", route)
        self.assertIn("galaxy_narrowband_requested and not low_signal_galaxy_safety", route)
        self.assertIn("final.exists() and not low_signal_galaxy_safety", route)
        self.assertIn("or low_signal_galaxy_safety", route)
        self.assertIn("skipping catalog PCC/Siril calibration", route)
        worker = inspect.getsource(_run_job)
        self.assertIn('settings.siril_deconvolution_enabled = object_type == "Galaxy" and bool(siril_deconvolution)', worker)
        creator = inspect.getsource(create_job)
        self.assertIn("selected_siril_deconvolution = False", creator)
        self.assertIn("Narrowband Color cannot enable it", worker)

    def test_finished_job_displays_native_png_instead_of_downsampled_preview(self) -> None:
        job = WebJob(
            id="native-preview",
            user_id="user-1",
            status="finished",
            result={
                "before_preview": Path("before_preview.png"),
                "after_preview": Path("after_preview.png"),
                "png": Path("final.png"),
                "final": Path("final.tif"),
            },
        )

        payload = _job_response(job)

        self.assertEqual(payload["after_preview"], "/api/jobs/native-preview/file/png?inline=1")
        self.assertEqual(payload["png"], "/api/jobs/native-preview/file/png")
        self.assertNotIn("after_preview?inline=1", payload["after_preview"])

    def test_public_pages_expose_blog_and_search_metadata(self) -> None:
        home = index()
        docs = docs_page()
        blog = blog_page()
        process = process_page()

        self.assertIn('href="/blog"', home)
        self.assertIn('href="/blog"', docs)
        self.assertIn('href="/blog"', process)
        self.assertIn("<title>Astrophotography Blog | DeepSky Processor</title>", blog)
        self.assertIn('rel="canonical" href="https://app.deepskyprocessor.com/blog"', blog)
        self.assertIn('"@type": "Blog"', blog)
        self.assertIn("The #1 Blog For Smart Telescope Users.", blog)
        self.assertIn("you're in the right place.", blog)
        self.assertIn('href="/blog/ai-for-astrophotography"', blog)
        self.assertIn("AI for Astrophotography: Helpful Assistant or Data Fabricator?", blog)

    def test_ai_astrophotography_article_has_complete_seo_and_images(self) -> None:
        article = blog_article_page()

        self.assertIn("<h1>AI for Astrophotography: Helpful Assistant or Data Fabricator?</h1>", article)
        self.assertIn('rel="canonical" href="https://app.deepskyprocessor.com/blog/ai-for-astrophotography"', article)
        self.assertIn('"@type": "BlogPosting"', article)
        self.assertIn('"@type": "BreadcrumbList"', article)
        self.assertIn('property="og:image"', article)
        self.assertIn('property="article:published_time" content="2026-07-27"', article)
        self.assertIn('href="/process"', article)
        for image_name in (
            "ai-astrophotography-cover-v2.png",
            "gemini-fits-processing-prompt.png",
            "generative-ai-veil-nebula-result.png",
            "deepsky-ai-processed-veil-nebula.png",
        ):
            self.assertIn(f"/static/blog/ai-for-astrophotography/{image_name}", article)
        self.assertNotIn("deepsky-non-generative-ai-result.png", article)
        self.assertIn('property="og:image" content="https://app.deepskyprocessor.com/static/blog/ai-for-astrophotography/ai-astrophotography-cover-v2.png"', article)
        self.assertIn('width="1983" height="793" fetchpriority="high"', article)
        self.assertIn('<header class="site-header">', article)
        self.assertNotIn("header {\n      position: sticky;", article)
        self.assertNotIn('alt=""', article)
        json_ld_blocks = re.findall(r'<script type="application/ld\+json">\s*(.*?)\s*</script>', article, re.S)
        self.assertEqual(len(json_ld_blocks), 2)
        self.assertEqual([json.loads(block)["@type"] for block in json_ld_blocks], ["BlogPosting", "BreadcrumbList"])

    def test_robots_and_sitemap_make_public_pages_discoverable(self) -> None:
        robots = robots_txt().body.decode("utf-8")
        sitemap_response = sitemap_xml()
        sitemap = sitemap_response.body.decode("utf-8")

        self.assertIn("Disallow: /api/", robots)
        self.assertIn("https://app.deepskyprocessor.com/sitemap.xml", robots)
        self.assertEqual(sitemap_response.media_type, "application/xml")
        for path in ("/", "/blog", "/blog/ai-for-astrophotography", "/docs", "/process"):
            self.assertIn(f"<loc>https://app.deepskyprocessor.com{path}</loc>", sitemap)
        self.assertIn("/blog", {route.path for route in app.routes})
        self.assertIn("/blog/ai-for-astrophotography", {route.path for route in app.routes})
        self.assertIn("/robots.txt", {route.path for route in app.routes})
        self.assertIn("/sitemap.xml", {route.path for route in app.routes})

    def test_pipeline_corrects_only_high_confidence_orientation_flips(self) -> None:
        height, width = 180, 260
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        reference_lum = (
            0.025
            + np.exp(-(((xx - 61.0) / 22.0) ** 2 + ((yy - 47.0) / 31.0) ** 2)) * 0.52
            + np.exp(-(((xx - 194.0) / 35.0) ** 2 + ((yy - 126.0) / 18.0) ** 2)) * 0.27
        )
        reference = np.stack(
            [reference_lum * 1.12, reference_lum * 0.94, reference_lum * 0.81],
            axis=2,
        ).astype(np.float32)
        flipped = np.flipud(reference) ** 0.91

        for orient in (orient_web_pipeline_like_reference, orient_current_pipeline_like_reference):
            messages: list[str] = []
            corrected = orient(flipped, reference, messages.append, "test image")
            self.assertLess(float(np.mean(np.abs(corrected - (reference ** 0.91)))), 1e-5)
            self.assertTrue(any("orientation corrected with vertical flip" in message for message in messages))

            ambiguous = np.full_like(reference, 0.08)
            preserved = orient(ambiguous, ambiguous, messages.append, "ambiguous image")
            self.assertTrue(np.array_equal(preserved, ambiguous))

    def test_creative_color_finish_is_removed_from_ui_and_api(self) -> None:
        html = process_page()
        self.assertNotIn("Creative Color Finish", html)
        self.assertNotIn("creative_color_finish", html)
        self.assertNotIn("data-creative-finish", html)
        self.assertNotIn("/finish/creative-color", {route.path for route in app.routes})

    def test_web_worker_is_pinned_to_nebula_filament_commit_150(self) -> None:
        self.assertIs(run_web_legacy_150_pipeline, expected_web_legacy_150_pipeline)
        self.assertIn("run_web_legacy_150_pipeline(", inspect.getsource(_run_job))

    def test_realesrgan_windows_missing_dll_error_is_actionable(self) -> None:
        message = _realesrgan_error_message(ToolExecutionError("realesrgan-ncnn-vulkan.exe", 3221225781, ""))
        self.assertIn("runtime DLL", message)
        self.assertIn("Visual C++", message)
        self.assertIn("Vulkan", message)

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

    def test_final_nebula_sky_neutralization_removes_blobs_and_preserves_object(self) -> None:
        height, width = 240, 320
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        blobs = 0.035 * np.sin(xx / 34.0) + 0.028 * np.cos(yy / 27.0)
        sky = 0.045 + blobs
        body = np.exp(-(((xx - 168.0) / 42.0) ** 2 + ((yy - 121.0) / 36.0) ** 2))
        image = np.stack(
            [sky + body * 0.30, sky * 1.10 + body * 0.12, sky * 0.92 + body * 0.08],
            axis=2,
        ).astype(np.float32)
        signal = cv2.GaussianBlur(body.astype(np.float32), (0, 0), 3.0)

        neutralized = _neutralize_nebula_sky_field(image, signal)
        before_lum = np.mean(image, axis=2)
        after_lum = np.mean(neutralized, axis=2)
        sky_mask = body < 0.025
        before_spread = float(np.percentile(before_lum[sky_mask], 90) - np.percentile(before_lum[sky_mask], 10))
        after_spread = float(np.percentile(after_lum[sky_mask], 90) - np.percentile(after_lum[sky_mask], 10))
        core = body > 0.70

        self.assertLess(after_spread, before_spread * 0.18)
        self.assertGreater(float(np.median(after_lum[core]) - np.median(after_lum[sky_mask])), 0.09)

    def test_reference_nebula_grade_lifts_sky_and_restrains_red_core(self) -> None:
        height, width = 180, 240
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        body = np.exp(-(((xx - 122.0) / 34.0) ** 2 + ((yy - 91.0) / 29.0) ** 2))
        image = np.zeros((height, width, 3), dtype=np.float32)
        image[:] = np.array([0.011, 0.008, 0.009], dtype=np.float32)
        image += body[..., None] * np.array([0.48, 0.19, 0.16], dtype=np.float32)
        signal = cv2.GaussianBlur(body.astype(np.float32), (0, 0), 2.5)

        graded = _apply_reference_nebula_tone_grade(image, signal)
        sky = body < 0.01
        core = body > 0.80

        sky_rgb = np.median(graded[sky], axis=0)
        self.assertGreater(float(np.mean(sky_rgb)), 0.040)
        self.assertLess(float(np.max(sky_rgb) / np.maximum(np.min(sky_rgb), 1e-5)), 1.025)
        self.assertLess(float(np.median(graded[core, 0])), float(np.median(image[core, 0])) * 0.82)
        self.assertGreater(float(np.median(graded[core, 1] / np.maximum(graded[core, 0], 1e-5))), 0.40)

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
        self.assertTrue(settings.narrowband_color_enabled)

    def test_starless_web_setting_is_preserved_for_all_object_types(self) -> None:
        for object_type in ("Nebula", "Galaxy", "Star Cluster"):
            settings = _configure_web_pipeline_settings(
                default_settings(),
                object_type=object_type,
                input_mode="Auto",
                pre_stretched=False,
                stretch_level="Standard",
                siril_deconvolution=False,
                star_setting="Starless",
                pcc_failure_policy="continue",
            )
            self.assertEqual(settings.star_handling_mode, "Starless")

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


    def test_narrowband_starnet_safety_lifts_only_exceptionally_dim_frames(self) -> None:
        dim = np.full((120, 160, 3), 0.001, dtype=np.float32)
        yy, xx = np.mgrid[:120, :160]
        signal = np.exp(-(((xx - 80.0) / 19.0) ** 2 + ((yy - 60.0) / 25.0) ** 2)).astype(np.float32)
        dim += signal[..., None] * np.array([0.010, 0.006, 0.004], dtype=np.float32)

        untouched = _prepare_narrowband_starnet_input(dim, False)
        lifted = _prepare_narrowband_starnet_input(dim, True).astype(np.float32) / 65535.0
        bright = np.full_like(dim, 0.080)

        self.assertTrue(np.array_equal(untouched, dim))
        self.assertGreater(float(np.percentile(lifted, 99.8)), float(np.percentile(dim, 99.8)) * 2.0)
        self.assertTrue(np.array_equal(_prepare_narrowband_starnet_input(bright, True), bright))


    def test_narrowband_starnet_polish_applies_option04_stars_and_dark_sky(self) -> None:
        height, width = 240, 320
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        nebula = np.exp(-(((xx - 165.0) / 70.0) ** 2 + ((yy - 120.0) / 55.0) ** 2))
        starless = np.empty((height, width, 3), dtype=np.float32)
        starless[:] = np.array([0.025, 0.030, 0.035], dtype=np.float32)
        starless += nebula[..., None] * np.array([0.16, 0.10, 0.19], dtype=np.float32)
        source = starless.copy()
        stars = (
            (72, 64, 1.55, 0.86, (1.00, 0.82, 0.64)),
            (242, 92, 2.00, 0.72, (0.72, 0.84, 1.00)),
            (188, 178, 1.25, 0.64, (1.00, 0.72, 0.55)),
        )
        star_area = np.zeros((height, width), dtype=bool)
        for center_x, center_y, sigma, amplitude, color in stars:
            profile = np.exp(
                -(((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma * sigma))
            ).astype(np.float32) * amplitude
            source += profile[..., None] * np.asarray(color, dtype=np.float32)
            star_area |= ((xx - center_x) ** 2 + (yy - center_y) ** 2) < 64.0
        source = np.clip(source, 0.0, 1.0)

        finished = apply_starnet_guided_narrowband_polish(
            np.round(source * 65535.0).astype(np.uint16),
            np.round(starless * 65535.0).astype(np.uint16),
        ).astype(np.float32) / 65535.0
        source_lum = source[..., 0] * 0.2126 + source[..., 1] * 0.7152 + source[..., 2] * 0.0722
        finished_lum = finished[..., 0] * 0.2126 + finished[..., 1] * 0.7152 + finished[..., 2] * 0.0722

        self.assertLess(float(finished_lum[64, 72]), float(source_lum[64, 72]) * 0.78)
        self.assertGreater(float(finished_lum[64, 72]), float(source_lum[64, 72]) * 0.50)
        self.assertLess(
            np.count_nonzero(finished_lum > 0.35),
            np.count_nonzero(source_lum > 0.35) * 0.60,
        )
        quiet_sky = (~star_area) & (nebula < 0.02)
        self.assertLess(
            float(np.median(finished_lum[quiet_sky])),
            float(np.median(source_lum[quiet_sky])) * 0.94,
        )
        self.assertLess(float(np.percentile(np.abs(finished[~star_area] - source[~star_area]), 99.0)), 0.030)
        source_color = source[64, 72] / np.max(source[64, 72])
        finished_color = finished[64, 72] / np.max(finished[64, 72])
        self.assertLess(float(np.max(np.abs(finished_color - source_color))), 0.015)

    def test_narrowband_color_is_always_on_for_nebula_and_galaxy(self) -> None:
        nebula = _configure_web_pipeline_settings(
            default_settings(),
            object_type="Nebula",
            input_mode="Auto",
            pre_stretched=False,
            stretch_level="Standard",
            siril_deconvolution=False,
            star_setting="Standard",
            pcc_failure_policy="continue",
        )
        galaxy = _configure_web_pipeline_settings(
            default_settings(),
            object_type="Galaxy",
            input_mode="Auto",
            pre_stretched=False,
            stretch_level="Standard",
            siril_deconvolution=False,
            star_setting="Standard",
            pcc_failure_policy="continue",
        )
        natural_nebula = _configure_web_pipeline_settings(
            default_settings(),
            object_type="Nebula",
            input_mode="Auto",
            pre_stretched=False,
            stretch_level="Standard",
            siril_deconvolution=False,
            star_setting="Standard",
            pcc_failure_policy="continue",
            narrowband_color=False,
        )

        self.assertTrue(nebula.narrowband_color_enabled)
        self.assertTrue(galaxy.narrowband_color_enabled)
        self.assertFalse(galaxy.siril_deconvolution_enabled)
        self.assertTrue(natural_nebula.narrowband_color_enabled)
        self.assertFalse(default_settings().narrowband_color_enabled)

    def test_narrowband_checkbox_selects_a_dedicated_end_to_end_pipeline(self) -> None:
        route = inspect.getsource(run_web_legacy_150_pipeline)
        dedicated = inspect.getsource(_run_dedicated_narrowband_pipeline)

        self.assertIn("_run_dedicated_narrowband_pipeline(", route)
        self.assertIn("_prepare_early_nebula_deepsnr_source(", dedicated)
        self.assertIn("preserve_extended_signal=True", dedicated)
        self.assertIn("_run_early_linear_nebula_deepsnr(", dedicated)
        self.assertIn("apply_masked_richardson_lucy_nebula(", dedicated)
        self.assertIn("apply_pixinsight_narrowband_finish(", dedicated)
        self.assertIn("run_starnet(", dedicated)
        self.assertIn("apply_starnet_guided_narrowband_polish(", dedicated)
        self.assertIn("starless_only_requested", dedicated)
        self.assertIn("complete StarNet starless output", dedicated)
        self.assertIn("_apply_mild_nebula_star_core_reduction(", dedicated)

    def test_stacked_rgb_narrowband_script_uses_siril_split_pixelmath_and_rgbcomp(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            script = create_stacked_rgb_narrowband_script(
                root / "stacked.fit",
                root / "narrowband.fit",
                root,
            )
            commands = script.read_text(encoding="utf-8")

        self.assertIn("split deepsky_nb_ha deepsky_nb_green deepsky_nb_blue", commands)
        self.assertIn('$deepsky_nb_green$ * 0.65 + $deepsky_nb_blue$ * 0.35', commands)
        self.assertIn('$deepsky_nb_ha$ * 0.18 + $deepsky_nb_oiii$ * 0.82', commands)
        self.assertIn("rgbcomp deepsky_nb_ha deepsky_nb_hoo_green deepsky_nb_oiii", commands)

    def test_process_page_applies_narrowband_color_automatically(self) -> None:
        html = process_page()
        self.assertIn('id="starlessButton"', html)
        self.assertIn('data.append("star_setting", starSetting)', html)
        self.assertNotIn('id="narrowbandColor"', html)
        self.assertNotIn('id="narrowbandColorOption"', html)
        self.assertNotIn('name="narrowband_color" type="checkbox"', html)
        self.assertIn(
            'data.append("narrowband_color", selectedObjectType === "Nebula" || selectedObjectType === "Galaxy" ? "true" : "false");',
            html,
        )
        self.assertNotIn('id="galaxyDeconvolutionOption"', html)
        self.assertNotIn('id="sirilDeconvolution"', html)
        self.assertIn('data.append("siril_deconvolution", "false");', html)
    def test_process_page_exposes_preprocessing_crop_editor(self) -> None:
        html = process_page()
        self.assertIn('id="cropButton"', html)
        self.assertIn('id="cropModal"', html)
        self.assertIn('id="cropSelection"', html)
        self.assertIn('data.append("crop_x"', html)
        self.assertIn('data.append("crop_y"', html)
        self.assertIn('data.append("crop_width"', html)
        self.assertIn('data.append("crop_height"', html)
        self.assertIn('await renderBeforePreview()', html)
        self.assertIn("async function ensurePreviewObjectUrl()", html)
        self.assertIn("const response = await fetchAuthed(previewImageUrl);", html)
        self.assertIn("previewObjectUrl = URL.createObjectURL(await response.blob());", html)
        self.assertIn("cropImage.src = displayUrl;", html)
        self.assertNotIn("cropImage.src = previewImageUrl;", html)

    def test_process_page_hides_pixel_restoration_button(self) -> None:
        html = process_page()
        self.assertNotIn('data-restore-kind="pixel"', html)
        self.assertNotIn("button[data-restore-kind='pixel']", html)
        self.assertNotIn(">Pixel Restoration</button>", html)
if __name__ == "__main__":
    unittest.main()
