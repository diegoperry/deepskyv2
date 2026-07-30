from __future__ import annotations

import cv2
import numpy as np

from app.narrowband_option04 import apply_option04_very_heavy_finish


def _synthetic_option04_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = 256, 256
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    rng = np.random.default_rng(42)
    background = 0.030 + rng.normal(0.0, 0.0045, (height, width, 1)).astype(np.float32)
    source = np.repeat(background, 3, axis=2)

    warm_gas = np.exp(-(((xx - 118.0) / 54.0) ** 2 + ((yy - 132.0) / 25.0) ** 2))
    cool_gas = np.exp(-(((xx - 151.0) / 24.0) ** 2 + ((yy - 127.0) / 48.0) ** 2))
    source += warm_gas[..., None] * np.asarray([0.18, 0.065, 0.020], dtype=np.float32)
    source += cool_gas[..., None] * np.asarray([0.015, 0.065, 0.125], dtype=np.float32)
    starless = source.copy()

    for x, y, color, sigma in [
        (48, 55, (0.72, 0.55, 0.36), 1.35),
        (202, 63, (0.34, 0.53, 0.82), 1.50),
        (185, 188, (0.75, 0.63, 0.44), 1.20),
        (89, 174, (0.46, 0.58, 0.78), 1.10),
    ]:
        profile = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2))
        source += profile[..., None] * np.asarray(color, dtype=np.float32)
    return np.clip(source, 0.0, 1.0), np.clip(starless, 0.0, 1.0), warm_gas


def test_option04_reduces_compact_stars_and_preserves_star_color_order() -> None:
    source, starless, _ = _synthetic_option04_scene()
    output = apply_option04_very_heavy_finish(
        source,
        source,
        starless,
        np.ones(source.shape[:2], dtype=np.float32),
    )

    assert float(np.max(output[55, 48])) < float(np.max(source[55, 48])) * 0.92
    assert float(np.max(output[63, 202])) < float(np.max(source[63, 202])) * 0.92
    assert output[55, 48, 0] > output[55, 48, 2]
    assert output[63, 202, 2] > output[63, 202, 0]


def test_option04_softens_and_darkens_only_quiet_background() -> None:
    source, starless, warm_gas = _synthetic_option04_scene()
    output = apply_option04_very_heavy_finish(
        source,
        source,
        starless,
        np.ones(source.shape[:2], dtype=np.float32),
    )
    yy, xx = np.mgrid[:256, :256]
    sky = ((xx < 38) | (xx > 218)) & ((yy < 38) | (yy > 218))
    source_sky = np.mean(source[sky], axis=1)
    output_sky = np.mean(output[sky], axis=1)

    assert float(np.std(output_sky)) < float(np.std(source_sky)) * 0.82
    assert float(np.median(output_sky)) < float(np.median(source_sky)) * 0.94
    assert float(np.median(output_sky)) > 0.012
    warm_region = warm_gas > 0.72
    assert float(np.mean(output[warm_region, 0])) > float(np.mean(output[warm_region, 2])) * 1.8
    assert float(np.mean(output[warm_region])) > float(np.mean(source[warm_region])) * 0.88


def test_option04_never_imports_a_fake_starless_blob() -> None:
    source, starless, _ = _synthetic_option04_scene()
    yy, xx = np.mgrid[:256, :256].astype(np.float32)
    fake_blob = np.exp(-(((xx - 224.0) / 12.0) ** 2 + ((yy - 222.0) / 10.0) ** 2))
    starless[..., 0] = np.clip(starless[..., 0] + fake_blob * 0.70, 0.0, 1.0)
    starless[..., 2] = np.clip(starless[..., 2] - fake_blob * 0.25, 0.0, 1.0)

    output = apply_option04_very_heavy_finish(
        source,
        source,
        starless,
        np.ones(source.shape[:2], dtype=np.float32),
    )
    blob = fake_blob > 0.70
    assert float(np.max(output[blob])) < 0.12
    assert float(np.percentile(np.abs(output[blob] - source[blob]), 99.0)) < 0.025


def test_option04_restores_nebula_microcontrast_without_sharpening_sky() -> None:
    source, starless, warm_gas = _synthetic_option04_scene()
    yy, xx = np.mgrid[:256, :256].astype(np.float32)
    filament = np.sin(xx * 0.58 + yy * 0.13) * warm_gas * 0.012
    source = np.clip(source + filament[..., None], 0.0, 1.0)
    starless = np.clip(starless + filament[..., None], 0.0, 1.0)

    output = apply_option04_very_heavy_finish(
        source,
        source,
        starless,
        np.ones(source.shape[:2], dtype=np.float32),
    )
    source_lum = np.mean(source, axis=2).astype(np.float32)
    output_lum = np.mean(output, axis=2).astype(np.float32)
    source_band = (
        cv2.GaussianBlur(source_lum, (0, 0), 0.65)
        - cv2.GaussianBlur(source_lum, (0, 0), 2.0)
    )
    output_band = (
        cv2.GaussianBlur(output_lum, (0, 0), 0.65)
        - cv2.GaussianBlur(output_lum, (0, 0), 2.0)
    )
    nebula = warm_gas > 0.60
    sky = warm_gas < 0.02
    sky[:24, :] = False
    sky[-24:, :] = False
    sky[:, :24] = False
    sky[:, -24:] = False

    assert float(np.std(output_band[nebula])) > float(np.std(source_band[nebula])) * 0.72
    source_sky_noise = float(np.std(cv2.Laplacian(source_lum, cv2.CV_32F)[sky]))
    output_sky_noise = float(np.std(cv2.Laplacian(output_lum, cv2.CV_32F)[sky]))
    assert output_sky_noise < source_sky_noise * 0.90
