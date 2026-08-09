from __future__ import annotations

import cv2
import numpy as np

from app.galaxy_narrowband import apply_galaxy_narrowband_finish, crop_galaxy_mosaic_footprint


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _synthetic_galaxy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = 256, 320
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    x = (xx - width * 0.50) / 82.0
    y = (yy - height * 0.50) / 34.0
    radius = np.sqrt(x * x + y * y)
    disk = np.exp(-(radius**1.45) * 1.55)
    arms = np.exp(-((radius - 0.72 - np.sin(np.arctan2(y, x) * 2.0) * 0.16) ** 2) / 0.028)
    signal = np.clip(disk * 0.52 + arms * 0.22, 0.0, 0.72)
    lum = np.full((height, width), 0.018, dtype=np.float32) + signal
    base = np.repeat(lum[..., None], 3, axis=2)

    reference = base.copy()
    warm_side = np.clip((xx - width * 0.42) / (width * 0.28), 0.0, 1.0)
    cool_side = 1.0 - warm_side
    reference[..., 0] += signal * warm_side * 0.16
    reference[..., 2] += signal * cool_side * 0.14
    reference[..., 1] += signal * cool_side * 0.035
    return (
        np.clip(base * 65535.0, 0, 65535).astype(np.uint16),
        np.clip(reference * 65535.0, 0, 65535).astype(np.uint16),
        signal,
    )


def test_galaxy_narrowband_rolls_highlights_and_keeps_neutral_background() -> None:
    base, reference, signal = _synthetic_galaxy()
    output = apply_galaxy_narrowband_finish(base, reference).astype(np.float32) / 65535.0
    source = base.astype(np.float32) / 65535.0

    assert float(np.max(_luminance(output))) < float(np.max(_luminance(source))) + 0.01
    sky = signal < 0.002
    assert np.percentile(np.max(output[sky], axis=1) - np.min(output[sky], axis=1), 99.0) < 0.003


def test_galaxy_narrowband_creates_continuous_warm_cool_signal_separation() -> None:
    base, reference, signal = _synthetic_galaxy()
    output = apply_galaxy_narrowband_finish(base, reference).astype(np.float32) / 65535.0
    width = signal.shape[1]
    columns = np.indices(signal.shape)[1]
    warm = (signal > 0.08) & (columns > width * 0.58)
    cool = (signal > 0.08) & (columns < width * 0.42)

    assert float(np.mean(output[..., 0][warm] - output[..., 2][warm])) > 0.0015
    assert float(np.mean(output[..., 2][cool] - output[..., 0][cool])) > 0.0015
    assert np.isfinite(output).all()


def test_post_denoise_deconvolution_donor_increases_galaxy_structure() -> None:
    base, reference, signal = _synthetic_galaxy()
    yy, xx = np.mgrid[: signal.shape[0], : signal.shape[1]]
    detail = base.astype(np.float32) / 65535.0
    ripple = np.sin(xx * 0.45) * np.cos(yy * 0.31) * signal * 0.035
    detail = np.clip(detail + ripple[..., None], 0.0, 1.0)
    detail = np.round(detail * 65535.0).astype(np.uint16)

    baseline = apply_galaxy_narrowband_finish(base, reference).astype(np.float32) / 65535.0
    restored = apply_galaxy_narrowband_finish(
        base, reference, detail_reference=detail
    ).astype(np.float32) / 65535.0
    mask = signal > 0.08
    baseline_lum = _luminance(baseline)
    restored_lum = _luminance(restored)
    baseline_detail = baseline_lum - cv2.GaussianBlur(baseline_lum, (0, 0), 1.2)
    restored_detail = restored_lum - cv2.GaussianBlur(restored_lum, (0, 0), 1.2)

    assert float(np.std(restored_detail[mask])) > float(np.std(baseline_detail[mask])) * 1.20
    assert float(np.max(restored_lum)) < float(np.max(baseline_lum)) + 0.02

def test_mosaic_footprint_is_cropped_to_clean_rectangle() -> None:
    image = np.full((120, 140, 3), 12000, dtype=np.uint16)
    coverage = image.copy()
    coverage[:20, :35] = 0
    coverage[:20, 105:] = 0
    coverage[-20:, :25] = 0
    coverage[-20:, 115:] = 0

    cropped = crop_galaxy_mosaic_footprint(image, coverage)

    assert cropped.shape[0] < image.shape[0] or cropped.shape[1] < image.shape[1]
    assert np.all(cropped > 0)
