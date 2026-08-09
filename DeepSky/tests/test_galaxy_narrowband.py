from __future__ import annotations

import numpy as np

from app.galaxy_narrowband import apply_galaxy_narrowband_finish


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


def test_galaxy_narrowband_preserves_luminance_and_neutral_background() -> None:
    base, reference, signal = _synthetic_galaxy()
    output = apply_galaxy_narrowband_finish(base, reference).astype(np.float32) / 65535.0
    source = base.astype(np.float32) / 65535.0

    assert np.percentile(np.abs(_luminance(output) - _luminance(source)), 99.0) < 2.5e-4
    sky = signal < 0.002
    assert np.percentile(np.max(output[sky], axis=1) - np.min(output[sky], axis=1), 99.0) < 0.003


def test_galaxy_narrowband_creates_continuous_warm_cool_signal_separation() -> None:
    base, reference, signal = _synthetic_galaxy()
    output = apply_galaxy_narrowband_finish(base, reference).astype(np.float32) / 65535.0
    width = signal.shape[1]
    columns = np.indices(signal.shape)[1]
    warm = (signal > 0.08) & (columns > width * 0.58)
    cool = (signal > 0.08) & (columns < width * 0.42)

    assert float(np.mean(output[..., 0][warm] - output[..., 2][warm])) > 0.025
    assert float(np.mean(output[..., 2][cool] - output[..., 0][cool])) > 0.012
    assert np.isfinite(output).all()
