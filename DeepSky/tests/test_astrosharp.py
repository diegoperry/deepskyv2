from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.astrosharp import (
    blend_astrosharp_structure,
    discover_astrosharp_runtime,
)


def _synthetic_nebula() -> tuple[np.ndarray, np.ndarray]:
    height, width = 192, 224
    y, x = np.mgrid[:height, :width].astype(np.float32)
    base = np.full((height, width, 3), 0.025, dtype=np.float32)
    ridge = np.exp(-((y - (72.0 + x * 0.18)) ** 2) / (2.0 * 9.0**2))
    texture = 0.5 + 0.5 * np.sin(x * 0.22)
    gas = ridge * (0.35 + texture * 0.18)
    base[..., 0] += gas * 0.78
    base[..., 1] += gas * 0.34
    base[..., 2] += gas * 0.16
    for center_y, center_x, color in (
        (42, 48, (0.65, 0.78, 1.0)),
        (108, 154, (1.0, 0.72, 0.48)),
    ):
        radius2 = (y - center_y) ** 2 + (x - center_x) ** 2
        star = np.exp(-radius2 / (2.0 * 1.55**2))
        for channel in range(3):
            base[..., channel] += star * color[channel] * 0.75
    base = np.clip(base, 0.0, 1.0)

    lum = (
        base[..., 0] * 0.2126
        + base[..., 1] * 0.7152
        + base[..., 2] * 0.0722
    )
    sharpened_lum = np.clip(
        lum + (lum - cv2.GaussianBlur(lum, (0, 0), 2.2)) * 0.9,
        0.0,
        1.0,
    )
    donor = np.clip(
        base * (sharpened_lum / np.maximum(lum, 1e-6))[..., None],
        0.0,
        1.0,
    )
    return base, donor


def test_blend_astrosharp_structure_accepts_nebula_detail_not_quiet_sky() -> None:
    base, donor = _synthetic_nebula()
    result = blend_astrosharp_structure(base, donor, maximum_mix=0.58)
    base_lum = np.mean(base, axis=2)
    result_lum = np.mean(result, axis=2)
    sky = base_lum < 0.035
    gas = base_lum > 0.10

    assert float(np.mean(np.abs(result_lum[sky] - base_lum[sky]))) < 2e-5
    assert float(np.mean(np.abs(result_lum[gas] - base_lum[gas]))) > 2e-4


def test_blend_astrosharp_structure_protects_stars_and_preserves_color_order() -> None:
    base, donor = _synthetic_nebula()
    result = blend_astrosharp_structure(base, donor, maximum_mix=0.70)

    for y, x in ((42, 48), (108, 154)):
        assert float(np.max(np.abs(result[y, x] - base[y, x]))) < 0.012
        assert tuple(np.argsort(result[y, x])) == tuple(np.argsort(base[y, x]))


def test_blend_astrosharp_structure_rejects_shape_mismatch() -> None:
    base, donor = _synthetic_nebula()
    with pytest.raises(ValueError):
        blend_astrosharp_structure(base, donor[:-1])


def test_discover_astrosharp_runtime_requires_complete_external_tool(
    tmp_path: Path,
) -> None:
    assert discover_astrosharp_runtime(tmp_path) is None
