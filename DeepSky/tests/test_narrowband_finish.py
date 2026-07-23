from __future__ import annotations

import cv2
import numpy as np

from app.narrowband_finish import (
    apply_pixinsight_narrowband_finish,
    apply_starnet_guided_narrowband_polish,
)


def _synthetic_linear_nebula() -> np.ndarray:
    height, width = 256, 256
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    image = np.full((height, width, 3), 0.018, dtype=np.float32)

    # Deliberate two-pixel color pattern resembling amplified CFA/chroma noise.
    checker = (((xx.astype(np.int32) + yy.astype(np.int32)) & 1) * 2 - 1).astype(np.float32)
    image[..., 0] += checker * 0.0030
    image[..., 2] -= checker * 0.0030

    warm = np.exp(-(((xx - 110.0) / 52.0) ** 2 + ((yy - 135.0) / 38.0) ** 2))
    cool = np.exp(-(((xx - 154.0) / 30.0) ** 2 + ((yy - 116.0) / 58.0) ** 2))
    image += warm[..., None] * np.array([0.090, 0.038, 0.014], dtype=np.float32)
    image += cool[..., None] * np.array([0.010, 0.042, 0.070], dtype=np.float32)

    for x, y, color in [
        (54, 64, (0.42, 0.35, 0.25)),
        (196, 78, (0.22, 0.34, 0.48)),
        (174, 194, (0.46, 0.40, 0.32)),
    ]:
        radius = (xx - x) ** 2 + (yy - y) ** 2
        profile = np.exp(-radius / (2.0 * 1.35**2))
        image += profile[..., None] * np.asarray(color, dtype=np.float32)
    return np.clip(image, 0.0, 1.0)


def test_pixinsight_narrowband_finish_suppresses_checker_and_keeps_background_neutral() -> None:
    source = _synthetic_linear_nebula()
    output = apply_pixinsight_narrowband_finish(source).astype(np.float32) / 65535.0

    yy, xx = np.mgrid[:256, :256]
    background = ((xx < 42) | (xx > 214)) & ((yy < 42) | (yy > 214))
    output_chroma = output - np.mean(output, axis=2, keepdims=True)
    parity = []
    for y_parity in range(2):
        for x_parity in range(2):
            mask = background & ((yy & 1) == y_parity) & ((xx & 1) == x_parity)
            parity.append(np.mean(output_chroma[mask], axis=0))
    parity = np.asarray(parity)

    assert output.dtype == np.float32
    assert float(np.max(parity, axis=0).max() - np.min(parity, axis=0).min()) < 0.0015
    assert float(np.percentile(np.max(output[background], axis=1) - np.min(output[background], axis=1), 95.0)) < 0.018
    assert float(np.median(np.mean(output[background], axis=1))) > 0.008


def test_pixinsight_narrowband_finish_preserves_stellar_color_order() -> None:
    source = _synthetic_linear_nebula()
    output = apply_pixinsight_narrowband_finish(source).astype(np.float32) / 65535.0

    warm_star = output[64, 54]
    cool_star = output[78, 196]
    assert warm_star[0] > warm_star[2]
    assert cool_star[2] > cool_star[0]
    assert float(np.max(warm_star)) < 1.0
    assert float(np.max(cool_star)) < 1.0


def test_pixinsight_narrowband_finish_keeps_real_warm_and_cool_nebula_separation() -> None:
    source = _synthetic_linear_nebula()
    output = apply_pixinsight_narrowband_finish(source).astype(np.float32) / 65535.0

    warm_region = cv2.GaussianBlur(output, (0, 0), 4.0)[142, 92]
    cool_region = cv2.GaussianBlur(output, (0, 0), 4.0)[112, 170]
    assert warm_region[0] > warm_region[2] * 1.15
    assert cool_region[2] > cool_region[0] * 1.10

def test_starnet_guided_polish_never_imports_starless_blobs() -> None:
    finished = apply_pixinsight_narrowband_finish(_synthetic_linear_nebula())
    starless = finished.astype(np.float32) / 65535.0
    yy, xx = np.mgrid[:256, :256].astype(np.float32)
    fake_blob = np.exp(-(((xx - 205.0) / 18.0) ** 2 + ((yy - 205.0) / 14.0) ** 2))
    starless[..., 0] = np.clip(starless[..., 0] + fake_blob * 0.55, 0.0, 1.0)
    starless[..., 2] = np.clip(starless[..., 2] - fake_blob * 0.30, 0.0, 1.0)

    polished = apply_starnet_guided_narrowband_polish(finished, starless)
    polished = polished.astype(np.float32) / 65535.0
    source = finished.astype(np.float32) / 65535.0

    blob_region = fake_blob > 0.60
    difference = np.abs(polished - source)
    assert float(np.percentile(difference[blob_region], 99.0)) < 0.025
    assert float(np.max(np.abs(polished[64, 54] - source[64, 54]))) < 0.01
    assert polished[64, 54, 0] > polished[64, 54, 2]
