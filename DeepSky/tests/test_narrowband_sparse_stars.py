from __future__ import annotations

import numpy as np

from app.narrowband_sparse_stars import keep_largest_narrowband_stars


def _synthetic_sparse_field() -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    height, width = 300, 360
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    starless = np.full((height, width, 3), 0.032, dtype=np.float32)
    gas = np.exp(-(((xx - 180.0) / 76.0) ** 2 + ((yy - 155.0) / 48.0) ** 2))
    starless += gas[..., None] * np.asarray([0.15, 0.060, 0.020], dtype=np.float32)
    source = starless.copy()
    centers: list[tuple[int, int]] = []
    colors = [
        np.asarray([1.00, 0.78, 0.55], dtype=np.float32),
        np.asarray([0.58, 0.76, 1.00], dtype=np.float32),
    ]
    for index in range(20):
        x = 28 + (index % 5) * 74
        y = 28 + (index // 5) * 72
        centers.append((y, x))
        sigma = 0.85 + index * 0.055
        amplitude = 0.20 + index * 0.032
        profile = np.exp(
            -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2)
        ).astype(np.float32)
        source += profile[..., None] * colors[index % 2] * amplitude
    return np.clip(source, 0.0, 1.0), np.clip(starless, 0.0, 1.0), centers


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def test_sparse_finish_keeps_only_largest_ten_percent() -> None:
    source, starless, centers = _synthetic_sparse_field()
    logs: list[str] = []
    output = keep_largest_narrowband_stars(
        source,
        source,
        starless,
        np.ones(source.shape[:2], dtype=np.float32),
        keep_fraction=0.10,
        log=logs.append,
    )
    source_lum = _luminance(source)
    starless_lum = _luminance(starless)
    output_lum = _luminance(output)
    before_contrast = np.asarray(
        [source_lum[y, x] - starless_lum[y, x] for y, x in centers]
    )
    after_contrast = np.asarray(
        [output_lum[y, x] - starless_lum[y, x] for y, x in centers]
    )

    largest = np.argsort(before_contrast)[-2:]
    smaller = np.argsort(before_contrast)[:-2]
    assert np.all(after_contrast[largest] > before_contrast[largest] * 0.82)
    assert float(np.median(after_contrast[smaller] / before_contrast[smaller])) < 0.32
    assert int(np.count_nonzero(after_contrast > 0.10)) <= 3
    removed_rgb_delta = np.asarray(
        [
            output[y, x] - starless[y, x]
            for index, (y, x) in enumerate(centers)
            if index in set(smaller.tolist())
        ]
    )
    assert float(np.percentile(np.abs(removed_rgb_delta), 99.0)) < 0.015
    removed_chroma_delta = removed_rgb_delta - np.mean(
        removed_rgb_delta,
        axis=1,
        keepdims=True,
    )
    assert float(np.percentile(np.abs(removed_chroma_delta), 99.0)) < 0.004
    assert any("kept=2" in message and "removed=18" in message for message in logs)


def test_sparse_finish_preserves_large_star_colors_and_nebula() -> None:
    source, starless, centers = _synthetic_sparse_field()
    output = keep_largest_narrowband_stars(
        source,
        source,
        starless,
        np.ones(source.shape[:2], dtype=np.float32),
        keep_fraction=0.10,
    )
    for y, x in centers[-2:]:
        source_color = source[y, x] / np.max(source[y, x])
        output_color = output[y, x] / np.max(output[y, x])
        assert float(np.max(np.abs(source_color - output_color))) < 0.01

    source_gas = np.mean(source[140:170, 160:200], axis=(0, 1))
    output_gas = np.mean(output[140:170, 160:200], axis=(0, 1))
    assert output_gas[0] > output_gas[2] * 1.8
    assert float(np.max(np.abs(output_gas - source_gas))) < 0.006


def test_sparse_finish_never_imports_starless_blob_pixels() -> None:
    source, starless, _ = _synthetic_sparse_field()
    yy, xx = np.mgrid[: source.shape[0], : source.shape[1]].astype(np.float32)
    blob = np.exp(-(((xx - 330.0) / 13.0) ** 2 + ((yy - 270.0) / 11.0) ** 2))
    starless[..., 0] = np.clip(starless[..., 0] + blob * 0.65, 0.0, 1.0)
    starless[..., 2] = np.clip(starless[..., 2] - blob * 0.22, 0.0, 1.0)
    output = keep_largest_narrowband_stars(
        source,
        source,
        starless,
        np.ones(source.shape[:2], dtype=np.float32),
        keep_fraction=0.10,
    )
    blob_region = blob > 0.70
    assert float(np.percentile(np.abs(output[blob_region] - source[blob_region]), 99.0)) < 0.01
