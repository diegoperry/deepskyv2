from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .image_io import load_tiff, save_tiff


def subtract_images(source_path: Path, subtract_path: Path, output_path: Path) -> None:
    source = load_tiff(source_path).astype(np.int32)
    subtract = load_tiff(subtract_path).astype(np.int32)
    stars = np.clip(source - subtract, 0, 65535).astype(np.uint16)
    save_tiff(output_path, stars)


def add_images(base_path: Path, add_path: Path, output_path: Path) -> None:
    base = load_tiff(base_path).astype(np.int32)
    add = load_tiff(add_path).astype(np.int32)
    final = np.clip(base + add, 0, 65535).astype(np.uint16)
    save_tiff(output_path, final)


def add_weighted_star_layer(
    base_path: Path,
    stars_path: Path,
    output_path: Path,
    floor_weight: float = 0.05,
    low_percentile: float = 96.5,
    high_percentile: float = 99.85,
    curve_power: float = 1.0,
) -> tuple[float, float]:
    base = load_tiff(base_path).astype(np.float32)
    stars = load_tiff(stars_path).astype(np.float32)
    if stars.ndim == 3:
        star_lum = stars[..., :3].max(axis=2)
    else:
        star_lum = stars

    low = float(np.percentile(star_lum, low_percentile))
    high = float(np.percentile(star_lum, high_percentile))
    strength = np.clip((star_lum - low) / max(1.0, high - low), 0.0, 1.0)
    smooth = strength * strength * (3.0 - 2.0 * strength)
    weight = float(np.clip(floor_weight, 0.0, 1.0)) + (1.0 - float(np.clip(floor_weight, 0.0, 1.0))) * (
        smooth ** max(0.1, float(curve_power))
    )
    if stars.ndim == 3:
        weight = weight[..., None]

    final = np.clip(base + stars * weight, 0, 65535).astype(np.uint16)
    save_tiff(output_path, final)
    return low, high


def add_bright_star_fraction(
    base_path: Path,
    stars_path: Path,
    output_path: Path,
    keep_fraction: float = 0.30,
) -> float:
    base = load_tiff(base_path).astype(np.int32)
    stars = load_tiff(stars_path).astype(np.float32)
    if stars.ndim == 3:
        star_lum = stars[..., :3].max(axis=2)
    else:
        star_lum = stars

    candidate_floor = max(64.0, float(np.percentile(star_lum, 96.0)))
    candidates = star_lum[star_lum > candidate_floor]
    if candidates.size == 0:
        save_tiff(output_path, np.clip(base, 0, 65535).astype(np.uint16))
        return 0.0

    keep_fraction = float(np.clip(keep_fraction, 0.0, 1.0))
    threshold = float(np.percentile(candidates, (1.0 - keep_fraction) * 100.0))
    mask = (star_lum >= threshold).astype(np.float32)
    if stars.ndim == 3:
        mask = mask[..., None]

    kept_stars = stars * mask
    final = np.clip(base + kept_stars.astype(np.int32), 0, 65535).astype(np.uint16)
    final = _repair_retained_star_pinholes(final, mask)
    save_tiff(output_path, final)
    return threshold


def _repair_retained_star_pinholes(image: np.ndarray, retained_star_mask: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        return image

    support = retained_star_mask[..., 0] if retained_star_mask.ndim == 3 else retained_star_mask
    support = (support > 0).astype(np.uint8)
    if int(np.count_nonzero(support)) == 0:
        return image

    support = cv2.dilate(support, np.ones((5, 5), dtype=np.uint8), iterations=1).astype(bool)
    arr = image.astype(np.float32)
    lum = arr[..., :3].max(axis=2)
    local_peak = cv2.dilate(lum, np.ones((7, 7), dtype=np.uint8), iterations=1)
    pinhole = support & (local_peak > 9000.0) & (lum < local_peak * 0.42)
    if int(np.count_nonzero(pinhole)) == 0:
        return image

    fill = cv2.GaussianBlur(arr, (0, 0), 1.35)
    repaired = arr.copy()
    repaired[pinhole] = np.maximum(repaired[pinhole], fill[pinhole] * 1.08)
    return np.clip(repaired, 0, 65535).astype(np.uint16)
