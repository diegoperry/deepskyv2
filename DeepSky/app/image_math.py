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
    bright_mask = star_lum >= threshold
    mask = _compact_retained_star_mask(star_lum, bright_mask, threshold)
    kept_stars = _build_halo_suppressed_star_layer(stars, star_lum, mask, bright_mask, threshold)
    final = np.clip(base + kept_stars.astype(np.int32), 0, 65535).astype(np.uint16)
    final = _repair_retained_star_pinholes(final, mask.astype(np.float32))
    save_tiff(output_path, final)
    return threshold


def _build_halo_suppressed_star_layer(
    stars: np.ndarray,
    star_lum: np.ndarray,
    retained_mask: np.ndarray,
    bright_mask: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Keep retained star cores while strongly reducing colored StarNet halos."""
    if int(np.count_nonzero(retained_mask)) == 0:
        return np.zeros_like(stars, dtype=np.float32)

    retained = retained_mask.astype(np.float32)
    core = bright_mask.astype(np.float32)
    halo = np.clip(retained - core, 0.0, 1.0)

    local_peak = cv2.dilate(star_lum.astype(np.float32), np.ones((9, 9), dtype=np.uint8), iterations=1)
    local_ratio = np.clip(star_lum / np.maximum(local_peak, 1.0), 0.0, 1.0)
    compact_core = np.clip((local_ratio - 0.42) / 0.40, 0.0, 1.0)
    compact_core = compact_core * compact_core * (3.0 - 2.0 * compact_core)

    peak_window = max(1.0, float(np.percentile(star_lum[retained_mask], 99.85)) - threshold)
    brightness_core = np.clip((star_lum - threshold) / peak_window, 0.0, 1.0)
    brightness_core = brightness_core * brightness_core * (3.0 - 2.0 * brightness_core)

    core_weight = np.clip(np.maximum(compact_core, brightness_core) * core, 0.0, 1.0)
    # Do not re-add the diffuse StarNet difference layer: it contains the colored
    # rings that appear around bright stars.  A tiny feather avoids hard cut-outs.
    halo_weight = np.clip(halo * 0.010 * compact_core, 0.0, 0.012)
    weight = np.clip(core_weight + halo_weight, 0.0, 1.0)

    if stars.ndim != 3:
        return stars.astype(np.float32) * weight

    star_rgb = stars.astype(np.float32)
    lum = star_lum.astype(np.float32)
    neutral = np.repeat(lum[..., None], star_rgb.shape[2], axis=2)
    halo_mix = halo[..., None] * (1.0 - core_weight[..., None]) * 0.985
    star_rgb = np.clip(star_rgb * (1.0 - halo_mix) + neutral * halo_mix, 0.0, 65535.0)
    return star_rgb * weight[..., None]


def _compact_retained_star_mask(star_lum: np.ndarray, bright_mask: np.ndarray, threshold: float) -> np.ndarray:
    seeds = bright_mask.astype(np.uint8)
    if int(np.count_nonzero(seeds)) == 0:
        return bright_mask

    count, labels, stats, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)
    if count <= 1:
        return bright_mask

    peaks = np.zeros(count, dtype=np.float32)
    np.maximum.at(peaks, labels.ravel(), star_lum.ravel().astype(np.float32))
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    valid = (np.arange(count) > 0) & (peaks >= threshold) & (areas <= max(900.0, float(np.percentile(areas[1:], 99.8)) * 1.8))
    valid[0] = False
    compact = valid[labels]
    compact = cv2.dilate(compact.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
    return compact


def _large_retained_star_mask(star_lum: np.ndarray, candidate_floor: float, threshold: float) -> np.ndarray:
    support_floor = max(64.0, threshold * 0.45)
    support = (star_lum > support_floor).astype(np.uint8)
    if int(np.count_nonzero(support)) == 0:
        return np.zeros_like(star_lum, dtype=bool)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
    if count <= 1:
        return np.zeros_like(star_lum, dtype=bool)

    area = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    peaks = np.zeros(count, dtype=np.float32)
    np.maximum.at(peaks, labels.ravel(), star_lum.ravel().astype(np.float32))

    valid = (np.arange(count) > 0) & (peaks >= max(candidate_floor * 1.5, threshold * 0.35))
    if not bool(np.any(valid)):
        return np.zeros_like(star_lum, dtype=bool)

    min_area = max(35.0, float(np.percentile(area[valid], 98.5)))
    keep_labels = valid & (area >= min_area)
    keep_labels[0] = False
    if not bool(np.any(keep_labels)):
        return np.zeros_like(star_lum, dtype=bool)

    retained = keep_labels[labels]
    retained = cv2.dilate(retained.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1)
    return retained.astype(bool)


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
