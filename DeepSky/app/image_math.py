from __future__ import annotations

from pathlib import Path

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
    save_tiff(output_path, final)
    return threshold
