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
