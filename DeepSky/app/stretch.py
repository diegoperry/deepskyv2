from __future__ import annotations

import numpy as np


def _as_float01(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if np.issubdtype(np.asarray(image).dtype, np.integer):
        info = np.iinfo(np.asarray(image).dtype)
        if info.max > 0:
            arr = arr / float(info.max)
    return arr


def stretch_channel(
    channel: np.ndarray,
    *,
    background_percentile: float = 0.5,
    background_subtract: float = 1.0,
    white_percentile: float = 99.8,
    black_percentile: float = 0.1,
    high_percentile: float = 99.9,
    stretch_factor: float = 12.0,
) -> np.ndarray:
    data = _as_float01(channel)
    background = np.percentile(data, background_percentile)
    data = data - background * background_subtract
    data = np.clip(data, 0.0, None)

    white = np.percentile(data, white_percentile)
    if white <= 0:
        white = float(np.max(data)) if np.max(data) > 0 else 1.0
    data = np.clip(data / white, 0.0, 1.0)

    black = np.percentile(data, black_percentile)
    high = np.percentile(data, high_percentile)
    if high <= black:
        high = 1.0
    data = np.clip((data - black) / (high - black), 0.0, 1.0)

    data = np.arcsinh(stretch_factor * data) / np.arcsinh(stretch_factor)
    return np.clip(data, 0.0, 1.0)


def astrophotography_stretch(image: np.ndarray, strength: str = "normal") -> np.ndarray:
    arr = np.asarray(image)
    if strength == "gentle":
        stretch_kwargs = {
            "background_percentile": 0.1,
            "background_subtract": 0.30,
            "white_percentile": 99.92,
            "black_percentile": 0.02,
            "high_percentile": 99.95,
            "stretch_factor": 4.5,
        }
        max_scale = 3.8
    else:
        stretch_kwargs = {}
        max_scale = 8.0

    if arr.ndim == 2:
        stretched = stretch_channel(arr, **stretch_kwargs)
    elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3]
        luminance = np.mean(_as_float01(rgb), axis=2)
        stretched_l = stretch_channel(luminance, **stretch_kwargs)
        old_l = np.maximum(luminance, 1e-6)
        scale = np.clip(stretched_l / old_l, 0.0, max_scale)
        stretched = np.clip(_as_float01(rgb) * scale[..., None], 0.0, 1.0)
    else:
        squeezed = np.squeeze(arr)
        if squeezed.ndim != 2:
            raise ValueError(f"Unsupported image shape for stretch: {arr.shape}")
        stretched = stretch_channel(squeezed)
    return (np.clip(stretched, 0.0, 1.0) * 65535.0).astype(np.uint16)
