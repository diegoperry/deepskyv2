from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    ).astype(np.float32)


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def apply_creative_color_finish(image: np.ndarray) -> np.ndarray:
    """Apply one continuous, signal-aware artistic grade to an RGB image."""
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] < 3:
        raise ValueError("Creative Color Finish requires an RGB image.")

    if np.issubdtype(source.dtype, np.integer):
        scale = float(np.iinfo(source.dtype).max)
        rgb = source[..., :3].astype(np.float32) / max(scale, 1.0)
    else:
        rgb = source[..., :3].astype(np.float32)
        if rgb.size and float(np.nanmax(rgb)) > 1.0:
            rgb /= 255.0
    rgb = np.clip(np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    lum = _luminance(rgb)
    height, width = lum.shape
    broad_sigma = float(np.clip(min(height, width) / 260.0, 3.0, 12.0))
    broad_lum = cv2.GaussianBlur(lum, (0, 0), broad_sigma)
    sky_level = float(np.percentile(broad_lum, 32.0))
    signal_high = float(np.percentile(broad_lum, 98.5))
    signal_span = max(signal_high - sky_level, 0.025)

    luminance_signal = _smoothstep(
        (broad_lum - sky_level - signal_span * 0.035) / (signal_span * 0.72)
    )
    chroma_extent = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    smooth_chroma = cv2.GaussianBlur(chroma_extent.astype(np.float32), (0, 0), broad_sigma * 0.55)
    quiet = broad_lum <= np.percentile(broad_lum, 55.0)
    chroma_floor = float(np.percentile(smooth_chroma[quiet], 65.0)) if np.any(quiet) else 0.0
    chroma_high = float(np.percentile(smooth_chroma, 98.0))
    chroma_signal = _smoothstep(
        (smooth_chroma - chroma_floor) / max(chroma_high - chroma_floor, 0.018)
    )
    chroma_gate = _smoothstep((broad_lum - sky_level) / max(signal_span * 0.30, 0.012))
    signal = np.clip(np.maximum(luminance_signal, chroma_signal * chroma_gate * 0.88), 0.0, 1.0)
    signal = cv2.GaussianBlur(signal.astype(np.float32), (0, 0), 2.2)

    # Compact bright structures are treated as stars and blended back from the
    # original. This keeps white/blue/gold star colors from becoming neon.
    fine_lum = np.maximum(lum - cv2.GaussianBlur(lum, (0, 0), 1.35), 0.0)
    fine_high = max(float(np.percentile(fine_lum, 99.20)), 0.006)
    bright_floor = float(np.percentile(lum, 93.0))
    star_core = _smoothstep(fine_lum / fine_high) * _smoothstep(
        (lum - bright_floor) / max(float(np.percentile(lum, 99.5)) - bright_floor, 0.030)
    )
    star_protect = cv2.GaussianBlur(
        cv2.dilate(star_core.astype(np.float32), np.ones((3, 3), dtype=np.uint8)),
        (0, 0),
        1.15,
    )
    star_protect = np.clip(star_protect, 0.0, 1.0)

    background = (1.0 - signal) * (1.0 - star_protect * 0.96)
    source_chroma = rgb - lum[..., None]
    # Low-signal background becomes darker and more neutral, but its luminance
    # texture remains continuous; there is no hard recoloring mask.
    background_lum = lum * (1.0 - background * 0.16)
    background_chroma = source_chroma * (1.0 - background[..., None] * 0.72)
    graded = background_lum[..., None] + background_chroma

    graded_lum = _luminance(graded)
    local_lum = cv2.GaussianBlur(graded_lum, (0, 0), broad_sigma * 0.75)
    local_detail = graded_lum - local_lum
    object_lum = np.clip(
        graded_lum
        + signal * local_detail * 0.30
        + signal * np.maximum(graded_lum - sky_level, 0.0) * (1.0 - graded_lum) * 0.10,
        0.0,
        1.0,
    )

    object_chroma = graded - graded_lum[..., None]
    object_chroma *= (1.0 + signal[..., None] * 0.52)
    # Increase only color differences already present in the data. The signed
    # opponent term creates warm/cool separation without assigning a hue.
    red_blue_axis = rgb[..., 0] - rgb[..., 2]
    separation = np.tanh(red_blue_axis / np.maximum(chroma_extent * 1.8 + 0.025, 0.025))
    separation *= chroma_extent * signal * 0.18
    opponent = np.stack((separation, separation * 0.22, -separation), axis=2)

    # A continuous structural split gives luminous cores a restrained cyan/
    # blue bias and lower-signal envelopes a gold/copper bias. Both sides are
    # proportional to measured signal and existing chroma, so empty sky is
    # never assigned a color and transitions cannot form hard mask edges.
    core_confidence = signal * _smoothstep(
        (broad_lum - sky_level - signal_span * 0.22) / (signal_span * 0.48)
    )
    envelope_confidence = signal * (1.0 - core_confidence) * 0.72
    opponent *= 1.0 - core_confidence[..., None] * 0.68
    split_strength = (0.060 + chroma_extent * 0.20) * (1.0 - star_protect * 0.96)
    cool_amount = core_confidence * split_strength
    warm_amount = envelope_confidence * split_strength
    split_tone = np.stack(
        (warm_amount * 0.90 - cool_amount * 1.20,
         warm_amount * 0.45 + cool_amount * 0.65,
         cool_amount * 1.40 - warm_amount * 0.70),
        axis=2,
    )
    colored = np.clip(object_lum[..., None] + object_chroma + opponent + split_tone, 0.0, 1.0)
    colored_lum = _luminance(colored)
    colored = np.clip(colored * (object_lum / np.maximum(colored_lum, 1e-5))[..., None], 0.0, 1.0)
    colored = np.clip(
        object_lum[..., None] + (colored - object_lum[..., None]) * (1.0 + signal[..., None] * 0.85),
        0.0,
        1.0,
    )
    colored_lum = _luminance(colored)
    colored = np.clip(colored * (object_lum / np.maximum(colored_lum, 1e-5))[..., None], 0.0, 1.0)

    result = colored * (1.0 - star_protect[..., None] * 0.92) + rgb * star_protect[..., None] * 0.92
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def create_creative_color_finish(source_path: Path, output_path: Path) -> Path:
    """Create the optional creative PNG without changing the source image."""
    with Image.open(source_path) as opened:
        source = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    finished = apply_creative_color_finish(source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(finished * 255.0).astype(np.uint8), mode="RGB").save(output_path, format="PNG")
    return output_path
