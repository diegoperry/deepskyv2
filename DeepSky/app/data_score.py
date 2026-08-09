from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from astropy.io import fits

from .image_io import load_image


def _robust_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32)
    median = float(np.median(values))
    return max(1e-7, 1.4826 * float(np.median(np.abs(values - median))))


def _exposure_seconds(path: Path) -> float | None:
    if path.suffix.lower() not in {".fit", ".fits", ".fts"}:
        return None
    try:
        with fits.open(path, memmap=False) as hdul:
            header = next((h.header for h in hdul if h.data is not None), hdul[0].header)
        exposure = next((header.get(key) for key in ("EXPTIME", "EXPOSURE") if header.get(key) is not None), None)
        if exposure is None:
            return None
        count = next((header.get(key) for key in ("STACKCNT", "NCOMBINE", "NFRAMES") if header.get(key) is not None), 1)
        total = float(exposure) * max(1.0, float(count))
        return total if np.isfinite(total) and total > 0 else None
    except (OSError, TypeError, ValueError):
        return None


def _format_duration(seconds: float) -> str:
    minutes = max(1, int(round(seconds / 60.0)))
    if minutes < 60:
        return f"{minutes} more minute{'s' if minutes != 1 else ''}"
    hours, remainder = divmod(minutes, 60)
    if remainder < 5:
        return f"about {hours} more hour{'s' if hours != 1 else ''}"
    return f"about {hours} hr {remainder} min more"


def evaluate_data_score(path: Path) -> dict[str, Any]:
    """Measure capture quality without stretching or modifying the source data."""
    source = np.asarray(load_image(path), dtype=np.float32)
    if source.ndim == 3:
        if source.shape[-1] >= 3:
            luminance = 0.2126 * source[..., 0] + 0.7152 * source[..., 1] + 0.0722 * source[..., 2]
        else:
            luminance = np.mean(source, axis=-1)
    else:
        luminance = np.squeeze(source)
    luminance = np.nan_to_num(luminance, nan=0.0, posinf=0.0, neginf=0.0)
    if luminance.ndim != 2 or min(luminance.shape) < 32:
        raise ValueError("Image is too small to score reliably.")

    step = max(1, int(np.ceil(max(luminance.shape) / 1600)))
    sample = luminance[::step, ::step].astype(np.float32, copy=False)
    lo, hi = np.percentile(sample, (0.1, 99.9))
    dynamic = float(hi - lo)
    if not np.isfinite(dynamic) or dynamic <= 1e-10:
        raise ValueError("Image has too little measurable variation to score.")
    normalized = np.clip((sample - lo) / dynamic, 0.0, 1.0)

    background_cut = float(np.percentile(normalized, 55.0))
    background_mask = normalized <= background_cut
    blur = cv2.GaussianBlur(normalized, (0, 0), 1.2, borderType=cv2.BORDER_REFLECT)
    residual = normalized - blur
    noise = _robust_scale(residual[background_mask])
    background = float(np.median(normalized[background_mask]))
    signal = max(0.0, float(np.percentile(normalized, 99.0)) - background)
    contrast_to_noise = signal / noise

    height, width = normalized.shape
    tile_medians = []
    for row in range(4):
        for col in range(4):
            tile = normalized[row * height // 4:(row + 1) * height // 4, col * width // 4:(col + 1) * width // 4]
            tile_medians.append(float(np.median(tile)))
    gradient = float(np.percentile(tile_medians, 90) - np.percentile(tile_medians, 10))
    clipped_high = float(np.mean(normalized >= 0.998))
    clipped_low = float(np.mean(normalized <= 0.002))

    snr_component = float(np.clip((np.log10(max(contrast_to_noise, 1.0)) - 0.65) / 1.25, 0.0, 1.0))
    noise_component = float(np.clip((0.045 - noise) / 0.038, 0.0, 1.0))
    gradient_component = float(np.clip((0.15 - gradient) / 0.13, 0.0, 1.0))
    clipping_component = float(np.clip(1.0 - clipped_high / 0.018 - clipped_low / 0.35, 0.0, 1.0))
    score = int(round(100.0 * (0.48 * snr_component + 0.24 * noise_component + 0.18 * gradient_component + 0.10 * clipping_component)))
    score = max(1, min(100, score))

    if score >= 85:
        grade = "Excellent"
    elif score >= 70:
        grade = "Good"
    elif score >= 50:
        grade = "Fair"
    else:
        grade = "Limited"

    limiting_factors: list[str] = []
    if snr_component < 0.58:
        limiting_factors.append("Faint signal relative to background noise")
    if noise_component < 0.55:
        limiting_factors.append("Visible background noise")
    if gradient_component < 0.55:
        limiting_factors.append("Uneven background or gradients")
    if clipping_component < 0.65:
        limiting_factors.append("Clipped shadows or highlights")
    if not limiting_factors:
        limiting_factors.append("No major capture limitation detected")

    target_score = 78.0
    exposure_multiplier = float(np.clip((target_score / max(score, 25)) ** 2, 1.0, 8.0))
    total_exposure = _exposure_seconds(path)
    additional = total_exposure * (exposure_multiplier - 1.0) if total_exposure and exposure_multiplier > 1.05 else None
    if exposure_multiplier <= 1.05:
        recommendation = "Exposure depth is already strong; prioritize calibration and processing."
    elif additional is not None:
        recommendation = f"For a stronger signal, capture {_format_duration(additional)} under similar conditions."
    else:
        recommendation = f"For a stronger signal, aim for about {exposure_multiplier:.1f}× the current total exposure."

    return {
        "score": score,
        "grade": grade,
        "summary": f"{grade} source data based on signal, noise, background uniformity, and clipping.",
        "recommendation": recommendation,
        "total_exposure_seconds": round(total_exposure, 1) if total_exposure else None,
        "recommended_additional_seconds": round(additional, 1) if additional is not None else None,
        "exposure_multiplier": round(exposure_multiplier, 1),
        "factors": limiting_factors[:3],
    }
