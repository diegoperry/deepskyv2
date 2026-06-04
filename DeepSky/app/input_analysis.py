from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

from .image_io import load_image


@dataclass(frozen=True)
class StretchAnalysis:
    likely_stretched: bool
    confidence: str
    message: str
    metrics: dict[str, float]
    recommended_mode: str
    recommended_reason: str


SEESTAR_HEADER_KEYS = ("TELESCOP", "INSTRUME", "CREATOR", "PRODUCER", "OBSERVER", "PROGRAM")


def detect_telescope_profile(path: Path) -> str:
    source = Path(path)
    if source.suffix.lower() not in {".fit", ".fits", ".fts"}:
        return "generic"
    try:
        header = fits.getheader(source)
    except Exception:
        return "generic"

    values = " ".join(str(header.get(key, "")) for key in SEESTAR_HEADER_KEYS).lower()
    if "seestar" in values or "zwo seestar" in values:
        return "seestar"
    return "generic"


def _luminance(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        rgb = arr[..., :3].astype(np.float32)
        return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    return np.squeeze(arr).astype(np.float32)


def _normalize_for_histogram(values: np.ndarray) -> np.ndarray:
    data = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.zeros(1, dtype=np.float32)

    low = float(np.percentile(data, 0.01))
    high = float(np.percentile(data, 99.99))
    if high <= low:
        high = float(np.max(data))
        low = float(np.min(data))
    if high <= low:
        return np.zeros_like(data, dtype=np.float32)
    return np.clip((data - low) / (high - low), 0.0, 1.0).astype(np.float32)


def analyze_input_stretch(path: Path) -> StretchAnalysis:
    image = load_image(path)
    raw_luminance = _luminance(image)
    finite_luminance = np.nan_to_num(raw_luminance.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite_luminance = finite_luminance[np.isfinite(finite_luminance)]
    raw_scale = 1.0
    if finite_luminance.size and float(np.nanmax(finite_luminance)) > 1.0:
        raw_scale = 65535.0 if float(np.nanmax(finite_luminance)) > 1024.0 else 255.0
    raw_p50 = float(np.percentile(finite_luminance, 50) / raw_scale) if finite_luminance.size else 0.0
    raw_p99 = float(np.percentile(finite_luminance, 99) / raw_scale) if finite_luminance.size else 0.0
    raw_p999 = float(np.percentile(finite_luminance, 99.9) / raw_scale) if finite_luminance.size else 0.0
    luminance = _normalize_for_histogram(raw_luminance)

    p01, p1, p10, p50, p90, p99 = [float(np.percentile(luminance, p)) for p in (0.1, 1, 10, 50, 90, 99)]
    bright_fraction = float(np.mean(luminance > 0.80))
    shadow_fraction = float(np.mean(luminance < 0.03))
    midtone_fraction = float(np.mean((luminance > 0.12) & (luminance < 0.65)))
    background_lift = p10
    dynamic_width = max(1e-6, p99 - p01)

    score = 0
    if p50 > 0.08:
        score += 2
    elif p50 > 0.045:
        score += 1
    if background_lift > 0.025:
        score += 1
    if midtone_fraction > 0.20:
        score += 1
    if shadow_fraction < 0.45:
        score += 1
    if bright_fraction > 0.005 and dynamic_width < 0.96:
        score += 1

    low_absolute_signal = raw_p50 < 0.14 and raw_p99 < 0.20 and raw_p999 < 0.35
    if low_absolute_signal and score <= 3:
        score = min(score, 2)

    if score >= 5 or (p50 > 0.075 and background_lift > 0.04 and shadow_fraction < 0.40 and not low_absolute_signal):
        recommended_mode = "pre_stretched"
        recommended_reason = "hard-stretched histogram; skip the main stretch to avoid noise and blown highlights"
    elif score >= 3:
        recommended_mode = "gentle_stretch"
        recommended_reason = "soft-stretched histogram; use a gentle stretch instead of full linear stretch"
    else:
        recommended_mode = "linear"
        recommended_reason = "linear/raw-style histogram; use the normal stretch"

    likely_stretched = score >= 3
    confidence = "high" if score >= 5 else "medium" if score >= 3 else "low"
    if likely_stretched:
        message = (
            "This input looks like it may already be stretched. DeepSky works best with raw/linear stacked "
            "FITS or TIFF data; pre-stretched files can produce washed-out color, crushed shadows, or extra noise."
        )
    else:
        message = "Input looks compatible with linear/raw-style processing."

    return StretchAnalysis(
        likely_stretched=likely_stretched,
        confidence=confidence,
        message=message,
        metrics={
            "p01": p01,
            "p1": p1,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "p99": p99,
            "bright_fraction": bright_fraction,
            "shadow_fraction": shadow_fraction,
            "midtone_fraction": midtone_fraction,
            "raw_p50": raw_p50,
            "raw_p99": raw_p99,
            "raw_p999": raw_p999,
            "score": float(score),
        },
        recommended_mode=recommended_mode,
        recommended_reason=recommended_reason,
    )
