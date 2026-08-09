from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np


LogCallback = Callable[[str], None]


def _float01(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def apply_galaxy_narrowband_finish(
    starless_image: np.ndarray,
    color_reference: np.ndarray,
    log: LogCallback | None = None,
) -> np.ndarray:
    """Finish a deconvolved starless galaxy without false-color painting."""
    base = _float01(starless_image)
    reference = _float01(color_reference)
    if base.ndim != 3 or base.shape[-1] < 3 or reference.shape != base.shape:
        return np.asarray(starless_image)

    raw_lum = _luminance(base).astype(np.float32)
    sky_lum = float(np.percentile(raw_lum, 35.0))
    sky_shift = max(0.0, sky_lum - 0.016)
    lum = np.clip(raw_lum - sky_shift, 0.0, 1.0)
    ref_lum = _luminance(reference).astype(np.float32)
    broad = cv2.GaussianBlur(lum, (0, 0), max(8.0, min(lum.shape[:2]) / 85.0))
    sky = float(np.percentile(broad, 45.0))
    mad = float(np.median(np.abs(broad - np.median(broad)))) * 1.4826
    signal = np.clip((broad - (sky + mad * 0.55)) / max(mad * 7.0, 0.012), 0.0, 1.0)
    signal = cv2.GaussianBlur((signal ** 0.72).astype(np.float32), (0, 0), 5.0)

    core_start = float(np.percentile(lum[signal > 0.08], 86.0)) if np.any(signal > 0.08) else 0.65
    core = np.clip((lum - core_start) / max(1.0 - core_start, 0.08), 0.0, 1.0)
    core = cv2.GaussianBlur(core.astype(np.float32), (0, 0), 2.2)
    rolled = lum - 0.30 * signal * core * lum * lum
    fine = lum - cv2.GaussianBlur(lum, (0, 0), 1.15)
    medium = cv2.GaussianBlur(lum, (0, 0), 1.4) - cv2.GaussianBlur(lum, (0, 0), 5.2)
    structure = np.clip(fine * 0.34 + medium * 0.72, -0.018, 0.030)
    out_lum = np.clip(rolled + structure * signal * (1.0 - core * 0.78), 0.0, 1.0)

    ref_chroma = reference - ref_lum[..., None]
    ref_chroma -= _luminance(ref_chroma)[..., None]
    measured = np.max(np.abs(ref_chroma), axis=2)
    confidence = np.clip(measured / max(float(np.percentile(measured, 98.8)), 0.006), 0.0, 1.0)
    confidence = cv2.GaussianBlur(confidence.astype(np.float32), (0, 0), 1.2)
    chroma_gain = 0.18 + signal * (1.12 + 0.48 * confidence)
    chroma = ref_chroma * chroma_gain[..., None]
    chroma_limit = np.minimum(0.105, 0.010 + measured * 1.72)
    chroma = np.clip(chroma, -chroma_limit[..., None], chroma_limit[..., None])
    chroma -= _luminance(chroma)[..., None]
    extent = np.max(np.abs(chroma), axis=2)
    headroom = np.minimum(out_lum, 1.0 - out_lum)
    chroma *= np.minimum(1.0, headroom / np.maximum(extent, 1e-6))[..., None]
    output = np.clip(out_lum[..., None] + chroma, 0.0, 1.0)

    if log:
        log(
            "Applied highlight-safe measured-color galaxy finish to deconvolved StarNet layer: "
            f"signal_mean={float(np.mean(signal)):.5f}, sky_shift={sky_shift:.5f}, core_start={core_start:.5f}, "
            f"luminance_p999={float(np.percentile(out_lum, 99.9)):.5f}; "
            "background neutralized and stars excluded from grading."
        )
    return np.clip(output * 65535.0 + 0.5, 0, 65535).astype(np.uint16)

def crop_galaxy_mosaic_footprint(
    image: np.ndarray,
    coverage_reference: np.ndarray,
    log: LogCallback | None = None,
) -> np.ndarray:
    """Crop stepped/rotated mosaic padding to a clean central rectangle."""
    output = np.asarray(image)
    coverage = _float01(coverage_reference)
    if output.shape[:2] != coverage.shape[:2] or coverage.ndim != 3:
        return output
    valid = np.max(coverage[..., :3], axis=2) > 0.001
    if float(np.mean(~valid)) < 0.02:
        return output
    h, w = valid.shape
    cx, cy = w // 2, h // 2
    y0, y1 = int(h * 0.18), int(h * 0.82)
    x0, x1 = 0, w
    for _ in range(5):
        good_columns = np.mean(valid[y0:y1], axis=0) > 0.992
        left = cx
        while left > 0 and good_columns[left - 1]:
            left -= 1
        right = cx
        while right < w - 1 and good_columns[right + 1]:
            right += 1
        x0, x1 = left, right + 1
        good_rows = np.mean(valid[:, x0:x1], axis=1) > 0.992
        top = cy
        while top > 0 and good_rows[top - 1]:
            top -= 1
        bottom = cy
        while bottom < h - 1 and good_rows[bottom + 1]:
            bottom += 1
        y0, y1 = top, bottom + 1
    if x1 - x0 < w * 0.25 or y1 - y0 < h * 0.25:
        return output
    if log:
        log(f"Cropped irregular galaxy mosaic footprint: x={x0}:{x1}, y={y0}:{y1}, output={x1-x0}x{y1-y0}.")
    return np.ascontiguousarray(output[y0:y1, x0:x1])
