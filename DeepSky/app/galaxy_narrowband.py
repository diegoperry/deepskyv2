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
    *,
    detail_reference: np.ndarray | None = None,
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
    core_excess = np.maximum(lum - core_start, 0.0)
    rolled = lum - signal * 0.22 * core_excess * core_excess / (core_excess + 0.10)
    fine = lum - cv2.GaussianBlur(lum, (0, 0), 1.15)
    medium = cv2.GaussianBlur(lum, (0, 0), 1.4) - cv2.GaussianBlur(lum, (0, 0), 5.2)
    structure = np.clip(fine * 0.46 + medium * 0.92, -0.020, 0.036)
    out_lum = np.clip(rolled + structure * signal, 0.0, 1.0)

    decon_detail_used = False
    if detail_reference is not None:
        detail_rgb = _float01(detail_reference)
        if detail_rgb.shape == base.shape:
            detail_lum = _luminance(detail_rgb).astype(np.float32)
            scale = max(float(np.percentile(detail_lum, 99.85)), 1e-4)
            detail_lum = np.arcsinh(np.clip(detail_lum / scale, 0.0, 6.0) * 2.2) / np.arcsinh(2.2)
            detail_lum = np.clip(detail_lum, 0.0, 1.0)
            decon_fine = detail_lum - cv2.GaussianBlur(detail_lum, (0, 0), 1.0)
            decon_medium = cv2.GaussianBlur(detail_lum, (0, 0), 1.3) - cv2.GaussianBlur(
                detail_lum, (0, 0), 4.5
            )
            source_fine = lum - cv2.GaussianBlur(lum, (0, 0), 1.0)
            recovered = np.clip(
                decon_fine * 0.38 + decon_medium * 0.60 - source_fine * 0.08,
                -0.014,
                0.034,
            )
            stellar_peak = np.clip(
                decon_fine / max(float(np.percentile(decon_fine, 99.65)), 1e-5) - 1.0,
                0.0,
                1.0,
            )
            stellar_peak = cv2.GaussianBlur(
                cv2.dilate(stellar_peak.astype(np.float32), np.ones((3, 3), np.uint8)),
                (0, 0),
                1.0,
            )
            bright_protect = np.clip((lum - 0.86) / 0.12, 0.0, 1.0)
            out_lum = np.clip(
                out_lum + recovered * signal * (1.0 - bright_protect * 0.75) * (1.0 - stellar_peak * 0.92),
                0.0,
                1.0,
            )
            decon_detail_used = True

    base_chroma = base - raw_lum[..., None]
    base_chroma -= _luminance(base_chroma)[..., None]
    ref_chroma = reference - ref_lum[..., None]
    ref_chroma -= _luminance(ref_chroma)[..., None]
    measured = np.max(np.abs(ref_chroma), axis=2)
    confidence = np.clip(measured / max(float(np.percentile(measured, 98.8)), 0.006), 0.0, 1.0)
    confidence = cv2.GaussianBlur(confidence.astype(np.float32), (0, 0), 1.2)
    chroma_gain = 0.06 + signal * (0.66 + 0.18 * confidence)
    chroma = base_chroma * chroma_gain[..., None] + ref_chroma * (signal * 0.08)[..., None]
    nucleus = cv2.GaussianBlur(
        np.clip((out_lum - core_start) / max(0.72 - core_start, 0.12), 0.0, 1.0).astype(np.float32),
        (0, 0),
        1.3,
    )
    chroma *= (1.0 - nucleus * 0.58)[..., None]
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
            f"post-denoise_decon_detail={str(decon_detail_used).lower()}; "
            "native starless texture retained, background neutralized, stars excluded from grading."
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

def finish_recomposed_galaxy_core(
    image: np.ndarray,
    detail_reference: np.ndarray | None = None,
    log: LogCallback | None = None,
) -> np.ndarray:
    """Restore a neutral, highlight-safe nucleus after StarNet recomposition."""
    rgb = _float01(image)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return np.asarray(image)
    lum = _luminance(rgb).astype(np.float32)
    sigma = max(9.0, min(lum.shape) / 110.0)
    broad = cv2.GaussianBlur(lum, (0, 0), sigma)
    sky = float(np.percentile(broad, 45.0))
    mad = float(np.median(np.abs(broad - np.median(broad)))) * 1.4826
    galaxy = np.clip((broad - sky - mad) / max(mad * 8.0, 0.010), 0.0, 1.0) ** 0.72
    galaxy = cv2.GaussianBlur(galaxy.astype(np.float32), (0, 0), 4.0)

    smooth = cv2.GaussianBlur(lum, (0, 0), 3.0)
    selected = smooth[galaxy > 0.18]
    core_start = float(np.percentile(selected, 87.0)) if selected.size else 0.72
    core_high = float(np.percentile(selected, 99.7)) if selected.size else 0.92
    core = np.clip((smooth - core_start) / max(core_high - core_start, 0.04), 0.0, 1.0)
    core = cv2.GaussianBlur((core * galaxy).astype(np.float32), (0, 0), 2.0)
    core /= max(float(np.max(core)), 1e-5)

    shoulder = 0.76
    excess = np.maximum(lum - shoulder, 0.0)
    compressed = np.where(
        lum > shoulder,
        shoulder + (1.0 - shoulder) * np.tanh(excess / (1.0 - shoulder)),
        lum,
    )
    out_lum = lum * (1.0 - core) + compressed * core
    disk_lift = np.sqrt(np.clip(out_lum, 0.0, 1.0))
    out_lum = out_lum * (1.0 - galaxy * 0.18) + disk_lift * (galaxy * 0.18)

    detail_used = False
    if detail_reference is not None:
        donor = _float01(detail_reference)
        if donor.shape == rgb.shape:
            donor_lum = _luminance(donor).astype(np.float32)
            scale = max(float(np.percentile(donor_lum, 99.85)), 1e-4)
            donor_lum = np.arcsinh(np.clip(donor_lum / scale, 0.0, 6.0) * 2.4) / np.arcsinh(2.4)
            detail = (
                donor_lum - cv2.GaussianBlur(donor_lum, (0, 0), 1.0)
            ) * 0.42 + (
                cv2.GaussianBlur(donor_lum, (0, 0), 1.3)
                - cv2.GaussianBlur(donor_lum, (0, 0), 5.0)
            ) * 0.76
            detail = np.clip(detail, -0.018, 0.040)
            out_lum = np.clip(out_lum + detail * galaxy * (1.0 - core * 0.45), 0.0, 1.0)
            detail_used = True

    white_nucleus = core**2.40
    out_lum += white_nucleus * np.maximum(0.90 - out_lum, 0.0) * 0.72

    chroma = rgb - lum[..., None]
    chroma -= _luminance(chroma)[..., None]
    chroma *= ((1.0 + galaxy * 0.48) * (1.0 - core * 0.92))[..., None]
    galaxy_level = np.clip(
        (broad - sky) / max(float(np.percentile(broad, 99.8)) - sky, 1e-5),
        0.0,
        1.0,
    )
    inner_disk = galaxy * np.sqrt(galaxy_level) * (1.0 - core)
    outer_disk = galaxy * ((1.0 - galaxy_level) ** 0.70) * (1.0 - core)
    pink = np.array([0.115, -0.034, 0.058], dtype=np.float32)
    blue = np.array([-0.040, -0.014, 0.120], dtype=np.float32)
    pink -= float(_luminance(pink))
    blue -= float(_luminance(blue))
    chroma += out_lum[..., None] * (
        inner_disk[..., None] * pink * 1.18 + outer_disk[..., None] * blue * 0.66
    )
    chroma *= (1.0 - white_nucleus)[..., None]
    extent = np.max(np.abs(chroma), axis=2)
    headroom = np.minimum(out_lum, 1.0 - out_lum)
    chroma *= np.minimum(1.0, headroom / np.maximum(extent, 1e-6))[..., None]
    output = np.clip(out_lum[..., None] + chroma, 0.0, 1.0)
    if log:
        log(
            "Finished recomposed galaxy nucleus: neutral-white highlight shoulder, "
            f"core_start={core_start:.5f}, core_high={core_high:.5f}, "
            f"core_max={float(np.max(core)):.5f}, "
            f"post_recomposition_decon_detail={str(detail_used).lower()}."
        )
    return np.clip(output * 65535.0 + 0.5, 0, 65535).astype(np.uint16)