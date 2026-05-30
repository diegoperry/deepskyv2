from __future__ import annotations

from typing import Callable

import numpy as np


LogCallback = Callable[[str], None]


def _to_float01(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    data = arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.integer):
        max_value = float(np.iinfo(arr.dtype).max)
        if max_value > 0:
            data /= max_value
    elif data.size and float(np.nanmax(data)) > 1.0:
        data /= 65535.0
    return np.nan_to_num(data[..., :3], nan=0.0, posinf=0.0, neginf=0.0).clip(0.0, 1.0)


def _to_uint16(data: np.ndarray) -> np.ndarray:
    return (np.clip(data, 0.0, 1.0) * 65535.0).round().astype(np.uint16)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _channel_gains(medians: np.ndarray, limit: tuple[float, float]) -> np.ndarray:
    medians = np.maximum(medians.astype(np.float32), 1e-6)
    target = float(np.median(medians))
    return np.clip(target / medians, limit[0], limit[1])


def _apply_gains(rgb: np.ndarray, gains: np.ndarray, blend: float) -> np.ndarray:
    blended = 1.0 + (gains - 1.0) * blend
    return np.clip(rgb * blended.reshape(1, 1, 3), 0.0, 1.0)


def _background_mask(rgb: np.ndarray) -> np.ndarray:
    lum = _luminance(rgb)
    low = np.percentile(lum, 5.0)
    high = np.percentile(lum, 28.0)
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    chroma_limit = np.percentile(chroma, 75.0)
    return (lum >= low) & (lum <= high) & (chroma <= chroma_limit)


def _local_maxima_mask(lum: np.ndarray) -> np.ndarray:
    neighbor_max = np.full(lum.shape, -np.inf, dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbor_max = np.maximum(neighbor_max, np.roll(np.roll(lum, dy, axis=0), dx, axis=1))
    mask = lum > neighbor_max
    mask[[0, -1], :] = False
    mask[:, [0, -1]] = False
    return mask


def _star_mask(rgb: np.ndarray) -> np.ndarray:
    lum = _luminance(rgb)
    bright = lum >= np.percentile(lum, 98.8)
    unsaturated = np.max(rgb, axis=2) < 0.94
    compact = _local_maxima_mask(lum)
    return bright & unsaturated & compact


def _log_median(log: LogCallback | None, label: str, medians: np.ndarray) -> None:
    if log:
        log(f"{label} median RGB={medians[0]:.5f}, {medians[1]:.5f}, {medians[2]:.5f}")


def _saturate(rgb: np.ndarray, saturation: int) -> np.ndarray:
    strength = max(0.0, min(1.0, float(saturation) / 100.0))
    if strength <= 0:
        return rgb
    color_power = min(1.2, strength * 6.5)
    lum2d = _luminance(rgb)
    lum = lum2d[..., None]
    star_lum = lum2d
    star_protect = np.clip(
        (star_lum - np.percentile(star_lum, 98.7))
        / max(1e-6, np.percentile(star_lum, 99.98) - np.percentile(star_lum, 98.7)),
        0.0,
        1.0,
    )
    factor = 1.0 + color_power * 7.0 * (1.0 - star_protect[..., None])
    saturated = np.clip(lum + (rgb - lum) * factor, 0.0, 1.0)

    red_excess = np.clip(rgb[..., 0] - 0.5 * (rgb[..., 1] + rgb[..., 2]), 0.0, 1.0)
    red_floor = np.percentile(red_excess, 55.0)
    red_high = np.percentile(red_excess, 99.7)
    if red_high > red_floor:
        red_mask = np.clip((red_excess - red_floor) / (red_high - red_floor), 0.0, 1.0) ** 0.55
        nebula_signal = np.clip(
            (lum2d - np.percentile(lum2d, 48.0))
            / max(1e-6, np.percentile(lum2d, 99.2) - np.percentile(lum2d, 48.0)),
            0.0,
            1.0,
        ) ** 0.9
        diffuse = nebula_signal * (1.0 - star_protect) * (0.45 + 0.55 * red_mask) * color_power
        saturated[..., 0] += diffuse * 0.34
        saturated[..., 1] += diffuse * 0.055
        saturated[..., 2] -= diffuse * 0.13

        cool_shadow = (1.0 - red_mask) * nebula_signal * (1.0 - star_protect) * color_power
        saturated[..., 2] += cool_shadow * 0.035

    return np.clip(saturated, 0.0, 1.0)


def calibrate_basic_color(image: np.ndarray, saturation: int, log: LogCallback | None = None) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return arr

    rgb = _to_float01(arr)
    bg_mask = _background_mask(rgb)
    if int(np.count_nonzero(bg_mask)) >= 32:
        before_bg = np.median(rgb[bg_mask], axis=0)
        _log_median(log, "Background before calibration", before_bg)
        bg_gains = _channel_gains(before_bg, (0.86, 1.16))
        if before_bg[0] > before_bg[1] * 1.08 and before_bg[0] > before_bg[2] * 1.08:
            bg_gains[0] = min(bg_gains[0], 0.94)
        rgb = _apply_gains(rgb, bg_gains, blend=0.70)
        after_bg = np.median(rgb[bg_mask], axis=0)
        _log_median(log, "Background after calibration", after_bg)
        if log:
            log(f"Background channel gains={bg_gains[0]:.3f}, {bg_gains[1]:.3f}, {bg_gains[2]:.3f}; blend=0.70")
    elif log:
        log("Background neutralization skipped: not enough low-luminance background pixels.")

    star_mask = _star_mask(rgb)
    if int(np.count_nonzero(star_mask)) >= 8:
        star_before = np.median(rgb[star_mask], axis=0)
        _log_median(log, "Star before calibration", star_before)
        star_gains = _channel_gains(star_before, (0.90, 1.11))
        rgb = _apply_gains(rgb, star_gains, blend=0.35)
        star_after = np.median(rgb[star_mask], axis=0)
        _log_median(log, "Star after calibration", star_after)
        if log:
            log(f"Star channel gains={star_gains[0]:.3f}, {star_gains[1]:.3f}, {star_gains[2]:.3f}; blend=0.35")
    elif log:
        log("Star white balance skipped: not enough compact unsaturated stars.")

    if saturation > 0:
        rgb = _saturate(rgb, saturation)
        if log:
            log(f"Applied conservative post-balance saturation={saturation}")
        final_bg_mask = _background_mask(rgb)
        if int(np.count_nonzero(final_bg_mask)) >= 32:
            final_bg = np.median(rgb[final_bg_mask], axis=0)
            if final_bg[0] > max(final_bg[1], final_bg[2]) * 1.05:
                red_gain = max(0.94, max(final_bg[1], final_bg[2]) / max(final_bg[0], 1e-6))
                rgb[..., 0] = np.clip(rgb[..., 0] * red_gain, 0.0, 1.0)
                if log:
                    log(f"Color cast protection reduced red after saturation with gain={red_gain:.3f}")
                final_bg = np.median(rgb[final_bg_mask], axis=0)
            _log_median(log, "Background after saturation", final_bg)
    else:
        if log:
            log("Post-balance saturation disabled.")

    return _to_uint16(rgb)


def enhance_color(image: np.ndarray, saturation: int) -> np.ndarray:
    return calibrate_basic_color(image, saturation)
