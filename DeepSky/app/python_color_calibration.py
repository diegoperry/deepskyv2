from __future__ import annotations

from typing import Callable

import cv2
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


def _mtf_curve(channel: np.ndarray, midpoint: float) -> np.ndarray:
    m = float(np.clip(midpoint, 0.01, 0.99))
    c = np.clip(channel, 0.0, 1.0)
    denominator = ((2.0 * m - 1.0) * c) - m
    denominator = np.where(np.abs(denominator) < 1e-6, -1e-6, denominator)
    return np.clip(((m - 1.0) * c) / denominator, 0.0, 1.0)


def _mtf_midpoint_for_background(background: float, target: float) -> float:
    c = float(np.clip(background, 1e-5, 1.0 - 1e-5))
    t = float(np.clip(target, 1e-5, 1.0 - 1e-5))
    denominator = (2.0 * t * c) - t - c
    if abs(denominator) < 1e-6:
        return 0.5
    return float(np.clip(c * (t - 1.0) / denominator, 0.01, 0.99))


def _robust_sigma(channel: np.ndarray, median: float) -> float:
    low_side = channel[channel <= median]
    sample = low_side if low_side.size >= 32 else channel.ravel()
    mad = np.median(np.abs(sample - np.median(sample)))
    sigma = 1.4826 * mad
    if sigma <= 1e-6:
        sigma = float(np.std(sample))
    return max(sigma, 1e-6)


def _auto_mtf_stretch(rgb: np.ndarray, target_background: float, log: LogCallback | None, label: str) -> np.ndarray:
    output = np.empty_like(rgb, dtype=np.float32)
    for channel_index, channel_name in enumerate(("R", "G", "B")):
        channel = rgb[..., channel_index]
        median = float(np.median(channel))
        sigma = _robust_sigma(channel, median)
        shadow = max(0.0, median - 2.8 * sigma)
        normalized = np.clip((channel - shadow) / max(1e-6, 1.0 - shadow), 0.0, 1.0)
        background = float(np.median(normalized))
        midpoint = _mtf_midpoint_for_background(background, target_background)
        output[..., channel_index] = _mtf_curve(normalized, midpoint)
        if log:
            log(
                f"{label} MTF {channel_name}: median={median:.5f}, sigma={sigma:.5f}, "
                f"shadow={shadow:.5f}, midpoint={midpoint:.5f}, target_bg={target_background:.3f}"
            )
    return output


def _polynomial_terms(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])


def _fit_background_surface(channel: np.ndarray, grid_size: int, log: LogCallback | None, channel_name: str) -> np.ndarray:
    height, width = channel.shape
    xs: list[float] = []
    ys: list[float] = []
    values: list[float] = []
    weights: list[float] = []

    for gy in range(grid_size):
        y0 = int(round(gy * height / grid_size))
        y1 = int(round((gy + 1) * height / grid_size))
        for gx in range(grid_size):
            x0 = int(round(gx * width / grid_size))
            x1 = int(round((gx + 1) * width / grid_size))
            cell = channel[y0:y1, x0:x1]
            if cell.size < 16:
                continue
            floor = float(np.percentile(cell, 15.0))
            spread = float(np.percentile(cell, 25.0) - np.percentile(cell, 5.0))
            xs.append(((x0 + x1) * 0.5 / max(1, width - 1)) * 2.0 - 1.0)
            ys.append(((y0 + y1) * 0.5 / max(1, height - 1)) * 2.0 - 1.0)
            values.append(floor)
            weights.append(1.0 / max(spread, 1e-4))

    if len(values) < 6:
        if log:
            log(f"Python background extraction {channel_name}: skipped, not enough grid samples.")
        return np.zeros_like(channel)

    x_arr = np.asarray(xs, dtype=np.float32)
    y_arr = np.asarray(ys, dtype=np.float32)
    value_arr = np.asarray(values, dtype=np.float32)
    weight_arr = np.sqrt(np.asarray(weights, dtype=np.float32))
    design = _polynomial_terms(x_arr, y_arr)
    coeffs, *_ = np.linalg.lstsq(design * weight_arr[:, None], value_arr * weight_arr, rcond=None)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    xx = (xx / max(1, width - 1)) * 2.0 - 1.0
    yy = (yy / max(1, height - 1)) * 2.0 - 1.0
    surface = _polynomial_terms(xx.ravel(), yy.ravel()).dot(coeffs).reshape(height, width)
    surface = np.clip(surface, 0.0, float(np.percentile(channel, 80.0)))
    if log:
        log(
            f"Python background extraction {channel_name}: samples={len(values)}, "
            f"surface_median={np.median(surface):.5f}, surface_max={np.max(surface):.5f}"
        )
    return surface.astype(np.float32)


def _extract_background(rgb: np.ndarray, log: LogCallback | None) -> np.ndarray:
    output = np.empty_like(rgb, dtype=np.float32)
    for channel_index, channel_name in enumerate(("R", "G", "B")):
        surface = _fit_background_surface(rgb[..., channel_index], 16, log, channel_name)
        output[..., channel_index] = np.clip(rgb[..., channel_index] - surface, 0.0, 1.0)
    return output


def _neutralize_background(rgb: np.ndarray, log: LogCallback | None) -> np.ndarray:
    luminance = _luminance(rgb)
    hsv = cv2.cvtColor(
        np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8),
        cv2.COLOR_RGB2HSV,
    )
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    mask = (luminance < np.percentile(luminance, 45.0)) & (saturation < np.percentile(saturation, 70.0))
    if int(np.count_nonzero(mask)) < 128:
        mask = luminance < np.percentile(luminance, 35.0)

    background = np.median(rgb[mask], axis=0)
    neutral = float(np.mean(background))
    gains = neutral / np.maximum(background, 1e-4)
    gains = np.clip(gains, 0.65, 1.55)
    output = np.clip(rgb * gains.reshape(1, 1, 3), 0.0, 1.0)
    after = np.median(output[mask], axis=0)
    if log:
        log(
            "Python background neutralization: "
            f"median_RGB={background[0]:.5f}, {background[1]:.5f}, {background[2]:.5f}, "
            f"gains={gains[0]:.3f}, {gains[1]:.3f}, {gains[2]:.3f}, "
            f"after_RGB={after[0]:.5f}, {after[1]:.5f}, {after[2]:.5f}"
        )
    return output


def _dilated_component_mask(labels: np.ndarray, component_id: int) -> np.ndarray:
    mask = (labels == component_id).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(mask, kernel, iterations=1).astype(bool)


def _detect_star_mask(rgb: np.ndarray, log: LogCallback | None) -> np.ndarray:
    luminance = _luminance(rgb)
    threshold = float(np.percentile(luminance, 97.0))
    star_map = (luminance >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(star_map, connectivity=8)

    mask = np.zeros(luminance.shape, dtype=np.float32)
    accepted = 0
    for component_id in range(1, count):
        x, y, width, height, area = stats[component_id]
        if area <= 4 or area >= 200:
            continue
        aspect_ratio = max(width / max(1, height), height / max(1, width))
        if aspect_ratio >= 2.5:
            continue
        mask[_dilated_component_mask(labels, component_id)] = 1.0
        accepted += 1

    if accepted:
        mask = cv2.GaussianBlur(mask, (0, 0), 0.8)
        peak = float(np.max(mask))
        if peak > 0:
            mask = np.clip(mask / peak, 0.0, 1.0)
    if log:
        log(f"Python star mask: accepted_components={accepted}, pixels={int(np.count_nonzero(mask > 0.35))}")
    return mask.astype(np.float32)


def _estimate_star_white_balance(rgb: np.ndarray, star_mask: np.ndarray, log: LogCallback | None) -> np.ndarray:
    star_pixels = star_mask > 0.35
    if int(np.count_nonzero(star_pixels)) < 24:
        if log:
            log("Python star white balance skipped: not enough star pixels.")
        return np.ones(3, dtype=np.float32)

    luminance = _luminance(rgb)
    star_luminance = luminance[star_pixels]
    low = np.percentile(star_luminance, 25.0)
    high = np.percentile(star_luminance, 92.0)
    unsaturated = star_pixels & (luminance >= low) & (luminance <= high)
    pixels = rgb[unsaturated]
    if pixels.shape[0] < 24:
        if log:
            log("Python star white balance skipped: not enough unsaturated star pixels.")
        return np.ones(3, dtype=np.float32)

    channel_max = np.max(pixels, axis=1)
    channel_min = np.min(pixels, axis=1)
    color_spread = (channel_max - channel_min) / np.maximum(channel_max, 1e-4)
    pixels = pixels[color_spread < np.percentile(color_spread, 70.0)]
    if pixels.shape[0] < 24:
        if log:
            log("Python star white balance skipped: color-spread filter removed too many stars.")
        return np.ones(3, dtype=np.float32)

    measured = np.median(pixels, axis=0)
    neutral = float(np.mean(measured))
    gains = neutral / np.maximum(measured, 1e-4)
    gains = gains / max(float(np.mean(gains)), 1e-6)
    gains = np.clip(gains, 0.78, 1.22).astype(np.float32)
    if log:
        log(
            "Python star white balance: "
            f"sample_pixels={pixels.shape[0]}, measured_RGB={measured[0]:.5f}, {measured[1]:.5f}, {measured[2]:.5f}, "
            f"gains={gains[0]:.3f}, {gains[1]:.3f}, {gains[2]:.3f}"
        )
    return gains


def _apply_white_balance(rgb: np.ndarray, gains: np.ndarray, star_mask: np.ndarray | None, log: LogCallback | None) -> np.ndarray:
    if np.allclose(gains, 1.0):
        return rgb

    balanced = np.clip(rgb * gains.reshape(1, 1, 3), 0.0, 1.0)
    luminance = _luminance(rgb)
    signal = np.clip((luminance - np.percentile(luminance, 20.0)) / 0.55, 0.0, 1.0)
    mix = np.clip(signal[:, :, None] * 0.72, 0.0, 0.72)
    if star_mask is not None:
        mix = np.maximum(mix, np.clip(star_mask[:, :, None] * 0.9, 0.0, 0.9))
    if log:
        log(f"Python white balance applied: mean_mix={float(np.mean(mix)):.5f}, max_mix={float(np.max(mix)):.5f}")
    return np.clip(rgb * (1.0 - mix) + balanced * mix, 0.0, 1.0)


def _fallback_remove_green(rgb: np.ndarray, log: LogCallback | None) -> np.ndarray:
    excess_green = np.maximum(0.0, rgb[..., 1] - 0.5 * (rgb[..., 0] + rgb[..., 2]))
    output = rgb.copy()
    output[..., 1] = np.clip(output[..., 1] - excess_green * 0.05, 0.0, 1.0)
    if log:
        log("Python star white balance fallback: applied mild green reduction.")
    return output


def _duoband_emission_green_control(rgb: np.ndarray, log: LogCallback | None) -> np.ndarray:
    luminance = _luminance(rgb)
    sky_mask = luminance < np.percentile(luminance, 55.0)
    if int(np.count_nonzero(sky_mask)) < 256:
        return rgb

    sky = np.median(rgb[sky_mask], axis=0)
    green_excess = sky[1] - max(sky[0], sky[2])
    if green_excess <= 0.015 and sky[1] <= (0.5 * (sky[0] + sky[2]) * 1.35 + 0.01):
        return rgb

    output = rgb.copy()
    neutral_green = 0.58 * output[..., 0] + 0.42 * output[..., 2]
    excess = np.maximum(0.0, output[..., 1] - neutral_green)
    sky_weight = np.clip(
        (np.percentile(luminance, 78.0) - luminance)
        / max(1e-6, np.percentile(luminance, 78.0) - np.percentile(luminance, 3.0)),
        0.0,
        1.0,
    )
    star_weight = np.clip(
        (luminance - np.percentile(luminance, 96.5))
        / max(1e-6, np.percentile(luminance, 99.95) - np.percentile(luminance, 96.5)),
        0.0,
        1.0,
    )
    reduction = np.clip((0.88 * sky_weight + 0.55) * (1.0 - star_weight * 0.70), 0.0, 0.95)
    output[..., 1] = np.clip(output[..., 1] - excess * reduction, 0.0, 1.0)

    red_signal = np.clip(
        (output[..., 0] - np.percentile(output[..., 0], 55.0))
        / max(1e-6, np.percentile(output[..., 0], 99.2) - np.percentile(output[..., 0], 55.0)),
        0.0,
        1.0,
    )
    emission_boost = np.clip(red_signal * (1.0 - star_weight) * 0.22, 0.0, 0.22)
    output[..., 0] = np.clip(output[..., 0] * (1.0 + emission_boost), 0.0, 1.0)

    if log:
        after = np.median(output[sky_mask], axis=0)
        log(
            "Python duo-band green control: "
            f"sky_before_RGB={sky[0]:.5f}, {sky[1]:.5f}, {sky[2]:.5f}, "
            f"sky_after_RGB={after[0]:.5f}, {after[1]:.5f}, {after[2]:.5f}, "
            f"mean_reduction={float(np.mean(reduction)):.5f}"
        )
    return np.clip(output, 0.0, 1.0)


def _color_calibrate(rgb: np.ndarray, log: LogCallback | None) -> np.ndarray:
    star_mask = _detect_star_mask(rgb, log)
    gains = _estimate_star_white_balance(rgb, star_mask, log)
    if np.allclose(gains, 1.0):
        star_pixels = star_mask > 0.35
        if int(np.count_nonzero(star_pixels)) < 24:
            return _fallback_remove_green(rgb, log)
        return rgb
    return _apply_white_balance(rgb, gains, star_mask, log)


def _legacy_component_color_calibrate(rgb: np.ndarray, log: LogCallback | None) -> np.ndarray:
    luminance = _luminance(rgb)
    threshold = float(np.percentile(luminance, 97.0))
    star_map = (luminance >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(star_map, connectivity=8)

    samples: list[np.ndarray] = []
    accepted = 0
    for component_id in range(1, count):
        x, y, width, height, area = stats[component_id]
        if area <= 4 or area >= 200:
            continue
        aspect_ratio = max(width / max(1, height), height / max(1, width))
        if aspect_ratio >= 2.5:
            continue
        mask = _dilated_component_mask(labels, component_id)
        samples.append(rgb[mask])
        accepted += 1

    if accepted < 5 or not samples:
        excess_green = np.maximum(0.0, rgb[..., 1] - 0.5 * (rgb[..., 0] + rgb[..., 2]))
        fallback = rgb.copy()
        fallback[..., 1] = np.clip(fallback[..., 1] - excess_green * 0.05, 0.0, 1.0)
        if log:
            log(f"Python star color calibration fallback: accepted_stars={accepted}; applied mild green reduction.")
        return fallback

    star_pixels = np.concatenate(samples, axis=0)
    medians = np.median(star_pixels, axis=0)
    target = float(np.mean(medians))
    gains = np.clip(target / np.maximum(medians, 1e-6), 0.6, 1.6)
    if log:
        log(
            f"Python star color calibration: accepted_stars={accepted}, "
            f"star_median_RGB={medians[0]:.5f}, {medians[1]:.5f}, {medians[2]:.5f}, "
            f"gains={gains[0]:.3f}, {gains[1]:.3f}, {gains[2]:.3f}"
        )
    return np.clip(rgb * gains.reshape(1, 1, 3), 0.0, 1.0)


def _second_mtf(rgb: np.ndarray, midpoint: float, log: LogCallback | None) -> np.ndarray:
    output = np.empty_like(rgb, dtype=np.float32)
    for channel_index, channel_name in enumerate(("R", "G", "B")):
        output[..., channel_index] = _mtf_curve(rgb[..., channel_index], midpoint)
        if log:
            log(f"Python second MTF {channel_name}: midpoint={midpoint:.3f}")
    return output


def _neutralize_sky(rgb: np.ndarray, log: LogCallback | None) -> np.ndarray:
    luminance = _luminance(rgb)
    mask = luminance <= np.percentile(luminance, 15.0)
    if int(np.count_nonzero(mask)) < 32:
        if log:
            log("Python sky neutralization skipped: not enough dark sky pixels.")
        return rgb
    sky = np.median(rgb[mask], axis=0)
    output = np.clip(rgb - (sky * 0.90).reshape(1, 1, 3), 0.0, 1.0)
    after = np.median(output[mask], axis=0)
    if log:
        log(f"Python sky median before RGB={sky[0]:.5f}, {sky[1]:.5f}, {sky[2]:.5f}")
        log(f"Python sky median after RGB={after[0]:.5f}, {after[1]:.5f}, {after[2]:.5f}")
    return output


def python_fallback_color_calibration(
    image: np.ndarray,
    log: LogCallback | None = None,
    second_midpoint: float = 0.32,
) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        if log:
            log("Python color calibration skipped: source is not RGB.")
        return arr

    rgb = _to_float01(arr)
    if log:
        log("Python fallback color calibration started.")
    rgb = _auto_mtf_stretch(rgb, target_background=0.125, log=log, label="Python first")
    rgb = _extract_background(rgb, log)
    rgb = _neutralize_background(rgb, log)
    rgb = _color_calibrate(rgb, log)
    rgb = _duoband_emission_green_control(rgb, log)
    rgb = _second_mtf(rgb, midpoint=second_midpoint, log=log)
    rgb = _duoband_emission_green_control(rgb, log)
    rgb = _neutralize_sky(rgb, log)
    rgb = _duoband_emission_green_control(rgb, log)
    return _to_uint16(rgb)
