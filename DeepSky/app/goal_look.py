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


def chroma_percentile(image: np.ndarray, percentile: float = 95.0) -> float:
    rgb = _to_float01(image)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return 0.0
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    return float(np.percentile(chroma, percentile))


def red_emission_dominance(image: np.ndarray) -> float:
    rgb = _to_float01(image)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return 0.0
    lum = _luminance(rgb)
    signal_mask = lum > np.percentile(lum, 35.0)
    red_excess = np.clip(rgb[..., 0] - 0.5 * (rgb[..., 1] + rgb[..., 2]), 0.0, 1.0)
    green_excess = np.clip(rgb[..., 1] - 0.5 * (rgb[..., 0] + rgb[..., 2]), 0.0, 1.0)
    blue_excess = np.clip(rgb[..., 2] - 0.5 * (rgb[..., 0] + rgb[..., 1]), 0.0, 1.0)
    red_p95 = float(np.percentile(red_excess[signal_mask], 95.0))
    other_p95 = max(
        float(np.percentile(green_excess[signal_mask], 95.0)),
        float(np.percentile(blue_excess[signal_mask], 95.0)),
        1e-6,
    )
    return red_p95 / other_p95


def reflection_nebula_bias(image: np.ndarray) -> float:
    rgb = _to_float01(image)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return 0.0
    lum = _luminance(rgb)
    signal_mask = lum > np.percentile(lum, 35.0)
    red_excess = np.clip(rgb[..., 0] - 0.5 * (rgb[..., 1] + rgb[..., 2]), 0.0, 1.0)
    green_excess = np.clip(rgb[..., 1] - 0.5 * (rgb[..., 0] + rgb[..., 2]), 0.0, 1.0)
    blue_excess = np.clip(rgb[..., 2] - 0.5 * (rgb[..., 0] + rgb[..., 1]), 0.0, 1.0)
    red_p95 = float(np.percentile(red_excess[signal_mask], 95.0))
    green_p95 = float(np.percentile(green_excess[signal_mask], 95.0))
    blue_p95 = float(np.percentile(blue_excess[signal_mask], 95.0))
    other_p95 = max(green_p95, blue_p95, 1e-6)
    low_red_bias = np.clip((1.55 - red_p95 / other_p95) / 0.85, 0.0, 1.0)
    green_cast = np.clip(
        (green_p95 - max(red_p95, blue_p95) * 0.72)
        / max(1e-6, green_p95 + max(red_p95, blue_p95)),
        0.0,
        1.0,
    )
    return float(np.clip(low_red_bias * (1.0 - green_cast * 2.8), 0.0, 1.0))


def _suppress_green_excess(rgb: np.ndarray, strength: float = 0.65) -> np.ndarray:
    lum = _luminance(rgb)
    green_target = np.maximum(0.5 * (rgb[..., 0] + rgb[..., 2]), rgb[..., 0] * 0.92)
    excess = np.maximum(0.0, rgb[..., 1] - green_target)
    signal = np.clip(
        (lum - np.percentile(lum, 15.0))
        / max(1e-6, np.percentile(lum, 95.0) - np.percentile(lum, 15.0)),
        0.0,
        1.0,
    )
    reduction = strength * (0.85 - 0.30 * signal)
    output = rgb.copy()
    output[..., 1] = np.clip(output[..., 1] - excess * reduction, 0.0, 1.0)
    return output


def _detail_mask(rgb: np.ndarray) -> np.ndarray:
    lum = _luminance(rgb).astype(np.float32)
    blurred = cv2.GaussianBlur(lum, (0, 0), 3.0)
    detail = np.abs(lum - blurred)
    high = float(np.percentile(detail, 99.2))
    if high <= 1e-6:
        return np.zeros_like(lum)
    mask = np.clip(detail / high, 0.0, 1.0)
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2)


def _large_star_mask(lum: np.ndarray) -> np.ndarray:
    threshold = max(float(np.percentile(lum, 99.45)), 0.055)
    mask = (lum > threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.6)
    return mask.astype(np.float32) / 255.0


def _repo_style_background_polish(
    rgb: np.ndarray,
    smooth: float,
    dark: float,
    chroma: float,
    protect_galaxy_detail: bool,
    log: LogCallback | None = None,
) -> np.ndarray:
    lum = _luminance(rgb)
    detail = _detail_mask(rgb)
    stars = _large_star_mask(lum)

    extended_lum = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 14.0 + 8.0 * smooth)
    galaxy_signal = np.clip(
        (extended_lum - np.percentile(extended_lum, 83.0))
        / max(1e-6, np.percentile(extended_lum, 99.6) - np.percentile(extended_lum, 83.0)),
        0.0,
        1.0,
    ) ** 0.72
    galaxy_signal = cv2.GaussianBlur(galaxy_signal.astype(np.float32), (0, 0), 5.0)
    if not protect_galaxy_detail:
        galaxy_signal *= 0.35

    protected = np.clip(np.maximum(stars, galaxy_signal), 0.0, 1.0)
    background_pixels = (
        (lum < np.percentile(lum, 72.0))
        & (protected < 0.16)
        & (detail < 0.28)
    )
    if int(np.count_nonzero(background_pixels)) < 512:
        background_pixels = (lum < np.percentile(lum, 58.0)) & (protected < 0.20)

    output = rgb.copy()
    if int(np.count_nonzero(background_pixels)) >= 512:
        background = np.median(output[background_pixels], axis=0)
        neutral = float(np.mean(background))
        gains = np.clip(neutral / np.maximum(background, 1e-4), 0.82, 1.18)
        corrected = np.clip(output * gains.reshape(1, 1, 3), 0.0, 1.0)
        shadow_mix = np.clip((0.46 - lum) / 0.46, 0.0, 1.0)
        shadow_mix *= np.clip(1.0 - protected * 0.88, 0.0, 1.0)
        output = np.clip(output * (1.0 - shadow_mix[..., None] * 0.80) + corrected * (shadow_mix[..., None] * 0.80), 0.0, 1.0)

    # Smooth chroma more than luminance, using RGB-safe Lab conversion.
    lab = cv2.cvtColor(np.clip(output * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    a_smooth = cv2.bilateralFilter(a_channel, d=9, sigmaColor=22 + 16 * chroma, sigmaSpace=13)
    b_smooth = cv2.bilateralFilter(b_channel, d=9, sigmaColor=22 + 16 * chroma, sigmaSpace=13)
    chroma_filtered = cv2.cvtColor(cv2.merge([l_channel, a_smooth, b_smooth]), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    background_mask = ((lum < np.percentile(lum, 76.0 + 10.0 * dark)) & (protected < 0.20)).astype(np.float32)
    background_mask = cv2.GaussianBlur(background_mask, (0, 0), 7.0 + 5.0 * smooth)
    background_mask = np.clip(background_mask * (1.0 - protected) * (0.72 + 0.26 * dark), 0.0, 0.98)
    chroma_mix = np.clip(background_mask[..., None] * (0.55 + 0.40 * chroma), 0.0, 0.95)
    output = np.clip(output * (1.0 - chroma_mix) + chroma_filtered * chroma_mix, 0.0, 1.0)

    smooth_source = cv2.bilateralFilter(
        np.clip(output * 255.0, 0, 255).astype(np.uint8),
        d=0,
        sigmaColor=20 + 32 * smooth,
        sigmaSpace=24 + 24 * smooth,
    ).astype(np.float32) / 255.0
    smooth_source = cv2.GaussianBlur(smooth_source, (0, 0), 0.8 + 1.6 * smooth)
    smooth_mix = np.clip(background_mask[..., None] * (0.55 + 0.40 * smooth), 0.0, 0.95)
    output = np.clip(output * (1.0 - smooth_mix) + smooth_source * smooth_mix, 0.0, 1.0)

    lum = _luminance(output)
    if int(np.count_nonzero(background_pixels)) >= 512:
        black = float(np.percentile(lum[background_pixels], 54.0 + 16.0 * dark))
    else:
        black = float(np.percentile(lum, 12.0 + 18.0 * dark))
    sky_scaled = np.clip((output - black * (0.68 + 0.26 * dark)) / max(1e-6, 1.0 - black * (0.68 + 0.26 * dark)), 0.0, 1.0)
    sky_scaled *= 0.58 + 0.32 * (1.0 - dark)
    black_mix = np.clip(background_mask[..., None] * (0.70 + 0.26 * dark), 0.0, 0.96)
    output = np.clip(output * (1.0 - black_mix) + sky_scaled * black_mix, 0.0, 1.0)

    if log:
        log(
            "Applied repo-style background polish: "
            f"background_pixels={int(np.count_nonzero(background_pixels))}, "
            f"background_mask_mean={float(np.mean(background_mask)):.5f}, black={black:.5f}"
        )
    return output


def _smooth_broadband_grain(rgb: np.ndarray, log: LogCallback | None = None) -> np.ndarray:
    lum = _luminance(rgb)
    star_protect = np.clip(
        (lum - np.percentile(lum, 97.0))
        / max(1e-6, np.percentile(lum, 99.98) - np.percentile(lum, 97.0)),
        0.0,
        1.0,
    ) ** 1.5
    core_protect = np.clip(
        (lum - np.percentile(lum, 76.0))
        / max(1e-6, np.percentile(lum, 99.4) - np.percentile(lum, 76.0)),
        0.0,
        1.0,
    )
    protect = np.maximum(star_protect, core_protect * 0.90)
    protect = cv2.GaussianBlur(protect.astype(np.float32), (0, 0), 2.0)
    smooth_mask = np.clip(1.0 - protect, 0.0, 1.0)
    smooth_mask *= np.clip(
        (np.percentile(lum, 98.0) - lum)
        / max(1e-6, np.percentile(lum, 98.0) - np.percentile(lum, 1.0)),
        0.0,
        1.0,
    ) ** 0.18

    rgb8 = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    denoised8 = cv2.fastNlMeansDenoisingColored(rgb8, None, 60, 120, 7, 45)
    denoised = denoised8.astype(np.float32) / 255.0

    lum_original = _luminance(rgb)
    blurred = cv2.GaussianBlur(rgb, (0, 0), 8.0)
    median = cv2.medianBlur(rgb8, 7).astype(np.float32) / 255.0
    smoothed = denoised * 0.35 + blurred * 0.50 + median * 0.15

    smoothed_lum_for_chroma = _luminance(smoothed)
    dark_chroma = np.clip(
        (np.percentile(lum_original, 62.0) - lum_original)
        / max(1e-6, np.percentile(lum_original, 62.0) - np.percentile(lum_original, 1.0)),
        0.0,
        1.0,
    ) ** 0.55
    smoothed = np.clip(
        smoothed_lum_for_chroma[..., None]
        + (smoothed - smoothed_lum_for_chroma[..., None]) * (1.0 - dark_chroma[..., None] * 0.82),
        0.0,
        1.0,
    )

    smooth_lum = cv2.GaussianBlur(lum_original, (0, 0), 3.5)
    target_lum = lum_original * (1.0 - smooth_mask * 0.96) + smooth_lum * (smooth_mask * 0.96)
    smoothed_lum = _luminance(smoothed)
    smoothed = np.clip(smoothed * (target_lum / np.maximum(smoothed_lum, 1e-5))[..., None], 0.0, 1.0)

    mix = np.clip(smooth_mask[..., None] * 0.998, 0.0, 0.998)
    result = np.clip(rgb * (1.0 - mix) + smoothed * mix, 0.0, 1.0)

    result_lum = _luminance(result)
    dark_sky = np.clip(
        (np.percentile(result_lum, 55.0) - result_lum)
        / max(1e-6, np.percentile(result_lum, 55.0) - np.percentile(result_lum, 1.0)),
        0.0,
        1.0,
    ) ** 0.8
    result = np.clip(
        result_lum[..., None] + (result - result_lum[..., None]) * (1.0 - dark_sky[..., None] * 0.70),
        0.0,
        1.0,
    )
    if log:
        log(f"Applied broadband grain smoothing: mean_mix={float(np.mean(mix)):.5f}, max_mix={float(np.max(mix)):.5f}")
    return result


def blend_broadband_background_denoise(
    image: np.ndarray,
    denoised_image: np.ndarray,
    smoothness: int = 55,
    darkness: int = 35,
    chroma_noise_reduction: int = 60,
    protect_galaxy_detail: bool = True,
    log: LogCallback | None = None,
) -> np.ndarray:
    rgb = _to_float01(image)
    denoised = _to_float01(denoised_image)
    if rgb.shape != denoised.shape:
        if log:
            log(f"Skipped background DeepSNR blend: shape mismatch {rgb.shape} vs {denoised.shape}")
        return image

    smooth = np.clip(float(smoothness) / 100.0, 0.0, 1.0)
    dark = np.clip(float(darkness) / 100.0, 0.0, 1.0)
    chroma = np.clip(float(chroma_noise_reduction) / 100.0, 0.0, 1.0)
    lum = _luminance(rgb)
    star_protect = np.clip(
        (lum - np.percentile(lum, 98.6))
        / max(1e-6, np.percentile(lum, 99.995) - np.percentile(lum, 98.6)),
        0.0,
        1.0,
    ) ** 1.15
    if protect_galaxy_detail:
        extended_lum = cv2.GaussianBlur(lum, (0, 0), 12.0 + 8.0 * smooth)
        galaxy_protect = np.clip(
            (extended_lum - np.percentile(extended_lum, 82.0))
            / max(1e-6, np.percentile(extended_lum, 99.5) - np.percentile(extended_lum, 82.0)),
            0.0,
            1.0,
        ) ** 0.72
        galaxy_protect = cv2.GaussianBlur(galaxy_protect.astype(np.float32), (0, 0), 5.0)
        protect = np.maximum(star_protect, galaxy_protect * 0.92)
    else:
        protect = np.maximum(star_protect, star_protect * 0.0)
    protect = cv2.GaussianBlur(protect.astype(np.float32), (0, 0), 1.5 + 1.8 * smooth)

    background = np.clip(
        (np.percentile(lum, 96.0) - lum)
        / max(1e-6, np.percentile(lum, 96.0) - np.percentile(lum, 1.0)),
        0.0,
        1.0,
    ) ** (0.34 - 0.20 * smooth)
    mask = np.clip(background * (1.0 - protect), 0.0, 1.0)

    denoised_lum = _luminance(denoised)
    matched = np.clip(denoised * (lum / np.maximum(denoised_lum, 1e-5))[..., None], 0.0, 1.0)

    background_blur = cv2.GaussianBlur(rgb, (0, 0), 2.0 + 5.0 * smooth)
    matched = np.clip(matched * (1.0 - smooth * 0.28) + background_blur * (smooth * 0.28), 0.0, 1.0)

    dark_sky = np.clip(
        (np.percentile(lum, 58.0) - lum)
        / max(1e-6, np.percentile(lum, 58.0) - np.percentile(lum, 1.0)),
        0.0,
        1.0,
    ) ** 0.7
    matched_lum = _luminance(matched)
    matched = np.clip(
        matched_lum[..., None]
        + (matched - matched_lum[..., None]) * (1.0 - dark_sky[..., None] * (0.18 + 0.78 * chroma)),
        0.0,
        1.0,
    )

    # Flatten only broad, low-level variation in empty sky. This keeps the sky sleek
    # without replacing local luminance detail in the galaxy or stars.
    broad_lum = cv2.GaussianBlur(lum, (0, 0), 16.0 + 18.0 * smooth)
    flattened_lum = np.clip(lum - (broad_lum - np.median(broad_lum)) * (0.12 + 0.20 * smooth), 0.0, 1.0)
    matched_lum = _luminance(matched)
    flatten_target_lum = matched_lum * (1.0 - mask * (0.10 + 0.22 * smooth)) + flattened_lum * (mask * (0.10 + 0.22 * smooth))
    matched = np.clip(matched * (flatten_target_lum / np.maximum(matched_lum, 1e-5))[..., None], 0.0, 1.0)

    mix_strength = 0.62 + 0.35 * smooth
    mix = np.clip(mask[..., None] * mix_strength, 0.0, mix_strength)
    blended = np.clip(rgb * (1.0 - mix) + matched * mix, 0.0, 1.0)

    blended_lum = _luminance(blended)
    empty_sky = np.clip(
        (np.percentile(blended_lum, 70.0) - blended_lum)
        / max(1e-6, np.percentile(blended_lum, 70.0) - np.percentile(blended_lum, 1.0)),
        0.0,
        1.0,
    ) ** (0.65 + 0.45 * (1.0 - dark))
    empty_sky *= np.clip(1.0 - protect, 0.0, 1.0)

    sky_floor = float(np.percentile(blended_lum, 12.0 + 20.0 * dark))
    sky_target = float(np.percentile(blended_lum, 0.5 + 2.0 * (1.0 - dark)))
    sky_lum = np.clip(
        (blended_lum - sky_floor * (0.70 + 0.40 * dark))
        / max(1e-6, 1.0 - sky_floor * (0.70 + 0.40 * dark)),
        0.0,
        1.0,
    )
    sky_lum = np.minimum(sky_lum, sky_target + sky_lum * (0.18 + 0.32 * (1.0 - dark)))
    darkened = np.clip(blended * (sky_lum / np.maximum(blended_lum, 1e-5))[..., None], 0.0, 1.0)
    darkened_lum = _luminance(darkened)
    darkened = np.clip(
        darkened_lum[..., None] + (darkened - darkened_lum[..., None]) * (1.0 - empty_sky[..., None] * (0.45 + 0.45 * chroma)),
        0.0,
        1.0,
    )
    black_mix = np.clip(empty_sky[..., None] * (0.55 + 0.40 * dark), 0.0, 0.95)
    blended = np.clip(blended * (1.0 - black_mix) + darkened * black_mix, 0.0, 1.0)

    final_lum = _luminance(blended)
    sky_cut = float(np.percentile(final_lum, 46.0 + 20.0 * dark))
    sky_rolloff = np.clip(final_lum / max(1e-6, sky_cut), 0.0, 1.0) ** (1.35 + 1.25 * dark)
    deep_black_mask = np.clip(empty_sky * (1.0 - sky_rolloff), 0.0, 1.0)
    black_strength = 0.72 + 0.24 * dark
    blended = np.clip(blended * (1.0 - deep_black_mask[..., None] * black_strength), 0.0, 1.0)

    final_lum = _luminance(blended)
    nearly_empty = np.clip(
        (sky_cut * 0.72 - final_lum) / max(1e-6, sky_cut * 0.72),
        0.0,
        1.0,
    ) * empty_sky
    neutral_lum = final_lum[..., None]
    blended = np.clip(
        neutral_lum + (blended - neutral_lum) * (1.0 - nearly_empty[..., None] * (0.75 + 0.20 * chroma)),
        0.0,
        1.0,
    )

    final_lum = _luminance(blended)
    star_core = np.clip(
        (final_lum - np.percentile(final_lum, 99.35))
        / max(1e-6, np.percentile(final_lum, 99.995) - np.percentile(final_lum, 99.35)),
        0.0,
        1.0,
    ) ** 0.95
    star_core = cv2.morphologyEx(
        star_core.astype(np.float32),
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    star_core = cv2.GaussianBlur(star_core.astype(np.float32), (0, 0), 0.65)

    extended_signal = cv2.GaussianBlur(final_lum, (0, 0), 14.0 + 10.0 * smooth)
    galaxy_signal = np.clip(
        (extended_signal - np.percentile(extended_signal, 84.0))
        / max(1e-6, np.percentile(extended_signal, 99.7) - np.percentile(extended_signal, 84.0)),
        0.0,
        1.0,
    ) ** 0.7
    galaxy_signal = cv2.GaussianBlur(galaxy_signal.astype(np.float32), (0, 0), 4.0)

    low_signal = np.clip(
        (np.percentile(final_lum, 82.0 + 8.0 * dark) - final_lum)
        / max(1e-6, np.percentile(final_lum, 82.0 + 8.0 * dark) - np.percentile(final_lum, 1.0)),
        0.0,
        1.0,
    ) ** (0.35 + 0.45 * (1.0 - dark))
    background_mask = np.clip(low_signal * (1.0 - np.maximum(star_core, galaxy_signal)), 0.0, 1.0)

    background_lum = cv2.GaussianBlur(final_lum, (0, 0), 2.5 + 5.0 * smooth)
    background_lum = np.minimum(background_lum, np.percentile(final_lum, 8.0 + 10.0 * (1.0 - dark)))
    black_target = background_lum[..., None] * (0.03 + 0.10 * (1.0 - dark))
    neutral_blended = final_lum[..., None] + (blended - final_lum[..., None]) * (1.0 - background_mask[..., None] * (0.80 + 0.18 * chroma))
    black_blend = np.clip(background_mask[..., None] * (0.78 + 0.20 * dark), 0.0, 0.98)
    blended = np.clip(neutral_blended * (1.0 - black_blend) + black_target * black_blend, 0.0, 1.0)
    blended = _repo_style_background_polish(blended, smooth, dark, chroma, protect_galaxy_detail, log)
    if log:
        log(
            "Blended DeepSNR into broadband background: "
            f"smoothness={smoothness}, darkness={darkness}, chroma_noise_reduction={chroma_noise_reduction}, "
            f"protect_galaxy_detail={protect_galaxy_detail}, mean_mix={float(np.mean(mix)):.5f}, "
            f"max_mix={float(np.max(mix)):.5f}, sky_floor={sky_floor:.5f}, sky_cut={sky_cut:.5f}, "
            f"background_mask_mean={float(np.mean(background_mask)):.5f}"
        )
    return _to_uint16(blended)


def blend_galaxy_deconvolution_detail(
    base_image: np.ndarray,
    deconvolved_image: np.ndarray,
    log: LogCallback | None = None,
) -> np.ndarray:
    base = _to_float01(base_image)
    decon = _to_float01(deconvolved_image)
    if base.shape != decon.shape:
        if log:
            log(f"Skipped galaxy deconvolution blend: shape mismatch {base.shape} vs {decon.shape}")
        return base_image

    base_lum = _luminance(base).astype(np.float32)
    decon_lum = _luminance(decon).astype(np.float32)

    broad = cv2.GaussianBlur(base_lum, (0, 0), 24.0)
    compact = cv2.GaussianBlur(base_lum, (0, 0), 7.0)
    extended = compact * 0.58 + broad * 0.42
    low = float(np.percentile(extended, 72.0))
    high = float(np.percentile(extended, 99.65))
    galaxy_mask = np.clip((extended - low) / max(1e-6, high - low), 0.0, 1.0) ** 0.86
    galaxy_mask = cv2.GaussianBlur(galaxy_mask.astype(np.float32), (0, 0), 11.0)

    star_core = np.clip(
        (base_lum - np.percentile(base_lum, 97.7))
        / max(1e-6, np.percentile(base_lum, 99.995) - np.percentile(base_lum, 97.7)),
        0.0,
        1.0,
    ) ** 1.45
    star_core = cv2.dilate(star_core.astype(np.float32), np.ones((3, 3), dtype=np.uint8), iterations=1)
    star_core = cv2.GaussianBlur(star_core.astype(np.float32), (0, 0), 1.25)
    blend_mask = np.clip(galaxy_mask * (1.0 - star_core * 0.96), 0.0, 0.46)

    decon_detail = decon_lum - cv2.GaussianBlur(decon_lum, (0, 0), 1.6)
    base_detail = base_lum - cv2.GaussianBlur(base_lum, (0, 0), 1.6)
    extra_detail = np.clip(decon_detail - base_detail * 0.35, -0.035, 0.055)
    detail_lum = np.clip(base_lum + extra_detail * blend_mask * 0.78, 0.0, 1.0)
    output = np.clip(base * (detail_lum / np.maximum(base_lum, 1e-5))[..., None], 0.0, 1.0)

    if log:
        log(
            "Blended Siril deconvolution as galaxy-only detail layer: "
            f"galaxy_mask_mean={float(np.mean(galaxy_mask)):.5f}, "
            f"star_reject_mean={float(np.mean(star_core)):.5f}, "
            f"blend_mask_mean={float(np.mean(blend_mask)):.5f}, "
            f"blend_mask_max={float(np.max(blend_mask)):.5f}"
        )
    return _to_uint16(output)


def apply_small_galaxy_darkroom_look(image: np.ndarray, log: LogCallback | None = None) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return arr

    rgb = _to_float01(arr)
    lum = _luminance(rgb).astype(np.float32)
    height, width = lum.shape

    extended_small = cv2.GaussianBlur(lum, (0, 0), 6.0)
    extended_large = cv2.GaussianBlur(lum, (0, 0), 20.0)
    extended = extended_small * 0.72 + extended_large * 0.28
    galaxy_low = float(np.percentile(extended, 92.0))
    galaxy_high = float(np.percentile(extended, 99.85))
    galaxy_core_mask = np.clip(
        (extended - galaxy_low) / max(1e-6, galaxy_high - galaxy_low),
        0.0,
        1.0,
    ) ** 0.62
    galaxy_halo_low = float(np.percentile(extended, 82.0))
    galaxy_halo_high = float(np.percentile(extended, 99.35))
    galaxy_halo_mask = np.clip(
        (extended - galaxy_halo_low) / max(1e-6, galaxy_halo_high - galaxy_halo_low),
        0.0,
        1.0,
    ) ** 1.15
    galaxy_mask = np.maximum(galaxy_core_mask, galaxy_halo_mask * 0.46)
    galaxy_mask = cv2.GaussianBlur(galaxy_mask.astype(np.float32), (0, 0), 14.0)
    galaxy_mask = np.clip(galaxy_mask, 0.0, 0.82)

    core_mask = np.clip(
        (extended - np.percentile(extended, 97.1))
        / max(1e-6, np.percentile(extended, 99.96) - np.percentile(extended, 97.1)),
        0.0,
        1.0,
    ) ** 0.72
    core_mask = cv2.GaussianBlur(core_mask.astype(np.float32), (0, 0), 2.4)

    star_mask = np.clip(
        (lum - np.percentile(lum, 98.85))
        / max(1e-6, np.percentile(lum, 99.997) - np.percentile(lum, 98.85)),
        0.0,
        1.0,
    ) ** 1.80
    star_mask = cv2.morphologyEx(
        star_mask.astype(np.float32),
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    star_mask = cv2.GaussianBlur(star_mask.astype(np.float32), (0, 0), 0.55)

    yy, xx = np.mgrid[0:height, 0:width]
    edge_distance = np.minimum.reduce([xx, yy, width - 1 - xx, height - 1 - yy]).astype(np.float32)
    edge_mask = np.clip((min(height, width) * 0.12 - edge_distance) / max(1.0, min(height, width) * 0.12), 0.0, 1.0)
    edge_mask = cv2.GaussianBlur(edge_mask.astype(np.float32), (0, 0), 18.0)
    edge_star_reject = np.clip(edge_mask * (1.0 - galaxy_mask * 0.85), 0.0, 1.0)

    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoisingColored(rgb8, None, 30, 42, 7, 35).astype(np.float32) / 255.0
    base = (
        rgb * 0.42
        + cv2.GaussianBlur(denoised, (0, 0), 8.0) * 0.28
        + cv2.GaussianBlur(denoised, (0, 0), 30.0) * 0.30
    )
    base_lum = _luminance(base)
    base = base_lum[..., None] + (base - base_lum[..., None]) * 0.24
    base = np.clip(base * (0.40 - edge_mask[..., None] * 0.08) + 0.0032, 0.0, 1.0)

    galaxy_rgb = rgb.copy()
    galaxy_lum = _luminance(galaxy_rgb)
    black = float(np.percentile(galaxy_lum, 9.0))
    galaxy_rgb = np.clip((galaxy_rgb - black * 0.38) / max(1e-6, 1.0 - black * 0.38), 0.0, 1.0)
    galaxy_lum = _luminance(galaxy_rgb)
    blur_1 = cv2.GaussianBlur(galaxy_lum, (0, 0), 0.9)
    blur_3 = cv2.GaussianBlur(galaxy_lum, (0, 0), 3.0)
    blur_10 = cv2.GaussianBlur(galaxy_lum, (0, 0), 10.0)
    fine_detail = galaxy_lum - blur_1
    small_detail = blur_1 - blur_3
    mid_detail = blur_3 - blur_10
    sharpened_lum = np.clip(galaxy_lum + fine_detail * 0.65 + small_detail * 1.15 + mid_detail * 0.62, 0.0, 1.0)
    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8))
    clahe_lum = clahe.apply(np.clip(galaxy_lum * 65535.0, 0, 65535).astype(np.uint16)).astype(np.float32) / 65535.0
    sharpened_lum = np.clip(sharpened_lum * (1.0 - core_mask * 0.35) + clahe_lum * (core_mask * 0.35), 0.0, 1.0)
    galaxy_rgb = np.clip(galaxy_rgb * (sharpened_lum / np.maximum(galaxy_lum, 1e-5))[..., None], 0.0, 1.0)
    galaxy_lum = _luminance(galaxy_rgb)
    warm = np.array([1.18, 0.92, 0.86], dtype=np.float32).reshape(1, 1, 3)
    galaxy_rgb = np.clip(
        galaxy_rgb * (1.0 - core_mask[..., None] * 0.25)
        + galaxy_lum[..., None] * warm * (core_mask[..., None] * 0.25),
        0.0,
        1.0,
    )
    halo_smooth = cv2.bilateralFilter(
        np.clip(galaxy_rgb * 255.0, 0, 255).astype(np.uint8),
        d=0,
        sigmaColor=18,
        sigmaSpace=5,
    ).astype(np.float32) / 255.0
    halo_mix = np.clip((galaxy_mask - core_mask) * 0.25, 0.0, 0.25)[..., None]
    galaxy_rgb = np.clip(galaxy_rgb * (1.0 - halo_mix) + halo_smooth * halo_mix, 0.0, 1.0)

    star_lum = np.clip(lum / max(float(np.percentile(lum, 99.98)), 1e-6), 0.0, 1.0) ** 1.20
    star_color = np.clip(rgb / np.maximum(lum[..., None], 1e-5), 0.65, 1.45)
    star_warm = np.array([1.12, 0.96, 0.86], dtype=np.float32).reshape(1, 1, 3)
    star_layer = np.clip(star_lum[..., None] * star_color * star_warm * 0.56, 0.0, 1.0)

    galaxy_weight = cv2.GaussianBlur(galaxy_mask.astype(np.float32), (0, 0), 8.0)
    galaxy_weight = np.clip(galaxy_weight * 0.64, 0.0, 0.58)[..., None]
    output = np.clip(base * (1.0 - galaxy_weight) + galaxy_rgb * galaxy_weight, 0.0, 1.0)
    field_detail_weight = np.clip(galaxy_mask * 0.18, 0.0, 0.14)[..., None]
    output = np.clip(output + (rgb - cv2.GaussianBlur(rgb, (0, 0), 2.2)) * field_detail_weight, 0.0, 1.0)
    star_mix = np.clip(star_mask[..., None] * (1.0 - galaxy_mask[..., None] * 0.45) * (1.0 - edge_star_reject[..., None] * 0.86), 0.0, 0.54)
    output = np.clip(output * (1.0 - star_mix) + star_layer * star_mix, 0.0, 1.0)

    final_lum = _luminance(output)
    white = float(np.percentile(final_lum, 99.975))
    output = np.clip(output / max(white, 1e-6), 0.0, 1.0) ** 1.10
    final_lum = _luminance(output)
    highlight_rolloff = np.clip((final_lum - 0.76) / 0.24, 0.0, 1.0)
    output = np.clip(output * (1.0 - highlight_rolloff[..., None] * 0.20), 0.0, 1.0)

    if log:
        log(
            "Applied small-galaxy darkroom finish: "
            f"galaxy_mask_mean={float(np.mean(galaxy_mask)):.5f}, "
            f"star_mask_mean={float(np.mean(star_mask)):.5f}, "
            f"edge_mask_mean={float(np.mean(edge_mask)):.5f}, "
            f"black={black:.5f}, white={white:.5f}"
        )
    return _to_uint16(output)


def apply_broadband_look(image: np.ndarray, log: LogCallback | None = None) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return arr

    rgb = _to_float01(arr)
    lum = _luminance(rgb)
    black = float(np.percentile(lum, 8.0))
    white = float(np.percentile(lum, 99.7))
    rgb = np.clip((rgb - black) / max(1e-6, white - black), 0.0, 1.0)

    rgb = np.clip(rgb, 0.0, 1.0) ** 1.08
    lum = _luminance(rgb)
    blurred = cv2.GaussianBlur(lum, (0, 0), 16)
    contrast_lum = np.clip(lum + (lum - blurred) * 0.035, 0.0, 1.0)
    rgb = np.clip(rgb * (contrast_lum / np.maximum(lum, 1e-5))[..., None], 0.0, 1.0)

    lum = _luminance(rgb)
    star_for_balance = np.clip(
        (lum - np.percentile(lum, 96.8))
        / max(1e-6, np.percentile(lum, 99.96) - np.percentile(lum, 96.8)),
        0.0,
        1.0,
    ) ** 1.65
    star_for_balance = cv2.GaussianBlur(star_for_balance.astype(np.float32), (0, 0), 0.8)
    dark_mask = (lum < np.percentile(lum, 48.0)) & (star_for_balance < 0.08)
    if int(np.count_nonzero(dark_mask)) >= 128:
        sky = np.median(rgb[dark_mask], axis=0)
        neutral = float(np.mean(sky))
        gains = np.clip(neutral / np.maximum(sky, 1e-4), 0.90, 1.10)
        rgb = np.clip(rgb * gains.reshape(1, 1, 3), 0.0, 1.0)
    else:
        gains = np.ones(3, dtype=np.float32)

    lum = _luminance(rgb)
    extended = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 18.0) * 0.62 + cv2.GaussianBlur(
        lum.astype(np.float32), (0, 0), 42.0
    ) * 0.38
    galaxy_mask = np.clip(
        (extended - np.percentile(extended, 72.0))
        / max(1e-6, np.percentile(extended, 99.55) - np.percentile(extended, 72.0)),
        0.0,
        1.0,
    ) ** 0.78
    galaxy_mask = cv2.GaussianBlur(galaxy_mask.astype(np.float32), (0, 0), 6.0)
    galaxy_core_mask = np.clip(
        (extended - np.percentile(extended, 92.5))
        / max(1e-6, np.percentile(extended, 99.88) - np.percentile(extended, 92.5)),
        0.0,
        1.0,
    ) ** 1.12
    galaxy_core_mask = cv2.GaussianBlur(galaxy_core_mask.astype(np.float32), (0, 0), 4.5)
    galaxy_dust_mask = np.clip(galaxy_mask - galaxy_core_mask * 0.45, 0.0, 1.0)

    star_color_mask = np.clip(
        (lum - np.percentile(lum, 97.2))
        / max(1e-6, np.percentile(lum, 99.985) - np.percentile(lum, 97.2)),
        0.0,
        1.0,
    ) ** 1.55
    star_color_mask = cv2.GaussianBlur(star_color_mask.astype(np.float32), (0, 0), 0.75)

    saturation = 0.58 + galaxy_mask[..., None] * 0.46 + star_color_mask[..., None] * 0.22
    rgb = np.clip(lum[..., None] + (rgb - lum[..., None]) * np.clip(saturation, 0.52, 1.16), 0.0, 1.0)

    lum = _luminance(rgb)
    warm_dust = np.array([1.12, 1.01, 0.86], dtype=np.float32).reshape(1, 1, 3)
    warm_core = np.array([1.24, 1.06, 0.80], dtype=np.float32).reshape(1, 1, 3)
    dust_tinted = np.clip(lum[..., None] * warm_dust, 0.0, 1.0)
    rgb = np.clip(rgb * (1.0 - galaxy_dust_mask[..., None] * 0.22) + dust_tinted * (galaxy_dust_mask[..., None] * 0.22), 0.0, 1.0)
    core_tinted = np.clip(lum[..., None] * warm_core, 0.0, 1.0)
    rgb = np.clip(rgb * (1.0 - galaxy_core_mask[..., None] * 0.34) + core_tinted * (galaxy_core_mask[..., None] * 0.34), 0.0, 1.0)

    core = np.clip(
        (lum - np.percentile(lum, 90.0))
        / max(1e-6, np.percentile(lum, 99.7) - np.percentile(lum, 90.0)),
        0.0,
        1.0,
    ) ** 1.2
    warm = np.array([1.10, 1.02, 0.88], dtype=np.float32).reshape(1, 1, 3)
    rgb = np.clip(rgb * (1.0 - core[..., None] * 0.24) + lum[..., None] * warm * (core[..., None] * 0.24), 0.0, 1.0)
    rgb = _suppress_green_excess(rgb, strength=0.70)
    rgb = _smooth_broadband_grain(rgb, log)

    lum = _luminance(rgb)
    dark_lane_base = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 4.5)
    dark_lane_fine = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 0.9)
    dark_lane = np.clip(dark_lane_base - dark_lane_fine, 0.0, None)
    dark_lane_high = float(np.percentile(dark_lane, 99.35))
    if dark_lane_high > 1e-6:
        dark_lane = np.clip(dark_lane / dark_lane_high, 0.0, 1.0)
    texture_mask = np.clip(dark_lane * galaxy_mask * (1.0 - galaxy_core_mask * 0.35) * (1.0 - star_color_mask * 0.82), 0.0, 0.85)
    texture_mask = cv2.GaussianBlur(texture_mask.astype(np.float32), (0, 0), 0.75)
    rgb = np.clip(rgb * (1.0 - texture_mask[..., None] * 0.42), 0.0, 1.0)
    texture_warmth = np.array([1.13, 0.98, 0.84], dtype=np.float32).reshape(1, 1, 3)
    rgb = np.clip(rgb * (1.0 - texture_mask[..., None] * 0.22) + lum[..., None] * texture_warmth * (texture_mask[..., None] * 0.22), 0.0, 1.0)

    lum = _luminance(rgb)
    galaxy_detail_mask = np.clip(galaxy_mask * (1.0 - star_color_mask * 0.88), 0.0, 0.72)
    micro_detail = lum - cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 1.15)
    rgb = np.clip(rgb * ((lum + micro_detail * galaxy_detail_mask * 0.10) / np.maximum(lum, 1e-5))[..., None], 0.0, 1.0)

    lum = _luminance(rgb)
    star = np.clip(
        (lum - np.percentile(lum, 97.5))
        / max(1e-6, np.percentile(lum, 99.98) - np.percentile(lum, 97.5)),
        0.0,
        1.0,
    ) ** 1.7
    neutral_star = lum[..., None] + np.clip(rgb - lum[..., None], -0.06, 0.06)
    rgb = rgb * (1.0 - star[..., None] * 0.18) + neutral_star * (star[..., None] * 0.18)

    lum = _luminance(rgb)
    star_protect = np.clip(
        (lum - np.percentile(lum, 94.5))
        / max(1e-6, np.percentile(lum, 99.92) - np.percentile(lum, 94.5)),
        0.0,
        1.0,
    ) ** 1.45
    star_protect = cv2.GaussianBlur(star_protect.astype(np.float32), (0, 0), 1.1)
    small_scale = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 2.8)
    broad_scale = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 20.0)
    reflection_detail = np.clip(small_scale - broad_scale * 0.84, 0.0, None)
    detail_high = float(np.percentile(reflection_detail, 99.35))
    if detail_high > 1e-6:
        reflection_detail = np.clip(reflection_detail / detail_high, 0.0, 1.0)
    broad_reflection = np.clip(
        (broad_scale - np.percentile(broad_scale, 50.0))
        / max(1e-6, np.percentile(broad_scale, 98.4) - np.percentile(broad_scale, 50.0)),
        0.0,
        1.0,
    ) ** 0.86
    reflection_mask = np.maximum(reflection_detail, broad_reflection * 0.40)
    reflection_signal = np.clip(
        (lum - np.percentile(lum, 22.0))
        / max(1e-6, np.percentile(lum, 97.0) - np.percentile(lum, 22.0)),
        0.0,
        1.0,
    ) ** 0.75
    reflection_mask = np.clip((reflection_mask ** 0.62) * reflection_signal * (1.0 - star_protect * 0.92), 0.0, 1.0)
    reflection_mask = cv2.GaussianBlur(reflection_mask.astype(np.float32), (0, 0), 1.7)
    lifted = np.clip(rgb + (1.0 - rgb) * reflection_mask[..., None] * 0.16, 0.0, 1.0)
    lifted_lum = _luminance(lifted)
    rgb = np.clip(lifted_lum[..., None] + (lifted - lifted_lum[..., None]) * (1.0 + reflection_mask[..., None] * 0.14), 0.0, 1.0)

    lum = _luminance(rgb)
    final_black = float(np.percentile(lum, 14.0))
    rgb = np.clip((rgb - final_black * 0.62) / max(1e-6, 1.0 - final_black * 0.62), 0.0, 1.0)
    lum = _luminance(rgb)
    sky_warmth_mask = np.clip(
        (np.percentile(lum, 54.0) - lum) / max(1e-6, np.percentile(lum, 54.0) - np.percentile(lum, 7.0)),
        0.0,
        1.0,
    )
    sky_warmth_mask = np.clip(sky_warmth_mask * (1.0 - galaxy_mask * 0.72) * (1.0 - star_color_mask * 0.92), 0.0, 0.42)
    sky_warm = np.array([1.16, 0.94, 1.02], dtype=np.float32).reshape(1, 1, 3)
    rgb = np.clip(rgb * (1.0 - sky_warmth_mask[..., None] * 0.10) + lum[..., None] * sky_warm * (sky_warmth_mask[..., None] * 0.10), 0.0, 1.0)
    if log:
        final_lum = _luminance(rgb)
        log(
            "Applied DeepSky broadband look: "
            f"black={black:.5f}, white={white:.5f}, final_black={final_black:.5f}, "
            f"sky_gains={gains[0]:.3f}, {gains[1]:.3f}, {gains[2]:.3f}, "
            f"median_luminance={np.median(final_lum):.5f}, chroma_p95={chroma_percentile(rgb, 95.0):.5f}, "
            f"galaxy_mask_mean={float(np.mean(galaxy_mask)):.5f}, "
            f"galaxy_core_mean={float(np.mean(galaxy_core_mask)):.5f}, "
            f"texture_mask_mean={float(np.mean(texture_mask)):.5f}, "
            f"reflection_mask_mean={float(np.mean(reflection_mask)):.5f}"
        )
    return _to_uint16(rgb)


def apply_prestretched_broadband_look(image: np.ndarray, log: LogCallback | None = None) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return arr

    rgb = _to_float01(arr)
    lum = _luminance(rgb)

    # Pre-stretched data already has useful tone mapping. Keep the histogram mostly intact.
    black = float(np.percentile(lum, 1.5))
    white = float(np.percentile(lum, 99.85))
    rgb = np.clip((rgb - black * 0.55) / max(1e-6, white - black * 0.55), 0.0, 1.0)

    lum = _luminance(rgb)
    extended = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 10.0)
    galaxy_mask = np.clip(
        (extended - np.percentile(extended, 52.0))
        / max(1e-6, np.percentile(extended, 98.8) - np.percentile(extended, 52.0)),
        0.0,
        1.0,
    ) ** 0.55
    star_mask = np.clip(
        (lum - np.percentile(lum, 97.2))
        / max(1e-6, np.percentile(lum, 99.98) - np.percentile(lum, 97.2)),
        0.0,
        1.0,
    ) ** 1.6
    protect = cv2.GaussianBlur(np.maximum(galaxy_mask, star_mask).astype(np.float32), (0, 0), 2.2)

    sky_mask = np.clip(1.0 - protect, 0.0, 1.0)
    sky_mask *= np.clip(
        (np.percentile(lum, 72.0) - lum)
        / max(1e-6, np.percentile(lum, 72.0) - np.percentile(lum, 1.0)),
        0.0,
        1.0,
    )
    sky_mask = cv2.GaussianBlur(sky_mask.astype(np.float32), (0, 0), 4.0)

    background_pixels = (sky_mask > 0.55) & (lum < np.percentile(lum, 68.0))
    gains = np.ones(3, dtype=np.float32)
    if int(np.count_nonzero(background_pixels)) >= 512:
        sky = np.median(rgb[background_pixels], axis=0)
        neutral = float(np.mean(sky))
        gains = np.clip(neutral / np.maximum(sky, 1e-4), 0.78, 1.35)
        balanced = np.clip(rgb * gains.reshape(1, 1, 3), 0.0, 1.0)
        rgb = np.clip(rgb * (1.0 - sky_mask[..., None] * 0.72) + balanced * (sky_mask[..., None] * 0.72), 0.0, 1.0)

    rgb = _suppress_green_excess(rgb, strength=0.78)

    lum_before_green = _luminance(rgb)
    green_target = 0.52 * rgb[..., 0] + 0.48 * rgb[..., 2]
    signal_green_excess = np.maximum(0.0, rgb[..., 1] - green_target)
    signal_mix = np.clip((galaxy_mask * 0.76 + sky_mask * 0.34) * (1.0 - star_mask * 0.60), 0.0, 0.82)
    rgb[..., 1] = np.clip(rgb[..., 1] - signal_green_excess * signal_mix, 0.0, 1.0)
    lum_after_green = _luminance(rgb)
    rgb = np.clip(rgb * (lum_before_green / np.maximum(lum_after_green, 1e-5))[..., None], 0.0, 1.0)

    lum = _luminance(rgb)
    galaxy_core = np.clip(
        (lum - np.percentile(lum, 94.0))
        / max(1e-6, np.percentile(lum, 99.82) - np.percentile(lum, 94.0)),
        0.0,
        1.0,
    ) ** 1.15
    galaxy_core = cv2.GaussianBlur((galaxy_core * galaxy_mask * (1.0 - star_mask * 0.75)).astype(np.float32), (0, 0), 1.35)
    core_white = lum[..., None] * np.array([1.04, 1.02, 0.96], dtype=np.float32).reshape(1, 1, 3)
    core_mix = np.clip(galaxy_core[..., None] * 0.58, 0.0, 0.62)
    rgb = np.clip(rgb * (1.0 - core_mix) + core_white * core_mix, 0.0, 1.0)

    # Smooth color speckle mostly in empty sky; preserve galaxy luminance and arms.
    lab = cv2.cvtColor(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    a_smooth = cv2.bilateralFilter(a_channel, d=7, sigmaColor=18, sigmaSpace=12)
    b_smooth = cv2.bilateralFilter(b_channel, d=7, sigmaColor=18, sigmaSpace=12)
    chroma_smoothed = cv2.cvtColor(cv2.merge([l_channel, a_smooth, b_smooth]), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    chroma_mix = np.clip(sky_mask[..., None] * 0.65, 0.0, 0.72)
    rgb = np.clip(rgb * (1.0 - chroma_mix) + chroma_smoothed * chroma_mix, 0.0, 1.0)

    lum = _luminance(rgb)
    sky_floor = float(np.percentile(lum[background_pixels], 24.0)) if int(np.count_nonzero(background_pixels)) >= 512 else float(np.percentile(lum, 5.0))
    darkened = np.clip((rgb - sky_floor * 0.42) / max(1e-6, 1.0 - sky_floor * 0.42), 0.0, 1.0)
    black_mix = np.clip(sky_mask[..., None] * 0.55, 0.0, 0.62)
    rgb = np.clip(rgb * (1.0 - black_mix) + darkened * black_mix, 0.0, 1.0)

    lum = _luminance(rgb)
    final_core = np.clip(
        (lum - np.percentile(lum, 92.0))
        / max(1e-6, np.percentile(lum, 99.72) - np.percentile(lum, 92.0)),
        0.0,
        1.0,
    ) ** 0.85
    final_core = cv2.GaussianBlur((final_core * galaxy_mask * (1.0 - star_mask * 0.30)).astype(np.float32), (0, 0), 1.4)
    neutral_core = lum[..., None] * np.array([1.01, 1.005, 0.995], dtype=np.float32).reshape(1, 1, 3)
    final_core_mix = np.clip(final_core[..., None] * 0.94, 0.0, 0.92)
    rgb = np.clip(rgb * (1.0 - final_core_mix) + neutral_core * final_core_mix, 0.0, 1.0)

    if log:
        log(
            "Applied pre-stretched broadband look: "
            f"black={black:.5f}, white={white:.5f}, sky_floor={sky_floor:.5f}, "
            f"sky_pixels={int(np.count_nonzero(background_pixels))}, "
            f"sky_gains={gains[0]:.3f}, {gains[1]:.3f}, {gains[2]:.3f}, "
            f"protect_mean={float(np.mean(protect)):.5f}, sky_mask_mean={float(np.mean(sky_mask)):.5f}, "
            f"core_white_mix={float(np.max(final_core_mix)):.5f}"
        )
    return _to_uint16(rgb)


def apply_prestretched_nebula_rgb_reveal(image: np.ndarray, log: LogCallback | None = None) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return arr

    source = _to_float01(arr)
    channels = []
    lows: list[float] = []
    highs: list[float] = []
    for index in range(3):
        channel = source[..., index]
        low = float(np.percentile(channel, 0.45))
        high = float(np.percentile(channel, 99.65))
        if high <= low:
            high = low + 1e-6
        lows.append(low)
        highs.append(high)
        channels.append(np.clip((channel - low) / (high - low), 0.0, 1.0))
    rgb = np.stack(channels, axis=-1).astype(np.float32)

    lum = _luminance(rgb)
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    hsv = cv2.cvtColor(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0

    star_mask = np.clip(
        (lum - np.percentile(lum, 97.4))
        / max(1e-6, np.percentile(lum, 99.985) - np.percentile(lum, 97.4)),
        0.0,
        1.0,
    ) ** 1.55
    star_mask = cv2.GaussianBlur(star_mask.astype(np.float32), (0, 0), 0.9)

    extended_lum = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 8.0)
    signal = np.clip(
        (extended_lum - np.percentile(extended_lum, 18.0))
        / max(1e-6, np.percentile(extended_lum, 96.5) - np.percentile(extended_lum, 18.0)),
        0.0,
        1.0,
    ) ** 0.80
    color_signal = np.clip(
        (chroma - np.percentile(chroma, 42.0))
        / max(1e-6, np.percentile(chroma, 98.5) - np.percentile(chroma, 42.0)),
        0.0,
        1.0,
    ) ** 0.72
    nebula_mask = np.clip(signal * (0.40 + 0.60 * color_signal) * (1.0 - star_mask * 0.88), 0.0, 1.0)
    nebula_mask = cv2.GaussianBlur(nebula_mask.astype(np.float32), (0, 0), 2.2)

    sky_mask = (
        (lum < np.percentile(lum, 48.0))
        & (saturation < np.percentile(saturation, 76.0))
        & (star_mask < 0.12)
        & (nebula_mask < 0.22)
    )
    gains = np.ones(3, dtype=np.float32)
    if int(np.count_nonzero(sky_mask)) >= 512:
        sky = np.median(rgb[sky_mask], axis=0)
        neutral = float(np.mean(sky))
        gains = np.clip(neutral / np.maximum(sky, 1e-4), 0.72, 1.38).astype(np.float32)
        balanced = np.clip(rgb * gains.reshape(1, 1, 3), 0.0, 1.0)
        sky_blend = np.clip(
            ((np.percentile(lum, 62.0) - lum) / max(1e-6, np.percentile(lum, 62.0) - np.percentile(lum, 1.0)))
            * (1.0 - star_mask)
            * (1.0 - nebula_mask * 0.72),
            0.0,
            1.0,
        )
        sky_blend = cv2.GaussianBlur(sky_blend.astype(np.float32), (0, 0), 3.0)
        rgb = np.clip(rgb * (1.0 - sky_blend[..., None] * 0.82) + balanced * (sky_blend[..., None] * 0.82), 0.0, 1.0)

    rgb = _suppress_green_excess(rgb, strength=0.62)

    lum = _luminance(rgb)
    shadow = np.clip(
        (np.percentile(lum, 32.0) - lum)
        / max(1e-6, np.percentile(lum, 32.0) - np.percentile(lum, 2.0)),
        0.0,
        1.0,
    )
    rgb = np.clip(rgb * (1.0 - shadow[..., None] * 0.28), 0.0, 1.0)

    lum = _luminance(rgb)
    chroma_boost = 1.0 + nebula_mask[..., None] * 1.70 + color_signal[..., None] * (1.0 - star_mask[..., None]) * 0.42
    rgb = np.clip(lum[..., None] + (rgb - lum[..., None]) * chroma_boost, 0.0, 1.0)

    lum = _luminance(rgb)
    bright_nebula = np.clip(
        (lum - np.percentile(lum, 58.0))
        / max(1e-6, np.percentile(lum, 98.2) - np.percentile(lum, 58.0)),
        0.0,
        1.0,
    ) ** 0.9
    warm_white = lum[..., None] * np.array([1.08, 1.00, 0.90], dtype=np.float32).reshape(1, 1, 3)
    white_mix = np.clip(nebula_mask[..., None] * bright_nebula[..., None] * 0.30, 0.0, 0.34)
    rgb = np.clip(rgb * (1.0 - white_mix) + warm_white * white_mix, 0.0, 1.0)

    # Reduce color speckle in the sky only; keep nebula RGB and star color alive.
    lab = cv2.cvtColor(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    a_smooth = cv2.bilateralFilter(a_channel, d=7, sigmaColor=16, sigmaSpace=10)
    b_smooth = cv2.bilateralFilter(b_channel, d=7, sigmaColor=16, sigmaSpace=10)
    chroma_smoothed = cv2.cvtColor(cv2.merge([l_channel, a_smooth, b_smooth]), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0
    sky_float = np.clip(
        ((np.percentile(lum, 55.0) - lum) / max(1e-6, np.percentile(lum, 55.0) - np.percentile(lum, 1.0)))
        * (1.0 - nebula_mask)
        * (1.0 - star_mask),
        0.0,
        1.0,
    )
    sky_float = cv2.GaussianBlur(sky_float.astype(np.float32), (0, 0), 2.4)
    rgb = np.clip(rgb * (1.0 - sky_float[..., None] * 0.44) + chroma_smoothed * (sky_float[..., None] * 0.44), 0.0, 1.0)

    lum = _luminance(rgb)
    sky_floor = float(np.percentile(lum[sky_mask], 32.0)) if int(np.count_nonzero(sky_mask)) >= 512 else float(np.percentile(lum, 6.0))
    darkened = np.clip((rgb - sky_floor * 0.46) / max(1e-6, 1.0 - sky_floor * 0.46), 0.0, 1.0)
    black_mix = np.clip(sky_float[..., None] * 0.42, 0.0, 0.48)
    rgb = np.clip(rgb * (1.0 - black_mix) + darkened * black_mix, 0.0, 1.0)

    lum = _luminance(rgb)
    neutral_star = lum[..., None] + np.clip(rgb - lum[..., None], -0.075, 0.075)
    rgb = np.clip(rgb * (1.0 - star_mask[..., None] * 0.26) + neutral_star * (star_mask[..., None] * 0.26), 0.0, 1.0)

    if log:
        log(
            "Applied pre-stretched nebula RGB reveal: "
            f"channel_lows={lows[0]:.5f}, {lows[1]:.5f}, {lows[2]:.5f}, "
            f"channel_highs={highs[0]:.5f}, {highs[1]:.5f}, {highs[2]:.5f}, sky_floor={sky_floor:.5f}, "
            f"sky_pixels={int(np.count_nonzero(sky_mask))}, "
            f"sky_gains={gains[0]:.3f}, {gains[1]:.3f}, {gains[2]:.3f}, "
            f"nebula_mask_mean={float(np.mean(nebula_mask)):.5f}, chroma_p95={chroma_percentile(rgb, 95.0):.5f}"
        )
    return _to_uint16(rgb)


def apply_goal_look(image: np.ndarray, log: LogCallback | None = None, stretch: bool = True) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return arr

    rgb = _to_float01(arr)
    lum = _luminance(rgb)
    if stretch:
        black = float(np.percentile(lum, 15.0))
        white = float(np.percentile(lum, 99.7))
        rgb = np.clip((rgb - black) / max(1e-6, white - black), 0.0, 1.0)
    else:
        black = 0.0
        white = 1.0

    lum = _luminance(rgb)
    star_mask = np.clip(
        (lum - np.percentile(lum, 96.8))
        / max(1e-6, np.percentile(lum, 99.96) - np.percentile(lum, 96.8)),
        0.0,
        1.0,
    ) ** 1.8
    smoothed = cv2.bilateralFilter((rgb * 255.0).astype(np.uint8), 7, 45, 7).astype(np.float32) / 255.0
    rgb = rgb * (0.65 + 0.35 * star_mask[..., None]) + smoothed * (0.35 * (1.0 - star_mask[..., None]))

    if stretch:
        rgb = np.clip(rgb, 0.0, 1.0) ** 0.50
    lum = _luminance(rgb)
    blurred = cv2.GaussianBlur(lum, (0, 0), 20)
    contrast_lum = np.clip(lum + (lum - blurred) * (0.20 if stretch else 0.08), 0.0, 1.0)
    rgb = np.clip(rgb * (contrast_lum / np.maximum(lum, 1e-5))[..., None], 0.0, 1.0)

    lum = _luminance(rgb)
    rgb = np.clip(lum[..., None] + (rgb - lum[..., None]) * (1.9 if stretch else 1.25), 0.0, 1.0)
    reflection_bias = reflection_nebula_bias(rgb)

    lum = _luminance(rgb)
    red_excess = np.clip(rgb[..., 0] - 0.52 * rgb[..., 1] - 0.48 * rgb[..., 2], 0.0, 1.0)
    blue_excess = np.clip(np.maximum(rgb[..., 1], rgb[..., 2]) - rgb[..., 0] * 0.82, 0.0, 1.0)
    red_low = float(np.percentile(red_excess, 42.0))
    red_high = float(np.percentile(red_excess, 99.5))
    red_mask = np.clip((red_excess - red_low) / max(1e-6, red_high - red_low), 0.0, 1.0) ** 0.7
    signal = np.clip(
        (lum - np.percentile(lum, 20.0))
        / max(1e-6, np.percentile(lum, 98.4) - np.percentile(lum, 20.0)),
        0.0,
        1.0,
    ) ** 0.85
    star_core = np.clip(
        (lum - np.percentile(lum, 99.1))
        / max(1e-6, np.percentile(lum, 99.98) - np.percentile(lum, 99.1)),
        0.0,
        1.0,
    )
    emission = red_mask * signal * (1.0 - star_core)
    emission_strength = 1.0 if stretch else 0.55
    rgb[..., 0] += emission * 0.34 * emission_strength
    rgb[..., 1] += emission * 0.041 * emission_strength
    rgb[..., 2] -= emission * 0.044 * emission_strength

    lum = _luminance(rgb)
    broad_emission = cv2.GaussianBlur((red_mask * signal).astype(np.float32), (0, 0), 5)
    bright_nebula = np.clip(
        (lum - np.percentile(lum, 55.0))
        / max(1e-6, np.percentile(lum, 98.0) - np.percentile(lum, 55.0)),
        0.0,
        1.0,
    )
    white_nebula = np.clip(broad_emission * bright_nebula * (1.0 - star_core), 0.0, 1.0) ** 0.75
    warm_white = np.dstack(
        [
            np.ones_like(lum),
            np.ones_like(lum) * 0.90,
            np.ones_like(lum) * 0.76,
        ]
    )
    rgb = np.clip(rgb + white_nebula[..., None] * (0.58 if stretch else 0.22) * warm_white, 0.0, 1.0)
    lum = _luminance(rgb)
    cream = lum[..., None] * np.array([1.10, 1.02, 0.92], dtype=np.float32).reshape(1, 1, 3)
    white_mix = np.clip(white_nebula[..., None] * (0.55 if stretch else 0.22), 0.0, 0.65)
    rgb = np.clip(rgb * (1.0 - white_mix) + cream * white_mix, 0.0, 1.0)

    lum = _luminance(rgb)
    star = np.clip(
        (lum - np.percentile(lum, 97.2))
        / max(1e-6, np.percentile(lum, 99.98) - np.percentile(lum, 97.2)),
        0.0,
        1.0,
    ) ** 1.5
    star_lum = lum[..., None]
    neutral_star = star_lum + np.clip(rgb - star_lum, -0.08, 0.08)
    rgb = rgb * (1.0 - 0.45 * star[..., None]) + neutral_star * (0.45 * star[..., None])

    lum = _luminance(rgb)
    shadow = np.clip(
        (np.percentile(lum, 35.0) - lum)
        / max(1e-6, np.percentile(lum, 35.0) - np.percentile(lum, 3.0)),
        0.0,
        1.0,
    ) * (1.0 - red_mask)
    rgb[..., 1] += shadow * 0.006
    rgb[..., 2] += shadow * 0.012

    lum = _luminance(rgb)
    star_protect = np.clip(
        (lum - np.percentile(lum, 94.8))
        / max(1e-6, np.percentile(lum, 99.92) - np.percentile(lum, 94.8)),
        0.0,
        1.0,
    ) ** 1.35
    star_protect = cv2.GaussianBlur(star_protect.astype(np.float32), (0, 0), 1.1)
    small_scale = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 2.6)
    broad_scale = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 23.0)
    diffuse_detail = np.clip(small_scale - broad_scale * 0.82, 0.0, None)
    detail_high = float(np.percentile(diffuse_detail, 99.35))
    if detail_high > 1e-6:
        diffuse_detail = np.clip(diffuse_detail / detail_high, 0.0, 1.0)
    broad_dust = np.clip(
        (broad_scale - np.percentile(broad_scale, 52.0))
        / max(1e-6, np.percentile(broad_scale, 98.6) - np.percentile(broad_scale, 52.0)),
        0.0,
        1.0,
    ) ** 0.82
    diffuse_detail = np.maximum(diffuse_detail, broad_dust * (0.46 + 0.18 * reflection_bias))
    faint_signal = np.clip(
        (lum - np.percentile(lum, 18.0 if reflection_bias > 0.35 else 24.0))
        / max(1e-6, np.percentile(lum, 97.5) - np.percentile(lum, 18.0 if reflection_bias > 0.35 else 24.0)),
        0.0,
        1.0,
    ) ** (0.60 if reflection_bias > 0.35 else 0.72)
    faint_dust = np.clip(
        (diffuse_detail ** (0.50 if reflection_bias > 0.35 else 0.58))
        * faint_signal
        * (1.0 - star_protect * (0.90 if reflection_bias > 0.35 else 0.94)),
        0.0,
        1.0,
    )
    faint_dust = cv2.GaussianBlur(faint_dust.astype(np.float32), (0, 0), 1.35 if reflection_bias > 0.35 else 1.8)
    dust_strength = (0.34 if stretch else 0.42) + reflection_bias * (0.13 if stretch else 0.19)
    lifted = np.clip(rgb + (1.0 - rgb) * faint_dust[..., None] * dust_strength, 0.0, 1.0)
    dust_lum = _luminance(lifted)
    dust_contrast = np.clip(
        dust_lum
        + (dust_lum - cv2.GaussianBlur(dust_lum, (0, 0), 7.0)) * (0.14 + 0.18 * reflection_bias),
        0.0,
        1.0,
    )
    lifted = np.clip(lifted * (dust_contrast / np.maximum(dust_lum, 1e-5))[..., None], 0.0, 1.0)
    dust_lum = _luminance(lifted)
    if reflection_bias > 0.05:
        cool_filament = np.clip((diffuse_detail ** 0.72) * faint_signal * reflection_bias * (1.0 - broad_dust * 0.35), 0.0, 1.0)
        warm_dust = np.clip((broad_dust ** 0.82) * faint_signal * reflection_bias * (1.0 - star_protect * 0.95), 0.0, 1.0)
        cool_target = dust_lum[..., None] * np.array([0.90, 0.98, 1.13], dtype=np.float32).reshape(1, 1, 3)
        warm_target = dust_lum[..., None] * np.array([1.16, 1.05, 0.86], dtype=np.float32).reshape(1, 1, 3)
        lifted = np.clip(lifted * (1.0 - cool_filament[..., None] * 0.22) + cool_target * (cool_filament[..., None] * 0.22), 0.0, 1.0)
        lifted = np.clip(lifted * (1.0 - warm_dust[..., None] * 0.18) + warm_target * (warm_dust[..., None] * 0.18), 0.0, 1.0)
        dust_lum = _luminance(lifted)
    dust_chroma = 1.0 + faint_dust[..., None] * ((0.36 if stretch else 0.30) + reflection_bias * 0.16)
    rgb = np.clip(dust_lum[..., None] + (lifted - dust_lum[..., None]) * dust_chroma, 0.0, 1.0)

    emission_score = red_emission_dominance(rgb)
    if emission_score >= 3.0:
        lum = _luminance(rgb)
        emission_grade = cv2.GaussianBlur((red_mask * signal * (1.0 - star_core)).astype(np.float32), (0, 0), 3.2)
        emission_grade = np.clip(emission_grade, 0.0, 1.0) ** (0.58 if stretch else 0.48)
        nebula_body = np.maximum(emission_grade, np.clip(faint_dust * 1.15, 0.0, 1.0))
        nebula_body = np.maximum(nebula_body, np.clip(broad_dust * faint_signal * (1.0 - star_protect * 0.90), 0.0, 1.0))
        nebula_body = cv2.GaussianBlur(nebula_body.astype(np.float32), (0, 0), 1.1)
        nebula_body = np.clip(nebula_body, 0.0, 1.0) ** (0.72 if stretch else 0.58)
        nebula_body = np.clip(nebula_body * (1.0 - star_protect * 0.82), 0.0, 1.0)
        edge_emission = np.clip(diffuse_detail * (1.0 - red_mask * 0.55) * signal, 0.0, 1.0)
        edge_emission = cv2.GaussianBlur(edge_emission.astype(np.float32), (0, 0), 1.6)
        cool_emission = cv2.GaussianBlur((np.maximum(blue_excess, edge_emission * 0.92) * signal * (1.0 - star_core)).astype(np.float32), (0, 0), 2.2)
        cool_emission = np.clip(cool_emission, 0.0, 1.0) ** (0.75 if stretch else 0.62)
        mixed_emission = np.clip(np.minimum(red_mask, blue_excess) * signal * (1.0 - star_core), 0.0, 1.0)
        mixed_emission = cv2.GaussianBlur(mixed_emission.astype(np.float32), (0, 0), 1.8)
        mixed_emission = np.clip(mixed_emission, 0.0, 1.0)
        pink_halo = np.clip((broad_dust * faint_signal * (1.0 - cool_emission * 0.45)).astype(np.float32), 0.0, 1.0)
        orange_target = lum[..., None] * np.array([1.58, 0.66, 0.28], dtype=np.float32).reshape(1, 1, 3)
        cyan_target = lum[..., None] * np.array([0.62, 0.96, 1.32], dtype=np.float32).reshape(1, 1, 3)
        pink_target = lum[..., None] * np.array([1.12, 0.76, 1.08], dtype=np.float32).reshape(1, 1, 3)
        warm_mix = np.clip((nebula_body * (1.0 - cool_emission * 0.58))[..., None] * (0.22 if stretch else 0.52), 0.0, 0.60)
        cool_mix = np.clip((cool_emission * np.maximum(nebula_body, mixed_emission * 1.2))[..., None] * (0.24 if stretch else 0.52), 0.0, 0.54)
        pink_mix = np.clip((pink_halo * (1.0 - red_mask * 0.28))[..., None] * (0.10 if stretch else 0.18), 0.0, 0.18)
        rgb = np.clip(rgb * (1.0 - pink_mix) + pink_target * pink_mix, 0.0, 1.0)
        rgb = np.clip(rgb * (1.0 - warm_mix) + orange_target * warm_mix, 0.0, 1.0)
        rgb = np.clip(rgb * (1.0 - cool_mix) + cyan_target * cool_mix, 0.0, 1.0)
        rgb[..., 0] = np.clip(rgb[..., 0] + nebula_body * (0.03 if stretch else 0.10), 0.0, 1.0)
        rgb[..., 1] = np.clip(rgb[..., 1] + mixed_emission * (0.012 if stretch else 0.022), 0.0, 1.0)
        rgb[..., 2] = np.clip(rgb[..., 2] + cool_emission * (0.03 if stretch else 0.08) - nebula_body * (0.004 if stretch else 0.018), 0.0, 1.0)

        lum = _luminance(rgb)
        bright_emission = np.clip(
            (lum - np.percentile(lum, 60.0))
            / max(1e-6, np.percentile(lum, 99.35) - np.percentile(lum, 60.0)),
            0.0,
            1.0,
        ) ** 0.72
        peach_target = lum[..., None] * np.array([1.12, 0.96, 0.76], dtype=np.float32).reshape(1, 1, 3)
        peach_mix = np.clip(bright_emission[..., None] * nebula_body[..., None] * (0.28 if stretch else 0.44), 0.0, 0.48)
        rgb = np.clip(rgb * (1.0 - peach_mix) + peach_target * peach_mix, 0.0, 1.0)
        lum = _luminance(rgb)
        post_star = np.clip(
            (lum - np.percentile(lum, 96.8))
            / max(1e-6, np.percentile(lum, 99.98) - np.percentile(lum, 96.8)),
            0.0,
            1.0,
        ) ** 1.5
        post_neutral = lum[..., None] + np.clip(rgb - lum[..., None], -0.055, 0.055)
        rgb = np.clip(rgb * (1.0 - post_star[..., None] * 0.34) + post_neutral * (post_star[..., None] * 0.34), 0.0, 1.0)
        lum = _luminance(rgb)
        texture_mask = np.clip(nebula_body * (1.0 - post_star * 0.88), 0.0, 1.0)
        fine_detail = lum - cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 2.4)
        broad_detail = lum - cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 10.0)
        textured_lum = np.clip(
            lum + fine_detail * texture_mask * (0.20 if stretch else 0.36) + broad_detail * texture_mask * (0.08 if stretch else 0.16),
            0.0,
            1.0,
        )
        rgb = np.clip(rgb * (textured_lum / np.maximum(lum, 1e-5))[..., None], 0.0, 1.0)

    lum = _luminance(rgb)
    final_black = float(np.percentile(lum, 2.0 if stretch else 4.0))
    black_scale = 0.72 if stretch else (0.46 + 0.20 * reflection_bias)
    rgb = np.clip((rgb - final_black * black_scale) / max(1e-6, 1.0 - final_black * black_scale), 0.0, 1.0)
    if reflection_bias > 0.05:
        lum = _luminance(rgb)
        empty_sky = np.clip(
            (np.percentile(lum, 48.0) - lum)
            / max(1e-6, np.percentile(lum, 48.0) - np.percentile(lum, 2.0)),
            0.0,
            1.0,
        )
        empty_sky *= np.clip(1.0 - faint_dust * 1.35, 0.0, 1.0)
        rgb = np.clip(rgb * (1.0 - empty_sky[..., None] * (0.24 + 0.12 * reflection_bias)), 0.0, 1.0)

    lum = _luminance(rgb)
    sky_mask = lum < np.percentile(lum, 52.0)
    if int(np.count_nonzero(sky_mask)) >= 512:
        sky = np.median(rgb[sky_mask], axis=0)
        green_target = 0.56 * sky[0] + 0.44 * sky[2]
        if sky[1] > green_target * 1.18 + 0.006:
            low = float(np.percentile(lum, 4.0))
            high = float(np.percentile(lum, 78.0))
            sky_weight = np.clip((high - lum) / max(1e-6, high - low), 0.0, 1.0)
            star_weight = np.clip(
                (lum - np.percentile(lum, 96.8))
                / max(1e-6, np.percentile(lum, 99.96) - np.percentile(lum, 96.8)),
                0.0,
                1.0,
            )
            red_signal = np.clip(
                (rgb[..., 0] - np.percentile(rgb[..., 0], 58.0))
                / max(1e-6, np.percentile(rgb[..., 0], 99.4) - np.percentile(rgb[..., 0], 58.0)),
                0.0,
                1.0,
            )
            neutral_green = 0.58 * rgb[..., 0] + 0.42 * rgb[..., 2]
            excess_green = np.maximum(0.0, rgb[..., 1] - neutral_green)
            reduction = np.clip(sky_weight * (1.0 - star_weight * 0.8) * (1.0 - red_signal * 0.45), 0.0, 0.82)
            rgb[..., 1] = np.clip(rgb[..., 1] - excess_green * reduction, 0.0, 1.0)
            if log:
                after_sky = np.median(rgb[sky_mask], axis=0)
                log(
                    "Applied nebula background green guard: "
                    f"sky_before_RGB={sky[0]:.5f}, {sky[1]:.5f}, {sky[2]:.5f}, "
                    f"sky_after_RGB={after_sky[0]:.5f}, {after_sky[1]:.5f}, {after_sky[2]:.5f}"
                )
    if log:
        chroma_95 = chroma_percentile(rgb, 95.0)
        final_lum = _luminance(rgb)
        log(
            "Applied DeepSky target look: "
            f"stretch={stretch}, black={black:.5f}, white={white:.5f}, final_black={final_black:.5f}, "
            f"median_luminance={np.median(final_lum):.5f}, chroma_p95={chroma_95:.5f}, "
            f"faint_dust_mean={float(np.mean(faint_dust)):.5f}, reflection_bias={float(reflection_bias):.3f}, "
            f"emission_score={float(emission_score):.3f}"
        )
    return _to_uint16(rgb)


def apply_starless_nebula_detail(image: np.ndarray, log: LogCallback | None = None) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return arr

    rgb = _to_float01(arr)
    lum = _luminance(rgb)
    reflection_bias = reflection_nebula_bias(rgb)

    broad = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 22.0)
    broad_low = float(np.percentile(broad, 34.0))
    broad_high = float(np.percentile(broad, 99.0))
    dust_field = np.clip((broad - broad_low) / max(1e-6, broad_high - broad_low), 0.0, 1.0)
    dust_field = dust_field ** (0.50 if reflection_bias > 0.35 else 0.68)
    dust_field = cv2.GaussianBlur(dust_field.astype(np.float32), (0, 0), 2.4)

    fine = np.clip(lum - cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 7.0) * 0.90, 0.0, None)
    fine_high = float(np.percentile(fine, 99.25))
    if fine_high > 1e-6:
        fine = np.clip(fine / fine_high, 0.0, 1.0)
    filament = np.clip(fine * dust_field, 0.0, 1.0)
    filament = cv2.GaussianBlur(filament.astype(np.float32), (0, 0), 1.0)

    lift_strength = 0.24 + 0.58 * reflection_bias
    lifted = np.clip(rgb + (1.0 - rgb) * dust_field[..., None] * lift_strength, 0.0, 1.0)

    lifted_lum = _luminance(lifted)
    contrast = np.clip(
        lifted_lum
        + (lifted_lum - cv2.GaussianBlur(lifted_lum.astype(np.float32), (0, 0), 9.0))
        * (0.12 + 0.32 * reflection_bias)
        * np.clip(dust_field + filament, 0.0, 1.0),
        0.0,
        1.0,
    )
    lifted = np.clip(lifted * (contrast / np.maximum(lifted_lum, 1e-5))[..., None], 0.0, 1.0)

    lifted_lum = _luminance(lifted)
    if reflection_bias > 0.05:
        cool_target = lifted_lum[..., None] * np.array([0.90, 0.99, 1.12], dtype=np.float32).reshape(1, 1, 3)
        warm_target = lifted_lum[..., None] * np.array([1.16, 1.05, 0.86], dtype=np.float32).reshape(1, 1, 3)
        cool_mix = np.clip(filament[..., None] * 0.24 * reflection_bias, 0.0, 0.26)
        warm_mix = np.clip(dust_field[..., None] * 0.16 * reflection_bias, 0.0, 0.18)
        lifted = np.clip(lifted * (1.0 - cool_mix) + cool_target * cool_mix, 0.0, 1.0)
        lifted = np.clip(lifted * (1.0 - warm_mix) + warm_target * warm_mix, 0.0, 1.0)

    lifted_lum = _luminance(lifted)
    chroma_boost = 1.0 + dust_field[..., None] * (0.08 + 0.26 * reflection_bias)
    lifted = np.clip(lifted_lum[..., None] + (lifted - lifted_lum[..., None]) * chroma_boost, 0.0, 1.0)

    lifted_lum = _luminance(lifted)
    sky = np.clip(
        (np.percentile(lifted_lum, 46.0) - lifted_lum)
        / max(1e-6, np.percentile(lifted_lum, 46.0) - np.percentile(lifted_lum, 2.0)),
        0.0,
        1.0,
    )
    sky *= np.clip(1.0 - dust_field * 1.10, 0.0, 1.0)
    lifted = np.clip(lifted * (1.0 - sky[..., None] * (0.16 + 0.20 * reflection_bias)), 0.0, 1.0)

    if log:
        log(
            "Enhanced starless nebula detail: "
            f"dust_field_mean={float(np.mean(dust_field)):.5f}, "
            f"filament_mean={float(np.mean(filament)):.5f}, reflection_bias={float(reflection_bias):.3f}"
        )
    return _to_uint16(lifted)


def apply_cosmos_style_nebula_finish(image: np.ndarray, log: LogCallback | None = None) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return arr

    rgb = _to_float01(arr)
    lum = _luminance(rgb).astype(np.float32)

    blur5 = cv2.GaussianBlur(lum, (0, 0), 5.0)
    blur20 = cv2.GaussianBlur(lum, (0, 0), 20.0)
    high_frequency = lum - blur20
    high_low = float(np.percentile(high_frequency, 58.0))
    high_peak = float(np.percentile(high_frequency, 99.2))
    structure = np.clip((high_frequency - high_low) / max(1e-6, high_peak - high_low), 0.0, 1.0)

    edge = np.abs(lum - blur5)
    edge_low = float(np.percentile(edge, 62.0))
    edge_peak = float(np.percentile(edge, 99.1))
    edge = np.clip((edge - edge_low) / max(1e-6, edge_peak - edge_low), 0.0, 1.0)

    red_excess = np.clip((rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]) * 0.88) / 0.24, 0.0, 1.0)
    blue_excess = np.clip((rgb[..., 2] - rgb[..., 0] * 0.66) / 0.22, 0.0, 1.0)
    signal_low = float(np.percentile(lum, 28.0))
    signal_high = float(np.percentile(lum, 98.8))
    signal = np.clip((lum - signal_low) / max(1e-6, signal_high - signal_low), 0.0, 1.0)

    filament = np.clip(
        structure * 0.75 + edge * 0.75 + (red_excess + blue_excess) * 0.28 * signal,
        0.0,
        1.0,
    )
    filament = cv2.GaussianBlur((filament**0.72).astype(np.float32), (0, 0), 1.5)

    star_low = float(np.percentile(lum, 97.2))
    star_high = float(np.percentile(lum, 99.96))
    stars = np.clip((lum - star_low) / max(1e-6, star_high - star_low), 0.0, 1.0)
    stars = cv2.GaussianBlur(stars.astype(np.float32), (0, 0), 0.65)

    sky = np.clip(1.0 - np.clip(filament + stars * 0.95, 0.0, 1.0), 0.0, 1.0)
    base_lum = lum[..., None]
    neutral_sky = base_lum * np.array([1.02, 0.92, 0.86], dtype=np.float32).reshape(1, 1, 3)
    warm_filament = base_lum * np.array([1.22, 0.76, 0.54], dtype=np.float32).reshape(1, 1, 3)
    cool_filament = base_lum * np.array([0.58, 0.78, 1.18], dtype=np.float32).reshape(1, 1, 3)

    finished = np.clip(rgb * (1.0 - sky[..., None] * 0.82), 0.0, 1.0)
    finished = np.clip(finished - 0.11 * sky[..., None], 0.0, 1.0)

    finished_lum = _luminance(finished)
    saturation = 0.50 + filament[..., None] * 0.42 + stars[..., None] * 0.20
    finished = np.clip(
        finished_lum[..., None] + (finished - finished_lum[..., None]) * saturation,
        0.0,
        1.0,
    )

    finished = np.clip(
        finished * (1.0 - sky[..., None] * 0.55) + neutral_sky * (sky[..., None] * 0.55),
        0.0,
        1.0,
    )
    finished = np.clip(
        finished * (1.0 - red_excess[..., None] * filament[..., None] * 0.30)
        + warm_filament * (red_excess[..., None] * filament[..., None] * 0.30),
        0.0,
        1.0,
    )
    finished = np.clip(
        finished * (1.0 - blue_excess[..., None] * filament[..., None] * 0.18)
        + cool_filament * (blue_excess[..., None] * filament[..., None] * 0.18),
        0.0,
        1.0,
    )
    finished = np.clip(finished**1.28, 0.0, 1.0)

    if log:
        final_lum = _luminance(finished)
        sky_sample = final_lum[sky > 0.7]
        sky_mean = float(np.mean(sky_sample)) if sky_sample.size else float(np.mean(final_lum))
        log(
            "Applied Cosmos-style nebula finish: "
            f"sky_mean={sky_mean:.5f}, "
            f"filament_mean={float(np.mean(filament)):.5f}, "
            f"star_mean={float(np.mean(stars)):.5f}"
        )
    return _to_uint16(finished)


def _edge_support(shape: tuple[int, int], fraction: float = 0.035) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    distance = np.minimum.reduce([xx, yy, width - 1 - xx, height - 1 - yy]).astype(np.float32)
    feather = max(8.0, min(height, width) * fraction)
    return np.clip(distance / feather, 0.0, 1.0)


def _safe_percentile(values: np.ndarray, percentile: float, fallback: float = 0.0) -> float:
    if values.size == 0:
        return fallback
    return float(np.percentile(values, percentile))


def apply_pixinsight_style_nebula_finish(image: np.ndarray, log: LogCallback | None = None) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return arr

    rgb = _to_float01(arr)
    lum = _luminance(rgb).astype(np.float32)
    support = _edge_support(lum.shape, 0.095)
    safe = support > 0.98
    safe_lum = lum[safe] if np.any(safe) else lum.reshape(-1)

    blur1 = cv2.GaussianBlur(lum, (0, 0), 1.0)
    blur3 = cv2.GaussianBlur(lum, (0, 0), 3.0)
    blur7 = cv2.GaussianBlur(lum, (0, 0), 7.0)
    blur20 = cv2.GaussianBlur(lum, (0, 0), 20.0)
    blur55 = cv2.GaussianBlur(lum, (0, 0), 55.0)
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)

    star_seed = np.clip(
        (lum - _safe_percentile(safe_lum, 98.15))
        / max(1e-6, _safe_percentile(safe_lum, 99.98) - _safe_percentile(safe_lum, 98.15)),
        0.0,
        1.0,
    ) ** 1.55
    star_mask = cv2.GaussianBlur(star_seed.astype(np.float32), (0, 0), 1.0)
    star_protect = cv2.GaussianBlur(star_seed.astype(np.float32), (0, 0), 3.2)

    ha_signal = np.clip((rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]) * 0.70) / 0.26, 0.0, 1.0)
    oiii_signal = np.clip((np.maximum(rgb[..., 1], rgb[..., 2]) - rgb[..., 0] * 0.74) / 0.24, 0.0, 1.0)
    emission_signal = np.maximum(ha_signal, oiii_signal)

    broad_signal = np.clip(
        (blur55 - _safe_percentile(blur55[safe], 34.0, float(np.percentile(blur55, 34.0))))
        / max(1e-6, _safe_percentile(blur55[safe], 98.4, float(np.percentile(blur55, 98.4))) - _safe_percentile(blur55[safe], 34.0, float(np.percentile(blur55, 34.0)))),
        0.0,
        1.0,
    )
    mid_signal = np.clip(
        (blur20 - _safe_percentile(blur20[safe], 45.0, float(np.percentile(blur20, 45.0))))
        / max(1e-6, _safe_percentile(blur20[safe], 98.9, float(np.percentile(blur20, 98.9))) - _safe_percentile(blur20[safe], 45.0, float(np.percentile(blur20, 45.0)))),
        0.0,
        1.0,
    )
    color_signal = np.clip(
        (np.maximum(chroma, emission_signal) - _safe_percentile(np.maximum(chroma, emission_signal)[safe], 45.0, float(np.percentile(np.maximum(chroma, emission_signal), 45.0))))
        / max(
            1e-6,
            _safe_percentile(np.maximum(chroma, emission_signal)[safe], 99.1, float(np.percentile(np.maximum(chroma, emission_signal), 99.1)))
            - _safe_percentile(np.maximum(chroma, emission_signal)[safe], 45.0, float(np.percentile(np.maximum(chroma, emission_signal), 45.0))),
        ),
        0.0,
        1.0,
    )

    ridge = np.maximum(blur3 - blur20, 0.0)
    ridge = np.clip(
        (ridge - _safe_percentile(ridge[safe], 54.0, float(np.percentile(ridge, 54.0))))
        / max(1e-6, _safe_percentile(ridge[safe], 99.25, float(np.percentile(ridge, 99.25))) - _safe_percentile(ridge[safe], 54.0, float(np.percentile(ridge, 54.0)))),
        0.0,
        1.0,
    )
    edge = np.abs(cv2.Laplacian(blur20.astype(np.float32), cv2.CV_32F, ksize=3))
    edge = np.clip(
        (edge - _safe_percentile(edge[safe], 68.0, float(np.percentile(edge, 68.0))))
        / max(1e-6, _safe_percentile(edge[safe], 99.4, float(np.percentile(edge, 99.4))) - _safe_percentile(edge[safe], 68.0, float(np.percentile(edge, 68.0)))),
        0.0,
        1.0,
    )

    structure_gate = np.clip(color_signal * 0.62 + ridge * 0.48 + edge * 0.42 + emission_signal * 0.54, 0.0, 1.0)
    broad_nebula = np.clip(broad_signal * (0.18 + structure_gate * 0.92), 0.0, 1.0)
    mid_nebula = np.clip(mid_signal * (0.22 + structure_gate * 0.88), 0.0, 1.0)
    artifact_guard = np.clip(support**3.0, 0.0, 1.0)

    nebula_mask = np.clip(
        broad_nebula * 0.40 + mid_nebula * 0.40 + color_signal * 0.48 + ridge * 0.42 + edge * 0.30,
        0.0,
        1.0,
    ) * artifact_guard
    nebula_mask = cv2.GaussianBlur((nebula_mask**0.82).astype(np.float32), (0, 0), 3.5)
    nebula_core = np.clip(np.maximum(ridge, edge) * (0.55 + color_signal * 0.70), 0.0, 1.0)
    nebula_core = cv2.GaussianBlur(((nebula_core * artifact_guard) ** 0.72).astype(np.float32), (0, 0), 1.2)
    nebula_protect = cv2.GaussianBlur(np.maximum(nebula_mask, broad_nebula * 0.54).astype(np.float32), (0, 0), 14.0)

    sky_mask = np.clip(support * (1.0 - nebula_protect * 1.38) * (1.0 - star_protect * 1.12), 0.0, 1.0)
    clean_sky = np.clip(
        sky_mask
        * (1.0 - color_signal * 0.75)
        * (1.0 - ridge * 0.72)
        * (1.0 - edge * 0.58),
        0.0,
        1.0,
    )
    sky_pixels = rgb[clean_sky > 0.72]
    if sky_pixels.size < 512:
        fallback = (support > 0.95) & (star_protect < 0.10) & (nebula_protect < 0.25) & (lum < np.percentile(lum, 52.0))
        sky_pixels = rgb[fallback]
    if sky_pixels.size < 512:
        sky_pixels = rgb.reshape(-1, 3)

    bg_before = np.median(sky_pixels, axis=0).astype(np.float32)
    neutral = float(np.mean(bg_before))
    gains = np.clip(neutral / np.maximum(bg_before, 1e-5), 0.65, 1.38)
    neutral_mix = np.clip(clean_sky * 0.88 + sky_mask * 0.18, 0.0, 0.92)
    output = rgb * (1.0 - neutral_mix[..., None]) + np.clip(rgb * gains.reshape(1, 1, 3), 0.0, 1.0) * neutral_mix[..., None]

    bg_fill = output.copy()
    bg_median = np.median(output[clean_sky > 0.66], axis=0) if np.count_nonzero(clean_sky > 0.66) > 512 else np.median(output.reshape(-1, 3), axis=0)
    bg_fill[clean_sky <= 0.66] = bg_median
    bg_field = np.stack(
        [cv2.GaussianBlur(bg_fill[..., channel].astype(np.float32), (0, 0), 76.0) for channel in range(3)],
        axis=2,
    )
    correction = bg_field - bg_median.reshape(1, 1, 3)
    correction_mix = np.clip(clean_sky * 0.92 + sky_mask * (1.0 - nebula_protect) * 0.34, 0.0, 1.0)
    output = np.clip(output - correction * correction_mix[..., None] * 0.70, 0.0, 1.0)

    out_lum = _luminance(output).astype(np.float32)
    sky_values = out_lum[clean_sky > 0.66]
    sky_floor = _safe_percentile(sky_values, 52.0, float(np.percentile(out_lum, 20.0)))
    black_point = max(_safe_percentile(sky_values, 12.0, float(np.percentile(out_lum, 5.0))) * 0.42, 0.002)
    dark_target = np.clip((out_lum - black_point) / max(1e-6, 1.0 - black_point), 0.0, 1.0)
    dark_target = np.clip(dark_target**1.10, 0.0, 1.0) * 0.64
    darkened = np.clip(output * (dark_target / np.maximum(out_lum, 1e-5))[..., None], 0.0, 1.0)
    dark_mix = np.clip(
        (clean_sky[..., None] * 0.64 + sky_mask[..., None] * (1.0 - nebula_protect[..., None]) * 0.26),
        0.0,
        0.68,
    )
    output = np.clip(output * (1.0 - dark_mix) + darkened * dark_mix, 0.0, 1.0)

    out_lum = _luminance(output).astype(np.float32)
    sky_floor_color = np.array([0.014, 0.0115, 0.014], dtype=np.float32).reshape(1, 1, 3)
    sky_floor_weight = np.clip((0.055 - out_lum) / 0.055, 0.0, 1.0)
    sky_floor_mix = np.clip(
        sky_floor_weight[..., None]
        * (clean_sky[..., None] * 0.16 + sky_mask[..., None] * (1.0 - nebula_protect[..., None]) * 0.08)
        * (1.0 - star_protect[..., None] * 0.80),
        0.0,
        0.14,
    )
    output = np.clip(output * (1.0 - sky_floor_mix) + sky_floor_color * sky_floor_mix, 0.0, 1.0)

    out_lum = _luminance(output).astype(np.float32)
    fine_detail = blur1 - blur7
    mid_detail = blur7 - blur20
    broad_shadow = blur20 - blur55
    sculpted_lum = np.clip(
        out_lum
        + fine_detail * nebula_core * 0.36
        + mid_detail * nebula_mask * 0.62
        + broad_shadow * nebula_mask * 0.12,
        0.0,
        1.0,
    )
    detail_mix = np.clip((nebula_mask * 0.70 + nebula_core * 0.42) * (1.0 - star_protect * 0.88), 0.0, 0.92)
    output = np.clip(output * (1.0 - detail_mix[..., None]) + output * (sculpted_lum / np.maximum(out_lum, 1e-5))[..., None] * detail_mix[..., None], 0.0, 1.0)

    out_lum = _luminance(output).astype(np.float32)
    saturation_mask = np.clip(nebula_mask[..., None] * 0.48 + nebula_core[..., None] * 0.18 + star_mask[..., None] * 0.08 - clean_sky[..., None] * 0.18, -0.18, 0.54)
    output = np.clip(out_lum[..., None] + (output - out_lum[..., None]) * (1.0 + saturation_mask), 0.0, 1.0)

    out_lum = _luminance(output).astype(np.float32)
    red_sheet = np.clip(output[..., 0] - np.maximum(output[..., 1], output[..., 2]) * 1.06, 0.0, 1.0)
    cyan_sheet = np.clip(np.maximum(output[..., 1], output[..., 2]) - output[..., 0] * 1.14, 0.0, 1.0)
    output[..., 0] = np.clip(output[..., 0] - red_sheet * clean_sky * 0.52, 0.0, 1.0)
    output[..., 1] = np.clip(output[..., 1] - cyan_sheet * clean_sky * 0.14, 0.0, 1.0)
    output[..., 2] = np.clip(output[..., 2] - cyan_sheet * clean_sky * 0.18, 0.0, 1.0)

    out_lum = _luminance(output).astype(np.float32)
    ha_signal = np.clip((output[..., 0] - np.maximum(output[..., 1], output[..., 2]) * 0.70) / 0.25, 0.0, 1.0)
    oiii_signal = np.clip((np.maximum(output[..., 1], output[..., 2]) - output[..., 0] * 0.70) / 0.25, 0.0, 1.0)
    cool_struct = np.clip(
        (output[..., 2] * 0.68 + output[..., 1] * 0.36 - output[..., 0] * 0.34)
        / max(1e-6, np.percentile(out_lum, 99.4) - np.percentile(out_lum, 35.0)),
        0.0,
        1.0,
    )
    cool_filament = np.clip(edge * nebula_core * (0.52 + ridge * 0.40) * (1.0 - star_protect * 0.72), 0.0, 1.0)
    cool_candidate = np.clip(
        np.maximum(oiii_signal, cool_struct * 0.54 + cool_filament * 0.62) * nebula_core * (1.0 - ha_signal * 0.18),
        0.0,
        1.0,
    )
    warm_signal = np.clip((ha_signal * 0.62 + ridge * 0.36 + color_signal * 0.12) * nebula_mask, 0.0, 1.0)
    highlight = np.clip(
        (out_lum - np.percentile(out_lum, 80.0)) / max(1e-6, np.percentile(out_lum, 99.6) - np.percentile(out_lum, 80.0)),
        0.0,
        1.0,
    ) * nebula_mask

    halo_lum = np.clip(out_lum * (0.76 + broad_nebula * 0.30), 0.0, 1.0)
    filament_lum = np.clip(out_lum * (0.88 + nebula_core * 0.46), 0.0, 1.0)
    core_lum = np.clip(out_lum * (0.96 + highlight * 0.22), 0.0, 1.0)

    pink_target = halo_lum[..., None] * np.array([1.08, 0.64, 0.96], dtype=np.float32).reshape(1, 1, 3)
    gold_target = filament_lum[..., None] * np.array([1.16, 0.78, 0.44], dtype=np.float32).reshape(1, 1, 3)
    cyan_target = filament_lum[..., None] * np.array([0.46, 0.86, 1.28], dtype=np.float32).reshape(1, 1, 3)
    cream = core_lum[..., None] * np.array([1.10, 0.94, 0.78], dtype=np.float32).reshape(1, 1, 3)

    halo_mix = np.clip(broad_nebula[..., None] * nebula_mask[..., None] * (1.0 - nebula_core[..., None]) * 0.14, 0.0, 0.16)
    gold_mix = np.clip((warm_signal * nebula_core * (1.0 - cool_candidate * 0.46))[..., None] * 0.31, 0.0, 0.33)
    cyan_mix = np.clip((cool_candidate * (0.54 + edge * 0.46))[..., None] * 0.34, 0.0, 0.34)
    cream_mix = np.clip((highlight * (0.12 + warm_signal * 0.13))[..., None], 0.0, 0.26)

    output = np.clip(output * (1.0 - halo_mix) + pink_target * halo_mix, 0.0, 1.0)
    output = np.clip(output * (1.0 - gold_mix) + gold_target * gold_mix, 0.0, 1.0)
    output = np.clip(output * (1.0 - cyan_mix) + cyan_target * cyan_mix, 0.0, 1.0)
    output = np.clip(output * (1.0 - cream_mix) + cream * cream_mix, 0.0, 1.0)

    diffuse_red_haze = np.clip(ha_signal * broad_nebula * (1.0 - nebula_core * 1.35), 0.0, 1.0)
    haze_mix = np.clip(diffuse_red_haze[..., None] * (0.06 + clean_sky[..., None] * 0.16), 0.0, 0.18)
    neutral_haze = _luminance(output)[..., None] * np.array([0.92, 0.84, 0.78], dtype=np.float32).reshape(1, 1, 3)
    output = np.clip(output * (1.0 - haze_mix) + neutral_haze * haze_mix, 0.0, 1.0)

    hot_nebula = np.clip((highlight * nebula_core * (1.0 - star_protect * 0.84))[..., None], 0.0, 1.0)
    highlight_mix = np.clip(hot_nebula * 0.18, 0.0, 0.18)
    controlled_highlight = np.clip(output * np.array([0.94, 0.94, 0.90], dtype=np.float32).reshape(1, 1, 3), 0.0, 1.0)
    output = np.clip(output * (1.0 - highlight_mix) + controlled_highlight * highlight_mix, 0.0, 1.0)

    final_lum = _luminance(output).astype(np.float32)
    star_soften = np.clip(star_mask * (1.0 - nebula_core * 0.50), 0.0, 0.46)
    output = np.clip(output * (1.0 - star_soften[..., None] * 0.10), 0.0, 1.0)
    output = np.clip(output * (1.0 - clean_sky[..., None] * 0.02), 0.0, 1.0)

    export_support = _edge_support(lum.shape, 0.18)
    edge_artifact = np.clip((1.0 - export_support) ** 1.08, 0.0, 1.0)
    edge_artifact = cv2.GaussianBlur(edge_artifact.astype(np.float32), (0, 0), 4.0)
    neutral_sky = np.array([0.014, 0.0115, 0.014], dtype=np.float32).reshape(1, 1, 3)
    border_noise = np.clip((chroma * 1.6 + red_sheet * 1.2 + cyan_sheet * 1.0) * edge_artifact, 0.0, 1.0)
    edge_mix = np.clip(
        edge_artifact[..., None] * (1.0 - nebula_mask[..., None] * 0.86) * (0.34 + border_noise[..., None] * 0.34),
        0.0,
        0.62,
    )
    output = np.clip(output * (1.0 - edge_mix) + neutral_sky * edge_mix, 0.0, 1.0)

    bg_after_pixels = output[clean_sky > 0.66]
    if bg_after_pixels.size < 512:
        bg_after_pixels = output.reshape(-1, 3)
    bg_after = np.median(bg_after_pixels, axis=0)
    if log:
        log(
            "Applied Cosmos-darkroom nebula finish: "
            f"bg_before_RGB={bg_before[0]:.5f},{bg_before[1]:.5f},{bg_before[2]:.5f}; "
            f"bg_after_RGB={bg_after[0]:.5f},{bg_after[1]:.5f},{bg_after[2]:.5f}; "
            f"ha_mean={float(np.mean(ha_signal * nebula_core)):.5f}; "
            f"oiii_mean={float(np.mean(oiii_signal * nebula_core)):.5f}; "
            f"cool_mean={float(np.mean(cool_candidate * nebula_core)):.5f}; "
            f"sky_mask_mean={float(np.mean(sky_mask)):.5f}; "
            f"clean_sky_mean={float(np.mean(clean_sky)):.5f}; "
            f"nebula_mask_mean={float(np.mean(nebula_mask)):.5f}; "
            f"nebula_core_mean={float(np.mean(nebula_core)):.5f}; "
            f"star_mask_mean={float(np.mean(star_mask)):.5f}"
        )
    return _to_uint16(output)


def _nebula_color_separation_strength(mode: str) -> float:
    normalized = str(mode or "Strong").strip().lower().replace("_", " ").replace("-", " ")
    if normalized in {"natural", "low", "subtle"}:
        return 0.18
    if normalized in {"strong", "high"}:
        return 0.68
    if normalized in {"showcase", "hoo", "showcase hoo"}:
        return 0.48
    return 0.30


def _is_showcase_hoo_mode(mode: str) -> bool:
    normalized = str(mode or "").strip().lower().replace("_", " ").replace("-", " ")
    return normalized in {"showcase", "hoo", "showcase hoo"}


def _apply_showcase_hoo_nebula_grade(
    output: np.ndarray,
    calibrated: np.ndarray,
    nebula_mask: np.ndarray,
    nebula_core: np.ndarray,
    ridge: np.ndarray,
    clean_sky: np.ndarray,
    star_protect: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Measured red/cyan nebula color separation for showcase nebula mode.

    This is intentionally separate from Siril calibration. Siril balances the data;
    this step separates real background-subtracted R/G/B nebula signal while masking
    the sky and stars.
    """
    base = np.clip(output.astype(np.float32), 0.0, 1.0)
    source = np.clip(calibrated.astype(np.float32), 0.0, 1.0)
    source_lum = _luminance(source).astype(np.float32)
    base_lum = _luminance(base).astype(np.float32)

    red = source[..., 0].astype(np.float32)
    green = source[..., 1].astype(np.float32)
    blue = source[..., 2].astype(np.float32)

    signal = np.clip(
        (nebula_mask * 0.68 + nebula_core * 0.26 + ridge * 0.34)
        * support
        * (1.0 - clean_sky * 0.92)
        * (1.0 - star_protect * 0.78),
        0.0,
        1.0,
    )
    signal = cv2.GaussianBlur(signal.astype(np.float32), (0, 0), 1.4)

    local_bg = cv2.GaussianBlur(source_lum, (0, 0), 42.0)
    top_hat = np.clip(source_lum - cv2.GaussianBlur(source_lum, (0, 0), 4.8), 0.0, 1.0)
    local_detail = np.clip(
        top_hat / max(1e-6, float(np.percentile(top_hat, 99.65))),
        0.0,
        1.0,
    ) ** 0.70
    strand_detail_raw = np.maximum(
        cv2.GaussianBlur(source_lum, (0, 0), 1.1)
        - cv2.GaussianBlur(source_lum, (0, 0), 8.5),
        0.0,
    )
    strand_detail = np.clip(
        strand_detail_raw / max(1e-6, float(np.percentile(strand_detail_raw, 99.45))),
        0.0,
        1.0,
    ) ** 0.62
    local_soft = cv2.GaussianBlur(source_lum, (0, 0), 5.5)
    filament_shadow = np.clip(
        (local_soft - source_lum)
        / max(1e-6, float(np.percentile(np.maximum(local_soft - source_lum, 0.0), 99.2))),
        0.0,
        1.0,
    )
    local_signal = np.clip(
        (source_lum - local_bg * 0.70)
        / max(1e-6, float(np.percentile(np.maximum(source_lum - local_bg * 0.70, 0.0), 99.55))),
        0.0,
        1.0,
    ) ** 0.55
    local_signal = cv2.GaussianBlur(local_signal.astype(np.float32), (0, 0), 2.0)

    bg_region = (clean_sky > 0.68) & (support > 0.96) & (star_protect < 0.14)
    if np.count_nonzero(bg_region) < 512:
        bg_region = (clean_sky > 0.55) & (support > 0.92)
    if np.count_nonzero(bg_region) >= 512:
        bg_rgb = np.median(source[bg_region], axis=0).astype(np.float32)
    else:
        bg_rgb = np.percentile(source.reshape(-1, 3), 22.0, axis=0).astype(np.float32)

    r_signal = np.clip(red - bg_rgb[0], 0.0, 1.0)
    g_signal = np.clip(green - bg_rgb[1], 0.0, 1.0)
    b_signal = np.clip(blue - bg_rgb[2], 0.0, 1.0)
    gb_signal = np.clip(g_signal * 0.52 + b_signal * 0.70, 0.0, 1.0)

    nebula_region = (signal > 0.06) & (clean_sky < 0.72) & (star_protect < 0.38)
    if np.count_nonzero(nebula_region) < 512:
        nebula_region = signal > 0.04
    r_high = max(1e-6, float(np.percentile(r_signal[nebula_region], 99.2)) if np.count_nonzero(nebula_region) else float(np.percentile(r_signal, 99.2)))
    gb_high = max(1e-6, float(np.percentile(gb_signal[nebula_region], 99.2)) if np.count_nonzero(nebula_region) else float(np.percentile(gb_signal, 99.2)))
    r_norm = np.clip(r_signal / r_high, 0.0, 1.0)
    gb_norm = np.clip(gb_signal / gb_high, 0.0, 1.0)

    ha_relative = np.clip(r_signal - np.maximum(g_signal, b_signal) * 0.40, 0.0, 1.0)
    oiii_relative = np.clip(gb_signal - r_signal * 0.30, 0.0, 1.0)
    ha_relative = np.clip(
        ha_relative
        / max(1e-6, float(np.percentile(ha_relative[nebula_region], 98.8)) if np.count_nonzero(nebula_region) else float(np.percentile(ha_relative, 98.8))),
        0.0,
        1.0,
    )
    oiii_relative = np.clip(
        oiii_relative
        / max(1e-6, float(np.percentile(oiii_relative[nebula_region], 98.8)) if np.count_nonzero(nebula_region) else float(np.percentile(oiii_relative, 98.8))),
        0.0,
        1.0,
    )

    filament_gate = np.clip(ridge * 0.58 + local_detail * 0.42 + nebula_core * 0.30, 0.0, 1.0)
    structured_signal = np.clip(local_detail * 0.46 + ridge * 0.42 + nebula_core * 0.24 + local_signal * 0.16, 0.0, 1.0)
    filament_texture = np.clip(0.38 + local_detail * 0.46 + ridge * 0.30 - filament_shadow * 0.16, 0.18, 1.0)
    ha_body = np.clip((r_norm * 0.68 + ha_relative * 0.32) * signal * (0.34 + filament_gate * 0.66) * filament_texture, 0.0, 1.0)
    ha_mask = cv2.GaussianBlur(ha_body.astype(np.float32), (0, 0), 2.05)
    oiii_mask = cv2.GaussianBlur(
        np.clip(
            (gb_norm * 0.62 + oiii_relative * 0.38)
            * signal
            * (0.18 + structured_signal * 0.86)
            * (1.0 - clean_sky * 0.98)
            * (1.0 - star_protect * 0.82),
            0.0,
            1.0,
        ).astype(np.float32),
        (0, 0),
        1.15,
    )
    mixed_mask = np.clip(np.minimum(ha_mask, oiii_mask) * signal, 0.0, 1.0)
    ha_mask = np.clip(ha_mask * (1.0 - oiii_mask * 0.08), 0.0, 1.0)
    oiii_mask = np.clip(oiii_mask * (1.0 - ha_mask * 0.30) + mixed_mask * 0.16, 0.0, 1.0)

    # Lift the empty sky toward the blue-gray showcase look without tinting stars.
    empty_sky = np.clip(clean_sky * (1.0 - nebula_mask * 0.78) * (1.0 - star_protect * 0.88), 0.0, 1.0)
    sky_floor = np.array([0.052, 0.067, 0.078], dtype=np.float32).reshape(1, 1, 3)
    showcase = np.clip(base * (1.0 - empty_sky[..., None] * 0.45) + sky_floor * (empty_sky[..., None] * 0.45), 0.0, 1.0)

    strand_carrier = np.clip(strand_detail * 0.64 + local_detail * 0.38 + ridge * 0.30, 0.0, 1.0)
    filament_color_texture = np.clip(0.12 + strand_carrier * 0.94 + filament_shadow * 0.08, 0.08, 1.0)
    red_structure = np.clip(
        strand_carrier * 0.78
        + np.maximum(strand_detail - cv2.GaussianBlur(strand_detail, (0, 0), 2.8), 0.0) * 0.60
        + ridge * 0.26,
        0.0,
        1.0,
    )
    red_push = cv2.GaussianBlur(
        np.clip((r_norm * 0.34 + ha_relative * 0.66) * signal * red_structure * 1.72, 0.0, 1.0).astype(np.float32),
        (0, 0),
        0.22,
    )
    red_fill_guard = cv2.GaussianBlur(red_push.astype(np.float32), (0, 0), 2.4)
    red_push = np.clip(red_push - red_fill_guard * 0.22, 0.0, 1.0)
    red_push = np.clip(red_push * (1.0 - oiii_mask * 0.05), 0.0, 0.96)
    blue_push = cv2.GaussianBlur(
        np.clip((gb_norm * 0.54 + oiii_relative * 0.46) * signal * red_structure * (0.34 + structured_signal * 0.46) * 1.28, 0.0, 1.0).astype(np.float32),
        (0, 0),
        0.28,
    )
    blue_fill_guard = cv2.GaussianBlur(blue_push.astype(np.float32), (0, 0), 2.6)
    blue_push = np.clip(blue_push - blue_fill_guard * 0.30, 0.0, 1.0)
    blue_push = np.clip(blue_push * (1.0 - red_push * 0.22), 0.0, 0.82)
    white_fiber = cv2.GaussianBlur(
        np.clip(strand_carrier * signal * (0.22 + red_push * 0.34 + blue_push * 0.42), 0.0, 1.0).astype(np.float32),
        (0, 0),
        0.32,
    )
    showcase = np.clip(showcase + white_fiber[..., None] * np.array([0.040, 0.044, 0.048], dtype=np.float32).reshape(1, 1, 3), 0.0, 1.0)
    ha_grade = np.stack(
        [
            showcase[..., 0] * (1.0 + red_push * 1.18),
            showcase[..., 1] * (1.0 + red_push * 0.28),
            showcase[..., 2] * (1.0 - red_push * 0.16),
        ],
        axis=2,
    )
    showcase = np.clip(showcase * (1.0 - red_push[..., None] * 0.78) + ha_grade * (red_push[..., None] * 0.78), 0.0, 1.0)

    oiii_grade = np.stack(
        [
            showcase[..., 0] * (1.0 - blue_push * 0.38),
            showcase[..., 1] * (1.0 + blue_push * 0.28),
            showcase[..., 2] * (1.0 + blue_push * 0.82),
        ],
        axis=2,
    )
    showcase = np.clip(showcase * (1.0 - blue_push[..., None] * 0.72) + oiii_grade * (blue_push[..., None] * 0.72), 0.0, 1.0)

    # Preserve luminance so the color pass does not turn filaments into flat paint.
    showcase_lum = _luminance(showcase).astype(np.float32)
    showcase = np.clip(showcase * (base_lum / np.maximum(showcase_lum, 1e-5))[..., None], 0.0, 1.0)

    color_mask = np.clip(red_push * 0.72 + blue_push * 0.78 + nebula_mask * 0.06, 0.0, 1.0)
    gray = _luminance(showcase).astype(np.float32)
    showcase = np.clip(gray[..., None] + (showcase - gray[..., None]) * (1.0 + color_mask[..., None] * 1.28), 0.0, 1.0)

    # Add fine chroma on top of measured color masks. This makes color ride the
    # filament texture instead of filling regions as flat paint.
    detail_carrier = np.clip(
        (local_detail * 0.40 + strand_detail * 0.52 + ridge * 0.38 + filament_shadow * 0.16)
        * signal
        * (1.0 - clean_sky * 0.94)
        * (1.0 - star_protect * 0.82),
        0.0,
        1.0,
    )
    ha_fiber = cv2.GaussianBlur(np.clip(red_push * detail_carrier, 0.0, 1.0).astype(np.float32), (0, 0), 0.30)
    oiii_fiber = cv2.GaussianBlur(
        np.clip((oiii_mask * 0.44 + oiii_relative * gb_norm * 0.56) * detail_carrier, 0.0, 1.0).astype(np.float32),
        (0, 0),
        0.40,
    )
    oiii_fiber = np.clip(oiii_fiber * (1.0 - ha_fiber * 0.28), 0.0, 1.0)
    fiber_lum = np.clip(base_lum * 0.62 + source_lum * 0.38, 0.018, 1.0)
    fiber_delta = (
        ha_fiber[..., None] * np.array([0.96, 0.22, -0.18], dtype=np.float32).reshape(1, 1, 3)
        + oiii_fiber[..., None] * np.array([-0.38, 0.24, 0.86], dtype=np.float32).reshape(1, 1, 3)
    )
    showcase = np.clip(showcase + fiber_delta * fiber_lum[..., None] * 1.08, 0.0, 1.0)
    fiber_lum_after = _luminance(showcase).astype(np.float32)
    fiber_target_lum = np.clip(base_lum + white_fiber * 0.070 + (ha_fiber + oiii_fiber) * signal * 0.050 + strand_detail * signal * 0.038, 0.0, 1.0)
    showcase = np.clip(showcase * (fiber_target_lum / np.maximum(fiber_lum_after, 1e-5))[..., None], 0.0, 1.0)

    # Add a soft H-alpha body and OIII filament haze. This keeps the result wispy
    # instead of only coloring isolated high-SNR knots.
    ha_haze = cv2.GaussianBlur((red_push * (0.02 + local_detail * 0.44 + strand_detail * 0.30)).astype(np.float32), (0, 0), 0.38)[..., None]
    oiii_haze = cv2.GaussianBlur((oiii_mask * (0.24 + ridge * 0.42 + local_detail * 0.24)).astype(np.float32), (0, 0), 0.82)[..., None]
    showcase = np.clip(
        showcase
        + ha_haze * np.array([0.006, 0.002, 0.000], dtype=np.float32).reshape(1, 1, 3)
        + oiii_haze * np.array([0.000, 0.022, 0.062], dtype=np.float32).reshape(1, 1, 3),
        0.0,
        1.0,
    )
    post_haze_lum = _luminance(showcase).astype(np.float32)
    texture_lum = np.clip(
        base_lum
        + local_detail * signal * 0.022
        + strand_detail * signal * 0.046
        + ridge * signal * 0.034
        - filament_shadow * signal * 0.026,
        0.0,
        1.0,
    )
    target_haze_lum = np.clip(texture_lum, 0.0, 1.0)
    haze_keep = np.clip(signal * 0.74 + ridge * 0.24, 0.0, 1.0)
    lum_fixed = np.clip(showcase * (target_haze_lum / np.maximum(post_haze_lum, 1e-5))[..., None], 0.0, 1.0)
    showcase = np.clip(showcase * (1.0 - haze_keep[..., None] * 0.78) + lum_fixed * (haze_keep[..., None] * 0.78), 0.0, 1.0)

    hot = np.clip((np.max(showcase, axis=2) - 0.70) / 0.30, 0.0, 1.0)
    compressed = showcase / (1.0 + showcase * 0.72) * 1.16
    showcase = np.clip(showcase * (1.0 - hot[..., None] * 0.48) + compressed * (hot[..., None] * 0.48), 0.0, 1.0)

    bg_smooth = cv2.bilateralFilter((showcase * 255.0).clip(0, 255).astype(np.uint8), d=0, sigmaColor=18, sigmaSpace=12).astype(np.float32) / 255.0
    bg_mix = np.clip(empty_sky[..., None] * 0.30, 0.0, 0.30)
    showcase = np.clip(showcase * (1.0 - bg_mix) + bg_smooth * bg_mix, 0.0, 1.0)

    return showcase, float(np.mean(ha_mask * signal)), float(np.mean(oiii_mask * signal))


def _apply_showcase_hoo_luminance_grade(
    output: np.ndarray,
    calibrated: np.ndarray,
    nebula_mask: np.ndarray,
    nebula_core: np.ndarray,
    ridge: np.ndarray,
    clean_sky: np.ndarray,
    star_protect: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Showcase HOO-like nebula grade driven by luminance ridges and real color bias."""
    base = np.clip(output.astype(np.float32), 0.0, 1.0)
    source = np.clip(calibrated.astype(np.float32), 0.0, 1.0)
    lum = _luminance(source).astype(np.float32)
    base_lum = _luminance(base).astype(np.float32)

    broad_bg = cv2.GaussianBlur(lum, (0, 0), 110.0)
    flat_lum = np.clip(lum - (broad_bg - float(np.median(broad_bg))) * 0.52, 0.0, 1.0)
    blur4 = cv2.GaussianBlur(flat_lum, (0, 0), 4.0)
    blur14 = cv2.GaussianBlur(flat_lum, (0, 0), 1.4)
    blur6 = cv2.GaussianBlur(flat_lum, (0, 0), 6.0)
    blur18 = cv2.GaussianBlur(flat_lum, (0, 0), 18.0)
    blur70 = cv2.GaussianBlur(flat_lum, (0, 0), 70.0)
    local_ridge = np.maximum(blur4 - blur18, 0.0)
    local_ridge = np.clip(local_ridge / max(1e-6, float(np.percentile(local_ridge, 99.45))), 0.0, 1.0)
    fine_ridge = np.maximum(blur14 - blur6, 0.0)
    fine_ridge = np.clip(fine_ridge / max(1e-6, float(np.percentile(fine_ridge, 99.55))), 0.0, 1.0)
    broad = np.clip(
        (blur70 - float(np.percentile(blur70, 35.0)))
        / max(1e-6, float(np.percentile(blur70, 99.0) - np.percentile(blur70, 35.0))),
        0.0,
        1.0,
    )
    signal = np.clip(
        (local_ridge * 0.58 + broad * 0.64 + nebula_mask * 0.22 + nebula_core * 0.12)
        * support
        * (1.0 - star_protect * 0.78)
        * (1.0 - clean_sky * 0.18),
        0.0,
        1.0,
    )
    signal = cv2.GaussianBlur(signal.astype(np.float32), (0, 0), 1.4)

    stretched_lum = np.arcsinh(flat_lum * 3.9) / np.arcsinh(3.9)
    stretched_lum = flat_lum * (1.0 - signal * 0.70) + stretched_lum * (signal * 0.70)
    showcase = np.clip(base * (stretched_lum / np.maximum(base_lum, 1e-5))[..., None], 0.0, 1.0)

    empty_sky = np.clip(clean_sky * (1.0 - signal * 0.88) * (1.0 - star_protect * 0.90) * support, 0.0, 1.0)
    sky_floor = np.array([0.056, 0.072, 0.084], dtype=np.float32).reshape(1, 1, 3)
    showcase = np.clip(showcase * (1.0 - empty_sky[..., None] * 0.46) + sky_floor * (empty_sky[..., None] * 0.46), 0.0, 1.0)

    red = source[..., 0].astype(np.float32)
    green = source[..., 1].astype(np.float32)
    blue = source[..., 2].astype(np.float32)
    red_raw = np.maximum(red - np.minimum(green, blue) * 0.72, 0.0)
    cyan_raw = np.maximum((green + blue) * 0.50 - red * 0.65, 0.0)
    red_bias = np.clip(red_raw / max(1e-6, float(np.percentile(red_raw, 99.0))), 0.0, 1.0)
    cyan_bias = np.clip(cyan_raw / max(1e-6, float(np.percentile(cyan_raw, 99.0))), 0.0, 1.0)

    soft_signal = cv2.GaussianBlur(signal.astype(np.float32), (0, 0), 5.5)
    red_mask = np.clip((local_ridge ** 1.22) * (0.24 + signal * 0.88) * (0.62 + red_bias * 0.58) * (0.42 + fine_ridge * 0.72), 0.0, 1.0)
    red_mask = cv2.GaussianBlur(red_mask.astype(np.float32), (0, 0), 0.72)
    cyan_outer = np.clip((soft_signal - red_mask * 0.40) * (local_ridge ** 0.38), 0.0, 1.0)
    cyan_mask = np.clip(cyan_outer * (0.42 + cyan_bias * 0.92) * (1.0 - red_mask * 0.72), 0.0, 1.0)
    cyan_mask = cv2.GaussianBlur(cyan_mask.astype(np.float32), (0, 0), 1.25)

    lum_target = np.clip(stretched_lum + local_ridge * signal * 0.052 + soft_signal * 0.012, 0.0, 1.0)
    neutral = np.clip(lum_target[..., None] * np.array([0.95, 0.98, 1.03], dtype=np.float32).reshape(1, 1, 3), 0.0, 1.0)
    red_palette = np.array([1.00, 0.14, 0.025], dtype=np.float32).reshape(1, 1, 3)
    cyan_palette = np.array([0.035, 0.52, 1.00], dtype=np.float32).reshape(1, 1, 3)
    colorized = neutral
    colorized = np.clip(
        colorized * (1.0 - cyan_mask[..., None] * 0.78)
        + cyan_palette * lum_target[..., None] * (cyan_mask[..., None] * 1.20),
        0.0,
        1.0,
    )
    colorized = np.clip(
        colorized * (1.0 - red_mask[..., None] * 0.90)
        + red_palette * lum_target[..., None] * (red_mask[..., None] * 1.34),
        0.0,
        1.0,
    )
    color_lum = _luminance(colorized).astype(np.float32)
    colorized = np.clip(colorized * (lum_target / np.maximum(color_lum, 1e-5))[..., None], 0.0, 1.0)
    color_mask = np.clip(red_mask + cyan_mask, 0.0, 1.0)
    colorized_lum = _luminance(colorized).astype(np.float32)
    colorized = np.clip(
        colorized_lum[..., None] + (colorized - colorized_lum[..., None]) * (1.0 + color_mask[..., None] * 0.82),
        0.0,
        1.0,
    )
    color_blend = np.clip(signal * (0.18 + color_mask * 1.05), 0.0, 0.86)
    showcase = np.clip(showcase * (1.0 - color_blend[..., None]) + colorized * color_blend[..., None], 0.0, 1.0)

    sky_smooth = cv2.bilateralFilter((showcase * 255.0).clip(0, 255).astype(np.uint8), d=0, sigmaColor=16, sigmaSpace=10).astype(np.float32) / 255.0
    smooth_mix = np.clip(empty_sky[..., None] * 0.18, 0.0, 0.18)
    showcase = np.clip(showcase * (1.0 - smooth_mix) + sky_smooth * smooth_mix, 0.0, 1.0)
    return showcase, float(np.mean(red_mask * signal)), float(np.mean(cyan_mask * signal))


def compose_pixinsight_nebula_layers(
    starless_image: np.ndarray,
    stars_image: np.ndarray,
    log: LogCallback | None = None,
    star_strength: float = 0.70,
    color_separation: str = "Balanced",
    color_reference_image: np.ndarray | None = None,
) -> np.ndarray:
    """Controlled nebula composer: process starless signal and stars separately."""
    starless = _to_float01(starless_image)
    stars = _to_float01(stars_image)
    color_reference = None
    if color_reference_image is not None:
        candidate = _to_float01(color_reference_image)
        if candidate.ndim == 3 and candidate.shape[-1] >= 3 and candidate.shape[:2] == starless.shape[:2]:
            color_reference = candidate[..., :3]
    if starless.ndim != 3 or starless.shape[-1] < 3:
        return _to_uint16(starless)
    if stars.ndim != 3 or stars.shape[-1] < 3 or stars.shape[:2] != starless.shape[:2]:
        stars = np.zeros_like(starless)

    height, width = starless.shape[:2]
    support = _edge_support((height, width), 0.045)
    safe = support > 0.98
    lum = _luminance(starless).astype(np.float32)
    safe_lum = lum[safe] if np.any(safe) else lum.reshape(-1)
    chroma = np.max(starless, axis=2) - np.min(starless, axis=2)

    star_lum = np.max(stars, axis=2).astype(np.float32)
    safe_star_lum = star_lum[safe] if np.any(safe) else star_lum.reshape(-1)
    star_low = _safe_percentile(safe_star_lum, 92.0, float(np.percentile(star_lum, 92.0)))
    star_high = _safe_percentile(safe_star_lum, 99.9, float(np.percentile(star_lum, 99.9)))
    star_seed = np.clip((star_lum - star_low) / max(1e-6, star_high - star_low), 0.0, 1.0) ** 1.15
    star_protect = cv2.GaussianBlur(star_seed.astype(np.float32), (0, 0), 2.2)

    blur8 = cv2.GaussianBlur(lum, (0, 0), 8.0)
    blur28 = cv2.GaussianBlur(lum, (0, 0), 28.0)
    blur80 = cv2.GaussianBlur(lum, (0, 0), 80.0)
    broad_low = _safe_percentile(blur80[safe], 34.0, float(np.percentile(blur80, 34.0)))
    broad_high = _safe_percentile(blur80[safe], 98.6, float(np.percentile(blur80, 98.6)))
    broad = np.clip((blur80 - broad_low) / max(1e-6, broad_high - broad_low), 0.0, 1.0)
    mid_low = _safe_percentile(blur28[safe], 45.0, float(np.percentile(blur28, 45.0)))
    mid_high = _safe_percentile(blur28[safe], 99.0, float(np.percentile(blur28, 99.0)))
    mid = np.clip((blur28 - mid_low) / max(1e-6, mid_high - mid_low), 0.0, 1.0)
    chroma_low = _safe_percentile(chroma[safe], 48.0, float(np.percentile(chroma, 48.0)))
    chroma_high = _safe_percentile(chroma[safe], 99.1, float(np.percentile(chroma, 99.1)))
    color_signal = np.clip((chroma - chroma_low) / max(1e-6, chroma_high - chroma_low), 0.0, 1.0)
    ridge = np.maximum(blur8 - blur28, 0.0)
    ridge_low = _safe_percentile(ridge[safe], 58.0, float(np.percentile(ridge, 58.0)))
    ridge_high = _safe_percentile(ridge[safe], 99.35, float(np.percentile(ridge, 99.35)))
    ridge = np.clip((ridge - ridge_low) / max(1e-6, ridge_high - ridge_low), 0.0, 1.0)
    fine_detail_raw = np.abs(lum - blur8).astype(np.float32)
    fine_low = _safe_percentile(fine_detail_raw[safe], 68.0, float(np.percentile(fine_detail_raw, 68.0)))
    fine_high = _safe_percentile(fine_detail_raw[safe], 99.45, float(np.percentile(fine_detail_raw, 99.45)))
    fine_detail = np.clip((fine_detail_raw - fine_low) / max(1e-6, fine_high - fine_low), 0.0, 1.0)
    fine_detail = cv2.GaussianBlur(fine_detail.astype(np.float32), (0, 0), 1.15)

    nebula_mask = np.clip(broad * 0.36 + mid * 0.42 + color_signal * 0.28 + ridge * 0.30, 0.0, 1.0)
    nebula_mask = cv2.GaussianBlur((nebula_mask * support * (1.0 - star_protect * 0.70)).astype(np.float32), (0, 0), 4.0)
    nebula_core = np.clip((ridge * 0.72 + color_signal * 0.28) * nebula_mask, 0.0, 1.0)
    nebula_core = cv2.GaussianBlur(nebula_core.astype(np.float32), (0, 0), 1.3)
    sky_mask = np.clip(support * (1.0 - nebula_mask * 1.34) * (1.0 - star_protect * 1.18), 0.0, 1.0)
    clean_sky = np.clip(sky_mask * (1.0 - ridge * 0.78) * (1.0 - color_signal * 0.72), 0.0, 1.0)

    sky_pixels = starless[clean_sky > 0.70]
    if sky_pixels.size < 512:
        fallback = safe & (lum < np.percentile(safe_lum, 55.0)) & (star_protect < 0.10) & (nebula_mask < 0.22)
        sky_pixels = starless[fallback]
    if sky_pixels.size < 512:
        sky_pixels = starless.reshape(-1, 3)
    bg_before = np.median(sky_pixels, axis=0).astype(np.float32)
    neutral = float(np.mean(bg_before))
    gains = np.clip(neutral / np.maximum(bg_before, 1e-5), 0.74, 1.28)
    calibrated = np.clip(
        starless * (1.0 - clean_sky[..., None] * 0.80)
        + starless * gains.reshape(1, 1, 3) * (clean_sky[..., None] * 0.80),
        0.0,
        1.0,
    )

    bg_median = (
        np.median(calibrated[clean_sky > 0.68], axis=0)
        if np.count_nonzero(clean_sky > 0.68) > 512
        else np.median(calibrated.reshape(-1, 3), axis=0)
    )
    bg_fill = calibrated.copy()
    bg_fill[clean_sky <= 0.68] = bg_median
    bg_field = np.stack(
        [cv2.GaussianBlur(bg_fill[..., channel].astype(np.float32), (0, 0), 96.0) for channel in range(3)],
        axis=2,
    )
    bg_correction = bg_field - bg_median.reshape(1, 1, 3)
    gradient_mix = np.clip(clean_sky[..., None] * 0.82 + sky_mask[..., None] * (1.0 - nebula_mask[..., None]) * 0.30, 0.0, 0.92)
    calibrated = np.clip(calibrated - bg_correction * gradient_mix * 0.66, 0.0, 1.0)

    color_calibrated = calibrated
    if color_reference is not None:
        ref_pixels = color_reference[clean_sky > 0.70]
        if ref_pixels.size < 512:
            ref_pixels = color_reference[fallback]
        if ref_pixels.size < 512:
            ref_pixels = color_reference.reshape(-1, 3)
        ref_bg = np.median(ref_pixels, axis=0).astype(np.float32)
        ref_neutral = float(np.mean(ref_bg))
        ref_gains = np.clip(ref_neutral / np.maximum(ref_bg, 1e-5), 0.62, 1.48)
        color_calibrated = np.clip(
            color_reference * (1.0 - clean_sky[..., None] * 0.52)
            + color_reference * ref_gains.reshape(1, 1, 3) * (clean_sky[..., None] * 0.52),
            0.0,
            1.0,
        )
        ref_bg_median = (
            np.median(color_calibrated[clean_sky > 0.68], axis=0)
            if np.count_nonzero(clean_sky > 0.68) > 512
            else np.median(color_calibrated.reshape(-1, 3), axis=0)
        )
        ref_fill = color_calibrated.copy()
        ref_fill[clean_sky <= 0.68] = ref_bg_median
        ref_bg_field = np.stack(
            [cv2.GaussianBlur(ref_fill[..., channel].astype(np.float32), (0, 0), 96.0) for channel in range(3)],
            axis=2,
        )
        color_calibrated = np.clip(color_calibrated - (ref_bg_field - ref_bg_median.reshape(1, 1, 3)) * gradient_mix * 0.46, 0.0, 1.0)

    lum = _luminance(calibrated).astype(np.float32)
    sky_lum = lum[clean_sky > 0.68]
    black = _safe_percentile(sky_lum, 18.0, float(np.percentile(lum, 4.0))) * 0.55
    linear = np.clip((calibrated - black) / max(1e-6, 1.0 - black), 0.0, 1.0)
    linear_lum = _luminance(linear).astype(np.float32)

    arcsinh_lum = np.arcsinh(linear_lum * 2.55) / np.arcsinh(2.55)
    lift_mask = np.clip(nebula_mask * 0.60 + nebula_core * 0.16 + sky_mask * 0.045, 0.0, 0.66)
    stretched_lum = linear_lum * (1.0 - lift_mask) + arcsinh_lum * lift_mask
    stretched_lum = stretched_lum * 0.62 + np.clip(stretched_lum**0.96, 0.0, 1.0) * 0.38
    highlight = np.clip(
        (stretched_lum - np.percentile(stretched_lum, 96.4))
        / max(1e-6, np.percentile(stretched_lum, 99.92) - np.percentile(stretched_lum, 96.4)),
        0.0,
        1.0,
    )
    compressed_lum = stretched_lum / (1.0 + highlight * 0.28)
    output = np.clip(linear * (compressed_lum / np.maximum(linear_lum, 1e-5))[..., None], 0.0, 1.0)

    output_lum = _luminance(output).astype(np.float32)
    sky_float = np.clip(clean_sky * (1.0 - nebula_mask * 0.80), 0.0, 1.0)
    output8 = (output * 255.0).clip(0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoisingColored(output8, None, 14, 22, 7, 25).astype(np.float32) / 255.0
    output = np.clip(output * (1.0 - sky_float[..., None] * 0.66) + denoised * (sky_float[..., None] * 0.66), 0.0, 1.0)

    output_lum = _luminance(output).astype(np.float32)
    fine = output_lum - cv2.GaussianBlur(output_lum, (0, 0), 2.2)
    mid_detail = cv2.GaussianBlur(output_lum, (0, 0), 4.0) - cv2.GaussianBlur(output_lum, (0, 0), 17.0)
    detail_lum = np.clip(output_lum + fine * nebula_core * 0.065 + mid_detail * nebula_mask * 0.105, 0.0, 1.0)
    detail_mix = np.clip((nebula_mask * 0.30 + nebula_core * 0.26) * (1.0 - star_protect * 0.92), 0.0, 0.38)
    output = np.clip(
        output * (1.0 - detail_mix[..., None])
        + output * (detail_lum / np.maximum(output_lum, 1e-5))[..., None] * detail_mix[..., None],
        0.0,
        1.0,
    )

    output_lum = _luminance(output).astype(np.float32)
    sat_scale = np.clip(1.0 + nebula_mask[..., None] * 0.22 + nebula_core[..., None] * 0.08 - clean_sky[..., None] * 0.28, 0.66, 1.26)
    output = np.clip(output_lum[..., None] + (output - output_lum[..., None]) * sat_scale, 0.0, 1.0)

    separation_strength = _nebula_color_separation_strength(color_separation)
    ha_mean = 0.0
    oiii_mean = 0.0
    if separation_strength > 0.0:
        red = color_calibrated[..., 0].astype(np.float32)
        green = color_calibrated[..., 1].astype(np.float32)
        blue = color_calibrated[..., 2].astype(np.float32)
        max_gb = np.maximum(green, blue)
        oiii_base = green * 0.48 + blue * 0.72
        ha_raw = np.clip(red - max_gb * 0.84, 0.0, 1.0)
        oiii_raw = np.clip(oiii_base - red * 0.92, 0.0, 1.0)

        local_bg = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 38.0)
        local_noise = cv2.GaussianBlur(np.abs(lum - local_bg).astype(np.float32), (0, 0), 9.0)
        local_snr = np.clip((lum - local_bg) / np.maximum(local_noise * 4.4, 1e-5), 0.0, 1.0)
        filament_snr = np.clip(local_snr * 0.42 + ridge * 0.38 + nebula_core * 0.34, 0.0, 1.0)

        signal_gate = np.clip(
            (nebula_mask * 0.58 + nebula_core * 0.16 + ridge * 0.18 + color_signal * 0.12)
            * support
            * (0.48 + filament_snr * 0.52)
            * (1.0 - clean_sky * 0.96)
            * (1.0 - star_protect * 0.86),
            0.0,
            1.0,
        )
        signal_gate = cv2.GaussianBlur(signal_gate.astype(np.float32), (0, 0), 2.4)

        ha_values = ha_raw[signal_gate > 0.10]
        oiii_values = oiii_raw[signal_gate > 0.10]
        ha_high = _safe_percentile(ha_values, 98.8, float(np.percentile(ha_raw, 99.0))) if ha_values.size else 1e-6
        oiii_high = _safe_percentile(oiii_values, 98.8, float(np.percentile(oiii_raw, 99.0))) if oiii_values.size else 1e-6
        ha_mask = cv2.GaussianBlur((np.clip(ha_raw / max(ha_high, 1e-6), 0.0, 1.0) * signal_gate).astype(np.float32), (0, 0), 2.8)
        oiii_mask = cv2.GaussianBlur((np.clip(oiii_raw / max(oiii_high, 1e-6), 0.0, 1.0) * signal_gate).astype(np.float32), (0, 0), 2.8)
        oiii_present = float(np.percentile(oiii_mask[signal_gate > 0.16], 96.0)) if np.count_nonzero(signal_gate > 0.16) > 128 else 0.0
        if oiii_present < 0.055:
            oiii_mask *= 0.18

        output_lum = _luminance(output).astype(np.float32)
        measured_lum = _luminance(color_calibrated).astype(np.float32)
        linear_chroma = np.clip(color_calibrated - measured_lum[..., None], -1.0, 1.0)
        soft_color = cv2.GaussianBlur(color_calibrated.astype(np.float32), (0, 0), 1.6)
        soft_lum = _luminance(soft_color).astype(np.float32)
        soft_chroma = np.clip(soft_color - soft_lum[..., None], -1.0, 1.0)
        broad_color = cv2.GaussianBlur(color_calibrated.astype(np.float32), (0, 0), 5.5)
        broad_lum = _luminance(broad_color).astype(np.float32)
        broad_chroma = np.clip(broad_color - broad_lum[..., None], -1.0, 1.0)
        measured_chroma = np.clip(linear_chroma * 0.68 + soft_chroma * 0.26 + broad_chroma * 0.06, -1.0, 1.0)

        color_permission = np.clip(
            (nebula_mask * 0.30 + ridge * 0.30 + fine_detail * 0.24 + color_signal * 0.24)
            * support
            * (0.58 + filament_snr * 0.42)
            * (1.0 - clean_sky * 0.98)
            * (1.0 - star_protect * 0.90),
            0.0,
            1.0,
        )
        color_permission = cv2.GaussianBlur(color_permission.astype(np.float32), (0, 0), 2.1)

        colorfulness = np.max(np.abs(measured_chroma), axis=2)
        color_floor = _safe_percentile(colorfulness[color_permission > 0.10], 42.0, float(np.percentile(colorfulness, 55.0))) if np.count_nonzero(color_permission > 0.10) else float(np.percentile(colorfulness, 55.0))
        color_ceiling = _safe_percentile(colorfulness[color_permission > 0.10], 99.4, float(np.percentile(colorfulness, 99.4))) if np.count_nonzero(color_permission > 0.10) else float(np.percentile(colorfulness, 99.4))
        chroma_snr = np.clip((colorfulness - color_floor) / max(1e-6, color_ceiling - color_floor), 0.0, 1.0)
        chroma_snr = cv2.GaussianBlur(chroma_snr.astype(np.float32), (0, 0), 1.2)

        saturation_mask = np.clip(
            color_permission
            * (0.52 + chroma_snr * 0.48)
            * (1.0 - clean_sky * 0.94)
            * (1.0 - star_protect * 0.92),
            0.0,
            1.0,
        )
        texture_gate = np.clip(
            0.34
            + ridge * 0.44
            + fine_detail * 0.48
            + color_signal * 0.20
            + nebula_core * 0.12,
            0.28,
            1.0,
        )
        texture_gate = cv2.GaussianBlur(texture_gate.astype(np.float32), (0, 0), 1.4)
        saturation_mask = np.clip(saturation_mask * texture_gate, 0.0, 1.0)

        saturation_gain = 1.0 + saturation_mask[..., None] * (1.55 + separation_strength * 3.75)
        enhanced_chroma = measured_chroma * saturation_gain
        chroma_extent = np.max(np.abs(enhanced_chroma), axis=2)
        chroma_limit = np.clip(
            0.072
            + nebula_mask * 0.185
            + ridge * 0.170
            + fine_detail * 0.130
            + chroma_snr * 0.135,
            0.075,
            0.520,
        )
        chroma_limit = cv2.GaussianBlur(chroma_limit.astype(np.float32), (0, 0), 2.1)
        chroma_rolloff = np.tanh(chroma_extent / np.maximum(chroma_limit, 1e-5)) * chroma_limit / np.maximum(chroma_extent, 1e-5)
        enhanced_chroma *= chroma_rolloff[..., None]

        base_chroma = output - output_lum[..., None]
        composite_chroma = np.clip(base_chroma * 0.34 + enhanced_chroma * 0.66, -1.0, 1.0)
        target = np.clip(output_lum[..., None] + composite_chroma, 0.0, 1.0)
        target_lum = _luminance(target).astype(np.float32)
        target = np.clip(target * (output_lum / np.maximum(target_lum, 1e-5))[..., None], 0.0, 1.0)
        blend_mask = np.clip(saturation_mask * (0.76 + separation_strength * 0.56), 0.0, 0.95)
        output = np.clip(output * (1.0 - blend_mask[..., None]) + target * blend_mask[..., None], 0.0, 1.0)

        output_lum = _luminance(output).astype(np.float32)
        hsv = cv2.cvtColor(output.astype(np.float32), cv2.COLOR_RGB2HSV)
        hue = hsv[..., 0]
        sat = hsv[..., 1]
        val = hsv[..., 2]
        vibrance_mask = np.clip(
            saturation_mask
            * (0.38 + fine_detail * 0.30 + ridge * 0.22 + chroma_snr * 0.22)
            * (1.0 - clean_sky * 0.96)
            * (1.0 - star_protect * 0.92),
            0.0,
            1.0,
        )
        vibrance_mask = cv2.GaussianBlur(vibrance_mask.astype(np.float32), (0, 0), 1.6)
        vibrance_strength = np.clip(0.58 + separation_strength * 1.70, 0.58, 1.90)
        sat_boost = np.clip((1.0 - sat) * vibrance_mask * vibrance_strength, 0.0, 0.32)
        sat = np.clip(sat + sat_boost, 0.0, 1.0)
        val = np.clip(val + (1.0 - val) * vibrance_mask * (0.018 + separation_strength * 0.026), 0.0, 1.0)
        output = cv2.cvtColor(np.dstack([hue, sat, val]).astype(np.float32), cv2.COLOR_HSV2RGB)
        vib_lum = _luminance(output).astype(np.float32)
        output = np.clip(output * (output_lum / np.maximum(vib_lum, 1e-5))[..., None], 0.0, 1.0)
        output_lum = _luminance(output).astype(np.float32)
        output_chroma = output - output_lum[..., None]
        sky_desat = np.clip(clean_sky * (1.0 - nebula_mask * 0.92) * (0.24 + 0.10 * separation_strength), 0.0, 0.42)
        output = np.clip(output_lum[..., None] + (output - output_lum[..., None]) * (1.0 - sky_desat[..., None]), 0.0, 1.0)
        if separation_strength >= 0.64:
            output_lum = _luminance(output).astype(np.float32)
            highpass_fine = output_lum - cv2.GaussianBlur(output_lum, (0, 0), 1.05)
            highpass_small = cv2.GaussianBlur(output_lum, (0, 0), 1.8) - cv2.GaussianBlur(output_lum, (0, 0), 5.8)
            highpass_mid = cv2.GaussianBlur(output_lum, (0, 0), 4.8) - cv2.GaussianBlur(output_lum, (0, 0), 15.0)
            structure_pop = np.clip(
                (ridge * 0.46 + fine_detail * 0.42 + nebula_core * 0.32 + color_signal * 0.18)
                * support
                * (1.0 - clean_sky * 0.98)
                * (1.0 - star_protect * 0.94),
                0.0,
                1.0,
            )
            structure_pop = cv2.GaussianBlur(structure_pop.astype(np.float32), (0, 0), 0.9)
            sharpened_lum = np.clip(
                output_lum
                + highpass_fine * structure_pop * 0.22
                + highpass_small * structure_pop * 0.62
                + highpass_mid * nebula_mask * (1.0 - clean_sky * 0.95) * 0.26,
                0.0,
                1.0,
            )
            # Keep the sharpened nebula from turning crunchy by rolling off the strongest highlights.
            highlight_guard = np.clip((output_lum - 0.58) / 0.32, 0.0, 1.0)
            sharpened_lum = np.clip(sharpened_lum * (1.0 - highlight_guard * 0.10) + output_lum * (highlight_guard * 0.10), 0.0, 1.0)
            output = np.clip(output * (sharpened_lum / np.maximum(output_lum, 1e-5))[..., None], 0.0, 1.0)

        ha_mean = float(np.mean(ha_mask * signal_gate))
        oiii_mean = float(np.mean(oiii_mask * signal_gate))

        if _is_showcase_hoo_mode(color_separation):
            output, ha_mean, oiii_mean = _apply_showcase_hoo_nebula_grade(
                output,
                color_calibrated,
                nebula_mask,
                nebula_core,
                ridge,
                clean_sky,
                star_protect,
                support,
            )

    output_lum = _luminance(output).astype(np.float32)
    clean_sky_lum = output_lum[clean_sky > 0.68]
    if clean_sky_lum.size >= 512:
        sky_median = float(np.median(clean_sky_lum))
        if sky_median > 0.070:
            sky_scale = np.clip(0.058 / max(sky_median, 1e-5), 0.34, 0.97)
            sky_compress = np.clip(clean_sky * (1.0 - nebula_mask * 0.82) * (1.0 - star_protect * 0.74), 0.0, 0.58)
            output = np.clip(
                output * (1.0 - sky_compress[..., None])
                + output * sky_scale * sky_compress[..., None],
                0.0,
                1.0,
            )
            output_lum = _luminance(output).astype(np.float32)

    if _is_showcase_hoo_mode(color_separation):
        sky_target = np.array([0.044, 0.056, 0.064], dtype=np.float32).reshape(1, 1, 3)
        sky_darken_strength = 0.05
        edge_darken_strength = 0.24
    elif separation_strength >= 0.64:
        sky_target = np.array([0.024, 0.029, 0.034], dtype=np.float32).reshape(1, 1, 3)
        sky_darken_strength = 0.13
        edge_darken_strength = 0.36
    else:
        sky_target = np.array([0.018, 0.018, 0.019], dtype=np.float32).reshape(1, 1, 3)
        sky_darken_strength = 0.24
        edge_darken_strength = 0.48
    dark_sky = np.clip((0.075 - output_lum) / 0.075, 0.0, 1.0) * clean_sky
    output = np.clip(output * (1.0 - dark_sky[..., None] * sky_darken_strength) + sky_target * (dark_sky[..., None] * sky_darken_strength), 0.0, 1.0)
    sky_smooth = cv2.GaussianBlur(output.astype(np.float32), (0, 0), 2.8)
    sky_smooth_mix = np.clip(clean_sky * (1.0 - nebula_mask * 0.90) * (1.0 - star_protect * 0.86) * 0.42, 0.0, 0.42)
    output = np.clip(output * (1.0 - sky_smooth_mix[..., None]) + sky_smooth * sky_smooth_mix[..., None], 0.0, 1.0)

    star_high = max(float(np.percentile(star_lum, 99.92)), 1e-6)
    star_norm = np.clip(star_lum / star_high, 0.0, 1.0)
    if _is_showcase_hoo_mode(color_separation):
        star_weight = np.clip((star_norm - 0.095) / 0.905, 0.0, 1.0) ** 1.06
    else:
        star_weight = np.clip((star_norm - 0.055) / 0.945, 0.0, 1.0) ** 0.86
    star_color = np.clip(stars / np.maximum(star_lum[..., None], 1e-5), 0.58, 1.62)
    star_neutral = np.array([1.04, 0.99, 0.94], dtype=np.float32).reshape(1, 1, 3)
    star_color_mix = np.clip(star_weight[..., None] * 0.22, 0.0, 0.24)
    star_color = np.clip(star_color * (1.0 - star_color_mix) + star_neutral * star_color_mix, 0.58, 1.55)
    soft_lum = cv2.GaussianBlur(star_lum.astype(np.float32), (0, 0), 0.48)
    core_lum = np.minimum(star_lum, np.arcsinh(star_lum * 4.5) / np.arcsinh(4.5))
    processed_star_lum = np.clip(core_lum * 0.82 + soft_lum * 0.18, 0.0, 1.0) * star_weight
    processed_stars = np.clip(processed_star_lum[..., None] * star_color, 0.0, 1.0)
    processed_stars = cv2.GaussianBlur(processed_stars.astype(np.float32), (0, 0), 0.22)
    star_strength = float(np.clip(star_strength, 0.0, 1.0))
    if _is_showcase_hoo_mode(color_separation):
        star_strength *= 0.74
    final = 1.0 - (1.0 - output) * (1.0 - processed_stars * star_strength)

    final_lum = _luminance(final).astype(np.float32)
    hot = np.clip((final_lum - 0.86) / 0.14, 0.0, 1.0)
    final = np.clip(final / (1.0 + hot[..., None] * 0.08), 0.0, 1.0)
    edge_artifact = cv2.GaussianBlur(np.clip((1.0 - _edge_support((height, width), 0.13)) ** 1.1, 0.0, 1.0).astype(np.float32), (0, 0), 4.0)
    edge_mix = np.clip(edge_artifact[..., None] * (1.0 - nebula_mask[..., None] * 0.84) * edge_darken_strength, 0.0, 0.56)
    final = np.clip(final * (1.0 - edge_mix) + sky_target * edge_mix, 0.0, 1.0)

    if log:
        bg_after = (
            np.median(final[clean_sky > 0.68], axis=0)
            if np.count_nonzero(clean_sky > 0.68) > 512
            else np.median(final.reshape(-1, 3), axis=0)
        )
        log(
            "Composed DeepSky nebula enhancement layers: "
            f"star_strength={star_strength:.2f}; "
            f"color_separation={color_separation}; "
            f"color_separation_strength={separation_strength:.2f}; "
            f"bg_before_RGB={bg_before[0]:.5f},{bg_before[1]:.5f},{bg_before[2]:.5f}; "
            f"bg_after_RGB={bg_after[0]:.5f},{bg_after[1]:.5f},{bg_after[2]:.5f}; "
            f"ha_color_mean={ha_mean:.5f}; "
            f"oiii_color_mean={oiii_mean:.5f}; "
            f"nebula_mask_mean={float(np.mean(nebula_mask)):.5f}; "
            f"nebula_core_mean={float(np.mean(nebula_core)):.5f}; "
            f"star_mean={float(np.mean(star_weight)):.5f}; "
            f"clean_sky_mean={float(np.mean(clean_sky)):.5f}"
        )
    return _to_uint16(final)
