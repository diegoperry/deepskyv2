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
        orange_target = lum[..., None] * np.array([1.58, 0.66, 0.28], dtype=np.float32).reshape(1, 1, 3)
        color_mix = np.clip(nebula_body[..., None] * (0.28 if stretch else 0.74), 0.0, 0.74)
        rgb = np.clip(rgb * (1.0 - color_mix) + orange_target * color_mix, 0.0, 1.0)
        rgb[..., 0] = np.clip(rgb[..., 0] + nebula_body * (0.05 if stretch else 0.17), 0.0, 1.0)
        rgb[..., 1] = np.clip(rgb[..., 1] + nebula_body * (0.012 if stretch else 0.026), 0.0, 1.0)
        rgb[..., 2] = np.clip(rgb[..., 2] - nebula_body * (0.010 if stretch else 0.050), 0.0, 1.0)

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
