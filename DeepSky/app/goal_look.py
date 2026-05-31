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
    dark_mask = lum < np.percentile(lum, 45.0)
    if int(np.count_nonzero(dark_mask)) >= 128:
        sky = np.median(rgb[dark_mask], axis=0)
        neutral = float(np.mean(sky))
        gains = np.clip(neutral / np.maximum(sky, 1e-4), 0.90, 1.08)
        gains[1] = min(gains[1], 0.98)
        rgb = np.clip(rgb * gains.reshape(1, 1, 3), 0.0, 1.0)
    else:
        gains = np.ones(3, dtype=np.float32)

    lum = _luminance(rgb)
    rgb = np.clip(lum[..., None] + (rgb - lum[..., None]) * 0.55, 0.0, 1.0)

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
    star = np.clip(
        (lum - np.percentile(lum, 97.5))
        / max(1e-6, np.percentile(lum, 99.98) - np.percentile(lum, 97.5)),
        0.0,
        1.0,
    ) ** 1.7
    neutral_star = lum[..., None] + np.clip(rgb - lum[..., None], -0.06, 0.06)
    rgb = rgb * (1.0 - star[..., None] * 0.42) + neutral_star * (star[..., None] * 0.42)

    lum = _luminance(rgb)
    final_black = float(np.percentile(lum, 22.0))
    rgb = np.clip((rgb - final_black) / max(1e-6, 1.0 - final_black), 0.0, 1.0)
    if log:
        final_lum = _luminance(rgb)
        log(
            "Applied DeepSky broadband look: "
            f"black={black:.5f}, white={white:.5f}, final_black={final_black:.5f}, "
            f"sky_gains={gains[0]:.3f}, {gains[1]:.3f}, {gains[2]:.3f}, "
            f"median_luminance={np.median(final_lum):.5f}, chroma_p95={chroma_percentile(rgb, 95.0):.5f}"
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
        gains = np.clip(neutral / np.maximum(sky, 1e-4), 0.90, 1.12)
        gains[1] = min(gains[1], 0.88)
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
    extended = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 13.0)
    galaxy_mask = np.clip(
        (extended - np.percentile(extended, 48.0))
        / max(1e-6, np.percentile(extended, 98.8) - np.percentile(extended, 48.0)),
        0.0,
        1.0,
    ) ** 0.50
    star_mask = np.clip(
        (lum - np.percentile(lum, 97.1))
        / max(1e-6, np.percentile(lum, 99.98) - np.percentile(lum, 97.1)),
        0.0,
        1.0,
    ) ** 1.55
    protect = cv2.GaussianBlur(np.maximum(galaxy_mask, star_mask).astype(np.float32), (0, 0), 2.8)
    clean_sky = np.clip(1.0 - protect, 0.0, 1.0)
    clean_sky *= np.clip(
        (np.percentile(lum, 70.0) - lum)
        / max(1e-6, np.percentile(lum, 70.0) - np.percentile(lum, 1.0)),
        0.0,
        1.0,
    )
    clean_sky = cv2.GaussianBlur(clean_sky.astype(np.float32), (0, 0), 5.0)

    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    smooth_rgb = cv2.bilateralFilter(rgb8, d=0, sigmaColor=24, sigmaSpace=22).astype(np.float32) / 255.0
    smooth_rgb = cv2.GaussianBlur(smooth_rgb, (0, 0), 0.9)
    smooth_lum = _luminance(smooth_rgb)
    original_lum = _luminance(rgb)
    smooth_rgb = np.clip(smooth_rgb * (original_lum / np.maximum(smooth_lum, 1e-5))[..., None], 0.0, 1.0)
    smooth_mix = np.clip(clean_sky[..., None] * 0.62, 0.0, 0.68)
    rgb = np.clip(rgb * (1.0 - smooth_mix) + smooth_rgb * smooth_mix, 0.0, 1.0)

    lum = _luminance(rgb)
    background_pixels = (clean_sky > 0.48) & (lum < np.percentile(lum, 68.0))
    if int(np.count_nonzero(background_pixels)) >= 512:
        floor = float(np.percentile(lum[background_pixels], 32.0))
    else:
        floor = float(np.percentile(lum, 6.0))
    darker = np.clip((rgb - floor * 0.34) / max(1e-6, 1.0 - floor * 0.34), 0.0, 1.0)
    dark_mix = np.clip(clean_sky[..., None] * 0.46, 0.0, 0.52)
    rgb = np.clip(rgb * (1.0 - dark_mix) + darker * dark_mix, 0.0, 1.0)

    lum = _luminance(rgb)
    arm_signal = np.clip(
        (galaxy_mask - 0.10)
        / max(1e-6, 0.86),
        0.0,
        1.0,
    ) ** 0.72
    sky_signal = np.clip(1.0 - arm_signal, 0.0, 1.0)

    # Pull pre-stretched broadband galaxies toward a quieter natural palette:
    # warm core, blue-gray arms, and low-chroma dark sky.
    neutral = lum[..., None]
    saturation_scale = 0.34 + 0.34 * arm_signal[..., None] + 0.18 * star_mask[..., None]
    rgb = np.clip(neutral + (rgb - neutral) * saturation_scale, 0.0, 1.0)

    warm_core = np.clip(
        (lum - np.percentile(lum, 82.0))
        / max(1e-6, np.percentile(lum, 99.5) - np.percentile(lum, 82.0)),
        0.0,
        1.0,
    ) ** 1.15
    core_tint = np.array([1.08, 1.00, 0.86], dtype=np.float32).reshape(1, 1, 3)
    rgb = np.clip(rgb * (1.0 - warm_core[..., None] * 0.20) + neutral * core_tint * (warm_core[..., None] * 0.20), 0.0, 1.0)

    arm_tint = np.array([0.91, 0.99, 1.08], dtype=np.float32).reshape(1, 1, 3)
    rgb = np.clip(rgb * (1.0 - arm_signal[..., None] * 0.12) + neutral * arm_tint * (arm_signal[..., None] * 0.12), 0.0, 1.0)

    sky_tint = np.array([0.82, 0.80, 0.90], dtype=np.float32).reshape(1, 1, 3)
    dark_sky = np.clip(clean_sky * sky_signal, 0.0, 1.0)
    rgb = np.clip(rgb * (1.0 - dark_sky[..., None] * 0.28) + neutral * sky_tint * (dark_sky[..., None] * 0.28), 0.0, 1.0)

    lum = _luminance(rgb)
    background_pixels = (clean_sky > 0.46) & (galaxy_mask < 0.24) & (lum < np.percentile(lum, 72.0))
    if int(np.count_nonzero(background_pixels)) >= 512:
        final_floor = float(np.percentile(lum[background_pixels], 46.0))
    else:
        final_floor = float(np.percentile(lum, 8.0))
    final_dark = np.clip((rgb - final_floor * 0.66) / max(1e-6, 1.0 - final_floor * 0.66), 0.0, 1.0)
    final_dark *= 0.88 + 0.12 * arm_signal[..., None]
    final_mix = np.clip(clean_sky[..., None] * (0.56 + 0.20 * sky_signal[..., None]), 0.0, 0.72)
    rgb = np.clip(rgb * (1.0 - final_mix) + final_dark * final_mix, 0.0, 1.0)

    lum = _luminance(rgb)
    highlight = np.clip(
        (lum - np.percentile(lum, 88.0))
        / max(1e-6, np.percentile(lum, 99.75) - np.percentile(lum, 88.0)),
        0.0,
        1.0,
    )
    compressed_lum = lum / (1.0 + highlight * 0.58 * lum)
    compressed_lum = np.clip(compressed_lum * (1.0 + highlight * 0.10), 0.0, 1.0)
    compression_mix = np.clip(highlight[..., None] * (0.58 + 0.18 * warm_core[..., None]), 0.0, 0.72)
    compressed_rgb = np.clip(rgb * (compressed_lum / np.maximum(lum, 1e-5))[..., None], 0.0, 1.0)
    rgb = np.clip(rgb * (1.0 - compression_mix) + compressed_rgb * compression_mix, 0.0, 1.0)

    if log:
        log(
            "Applied pre-stretched broadband look: "
            f"black={black:.5f}, white={white:.5f}, sky_floor={sky_floor:.5f}, "
            f"sky_pixels={int(np.count_nonzero(background_pixels))}, "
            f"sky_gains={gains[0]:.3f}, {gains[1]:.3f}, {gains[2]:.3f}, "
            f"protect_mean={float(np.mean(protect)):.5f}, sky_mask_mean={float(np.mean(sky_mask)):.5f}, "
            f"clean_sky_mean={float(np.mean(clean_sky)):.5f}, final_floor={final_floor:.5f}"
        )
    return _to_uint16(rgb)


def apply_goal_look(image: np.ndarray, log: LogCallback | None = None) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return arr

    rgb = _to_float01(arr)
    lum = _luminance(rgb)
    black = float(np.percentile(lum, 15.0))
    white = float(np.percentile(lum, 99.7))
    rgb = np.clip((rgb - black) / max(1e-6, white - black), 0.0, 1.0)

    lum = _luminance(rgb)
    star_mask = np.clip(
        (lum - np.percentile(lum, 96.8))
        / max(1e-6, np.percentile(lum, 99.96) - np.percentile(lum, 96.8)),
        0.0,
        1.0,
    ) ** 1.8
    smoothed = cv2.bilateralFilter((rgb * 255.0).astype(np.uint8), 7, 45, 7).astype(np.float32) / 255.0
    rgb = rgb * (0.65 + 0.35 * star_mask[..., None]) + smoothed * (0.35 * (1.0 - star_mask[..., None]))

    rgb = np.clip(rgb, 0.0, 1.0) ** 0.50
    lum = _luminance(rgb)
    blurred = cv2.GaussianBlur(lum, (0, 0), 20)
    contrast_lum = np.clip(lum + (lum - blurred) * 0.20, 0.0, 1.0)
    rgb = np.clip(rgb * (contrast_lum / np.maximum(lum, 1e-5))[..., None], 0.0, 1.0)

    lum = _luminance(rgb)
    rgb = np.clip(lum[..., None] + (rgb - lum[..., None]) * 1.9, 0.0, 1.0)

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
    rgb[..., 0] += emission * 0.34
    rgb[..., 1] += emission * 0.041
    rgb[..., 2] -= emission * 0.044

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
    rgb = np.clip(rgb + white_nebula[..., None] * 0.58 * warm_white, 0.0, 1.0)
    lum = _luminance(rgb)
    cream = lum[..., None] * np.array([1.10, 1.02, 0.92], dtype=np.float32).reshape(1, 1, 3)
    white_mix = np.clip(white_nebula[..., None] * 0.55, 0.0, 0.65)
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
    final_black = float(np.percentile(lum, 5.0))
    rgb = np.clip((rgb - final_black) / max(1e-6, 1.0 - final_black), 0.0, 1.0)
    if log:
        chroma_95 = chroma_percentile(rgb, 95.0)
        final_lum = _luminance(rgb)
        log(
            "Applied DeepSky target look: "
            f"black={black:.5f}, white={white:.5f}, final_black={final_black:.5f}, "
            f"median_luminance={np.median(final_lum):.5f}, chroma_p95={chroma_95:.5f}"
        )
    return _to_uint16(rgb)
