from __future__ import annotations

from typing import Callable

import cv2
import numpy as np


LogCallback = Callable[[str], None]


def _to_float01(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.float32) / float(np.iinfo(array.dtype).max)
    result = array.astype(np.float32)
    finite = result[np.isfinite(result)]
    if finite.size and float(np.max(finite)) > 1.5:
        result /= float(np.max(finite))
    return np.clip(np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    ).astype(np.float32)


def _smoothstep(value: np.ndarray, low: float, high: float) -> np.ndarray:
    x = np.clip((value - low) / max(high - low, 1e-8), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _edge_support(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    border = max(8, int(min(height, width) * 0.018))
    support = np.ones((height, width), dtype=np.float32)
    support[:border] = 0.0
    support[-border:] = 0.0
    support[:, :border] = 0.0
    support[:, -border:] = 0.0
    return cv2.GaussianBlur(support, (0, 0), max(2.0, border * 0.28))


def _chroma_denoise(rgb: np.ndarray, signal: np.ndarray, star_protect: np.ndarray) -> np.ndarray:
    """Denoise chroma only; keep the linked luminance and stellar RGB intact."""
    luminance = _luminance(rgb)
    chroma = rgb - luminance[..., None]
    smooth = np.empty_like(chroma)
    for channel in range(3):
        # A small bilateral pass removes CFA/checker color without smearing
        # genuine nebula boundaries. A wider Gaussian is used only in sky.
        bilateral = cv2.bilateralFilter(
            chroma[..., channel].astype(np.float32),
            d=0,
            sigmaColor=0.025,
            sigmaSpace=2.6,
        )
        wide = cv2.GaussianBlur(bilateral, (0, 0), 1.45)
        sky_mix = np.clip(0.88 - signal * 0.58, 0.28, 0.88)
        smooth[..., channel] = bilateral * (1.0 - sky_mix) + wide * sky_mix
    mix = np.clip((1.0 - star_protect) * (0.46 + (1.0 - signal) * 0.44), 0.0, 0.92)
    cleaned = luminance[..., None] + chroma * (1.0 - mix[..., None]) + smooth * mix[..., None]
    cleaned_lum = _luminance(cleaned)
    return np.clip(cleaned * (luminance / np.maximum(cleaned_lum, 1e-6))[..., None], 0.0, 1.0)


def apply_pixinsight_narrowband_finish(
    linear_image: np.ndarray,
    log: LogCallback | None = None,
) -> np.ndarray:
    """Finish a stacked RGB/duoband master with a conservative PixInsight-style flow.

    This is intentionally not a synthetic SHO mapper. It performs a linked
    luminance stretch, measures only spatially coherent red/cyan separation,
    neutralizes low-SNR chroma, protects stars, and applies local contrast only
    where extended signal is present.
    """
    source = _to_float01(linear_image)
    if source.ndim != 3 or source.shape[-1] < 3:
        return np.asarray(linear_image)
    source = source[..., :3]
    height, width = source.shape[:2]
    support = _edge_support((height, width))
    safe = support > 0.97

    source_lum = _luminance(source)
    safe_lum = source_lum[safe] if np.any(safe) else source_lum.reshape(-1)
    black = float(np.percentile(safe_lum, 10.0))
    sky_median = float(np.percentile(safe_lum, 42.0))
    highlight = max(black + 1e-6, float(np.percentile(safe_lum, 99.72)))

    # Detect compact stellar structure on the linear master. The broad halo is
    # deliberately larger than the visible core so neither palette nor local
    # contrast can create cyan/orange donuts around stars.
    # Suppress one/two-pixel CFA periodicity before star detection. Otherwise
    # alternating chroma can leak into luminance and every other pixel is
    # incorrectly protected as a tiny "star" from chroma cleanup.
    detection_lum = cv2.GaussianBlur(source_lum, (0, 0), 0.78)
    local = cv2.GaussianBlur(detection_lum, (0, 0), 2.0)
    positive_detail = np.maximum(detection_lum - local, 0.0)
    detail_scale = max(1e-7, float(np.percentile(positive_detail[safe], 99.55)))
    core = np.clip(positive_detail / detail_scale, 0.0, 1.0) ** 0.58
    bright_gate = _smoothstep(source_lum, float(np.percentile(safe_lum, 96.0)), highlight)
    star_core = np.maximum(core, bright_gate * 0.72)
    star_protect = cv2.GaussianBlur(star_core.astype(np.float32), (0, 0), 3.2)
    star_protect = np.clip(star_protect * 1.55, 0.0, 1.0)

    # Linked arcsinh stretch preserves channel ratios and avoids independent
    # channel stretching, one of the main causes of colored stellar halos.
    normalized_lum = np.clip((source_lum - black * 0.88) / max(highlight - black * 0.88, 1e-7), 0.0, 1.0)
    stretch = np.arcsinh(normalized_lum * 6.2) / np.arcsinh(6.2)
    # Keep a visible, non-clipped sky floor while preventing star cores from
    # being inflated by the faint-nebula stretch.
    sky_floor = 0.014
    target_lum = sky_floor + stretch * (1.0 - sky_floor)
    star_target = sky_floor + np.arcsinh(normalized_lum * 2.8) / np.arcsinh(2.8) * (1.0 - sky_floor)
    target_lum = target_lum * (1.0 - star_protect * 0.72) + star_target * (star_protect * 0.72)
    normalized_rgb = np.clip((source - black * 0.88) / max(highlight - black * 0.88, 1e-7), 0.0, 1.0)
    prefilter_lum = _luminance(normalized_rgb)
    prefilter_chroma = normalized_rgb - prefilter_lum[..., None]
    smooth_chroma = cv2.GaussianBlur(prefilter_chroma.astype(np.float32), (0, 0), 0.86)
    periodic_mix = np.clip((1.0 - star_protect) * 0.92, 0.0, 0.92)
    normalized_rgb = np.clip(
        prefilter_lum[..., None]
        + prefilter_chroma * (1.0 - periodic_mix[..., None])
        + smooth_chroma * periodic_mix[..., None],
        0.0,
        1.0,
    )
    normalized_rgb_lum = _luminance(normalized_rgb)
    stretched = np.clip(
        normalized_rgb * (target_lum / np.maximum(normalized_rgb_lum, 1e-6))[..., None],
        0.0,
        1.0,
    )

    broad = cv2.GaussianBlur(target_lum, (0, 0), max(8.0, min(height, width) * 0.010))
    broad_safe = broad[safe] if np.any(safe) else broad.reshape(-1)
    signal_low = float(np.percentile(broad_safe, 46.0))
    signal_high = max(signal_low + 1e-6, float(np.percentile(broad_safe, 98.8)))
    signal = _smoothstep(broad, signal_low, signal_high) ** 0.62
    signal = cv2.GaussianBlur(signal.astype(np.float32), (0, 0), 2.0) * support

    stretched = _chroma_denoise(stretched, signal, star_protect)
    lum = _luminance(stretched)

    # Measure H-alpha/OIII tendencies only after low-pass chroma cleanup.
    # No hard region masks and no color assignment from luminance alone.
    color_base = cv2.GaussianBlur(stretched.astype(np.float32), (0, 0), 2.2)
    red = color_base[..., 0]
    green = color_base[..., 1]
    blue = color_base[..., 2]
    ha_raw = np.maximum(red - (green + blue) * 0.50, 0.0)
    oiii_raw = np.maximum((green + blue) * 0.50 - red, 0.0)
    quiet = (signal < 0.10) & (star_protect < 0.05) & safe
    active = (signal > 0.16) & (star_protect < 0.22) & safe
    if np.count_nonzero(quiet) < 256:
        quiet = safe
    if np.count_nonzero(active) < 256:
        active = safe
    ha_noise = float(np.percentile(ha_raw[quiet], 92.0))
    oiii_noise = float(np.percentile(oiii_raw[quiet], 92.0))
    ha_scale = max(ha_noise + 1e-6, float(np.percentile(ha_raw[active], 99.0)))
    oiii_scale = max(oiii_noise + 1e-6, float(np.percentile(oiii_raw[active], 99.0)))
    ha = _smoothstep(ha_raw, ha_noise, ha_scale)
    oiii = _smoothstep(oiii_raw, oiii_noise, oiii_scale)
    coherent_color = cv2.GaussianBlur(np.maximum(ha, oiii).astype(np.float32), (0, 0), 1.4)
    color_gate = np.clip(signal * coherent_color * (1.0 - star_protect * 0.96), 0.0, 0.78)

    warm = np.array([1.00, 0.47, 0.15], dtype=np.float32)
    cool = np.array([0.13, 0.69, 1.00], dtype=np.float32)
    weight_sum = ha + oiii + 1e-5
    palette = (ha[..., None] * warm + oiii[..., None] * cool) / weight_sum[..., None]
    palette_lum = _luminance(palette)
    palette = np.clip(palette * (lum / np.maximum(palette_lum, 1e-6))[..., None], 0.0, 1.0)
    result = np.clip(stretched * (1.0 - color_gate[..., None]) + palette * color_gate[..., None], 0.0, 1.0)

    # Neutralize background chroma continuously. This is the equivalent of a
    # protected background-neutralization/chrominance NR pass, not a black mask.
    result_lum = _luminance(result)
    background_gate = cv2.GaussianBlur(
        np.clip((0.18 - signal) / 0.16, 0.0, 1.0).astype(np.float32),
        (0, 0),
        2.0,
    ) * (1.0 - star_protect)
    result = np.clip(
        result_lum[..., None]
        + (result - result_lum[..., None]) * (1.0 - background_gate[..., None] * 0.995),
        0.0,
        1.0,
    )

    # Multiscale luminance contrast is confined to coherent nebula and excluded
    # from stars/background, preventing crispy noise and dark stellar rings.
    result_lum = _luminance(result)
    fine = result_lum - cv2.GaussianBlur(result_lum, (0, 0), 1.4)
    medium = cv2.GaussianBlur(result_lum, (0, 0), 3.0) - cv2.GaussianBlur(result_lum, (0, 0), 11.0)
    detail_gate = signal * (1.0 - star_protect) * support
    enhanced_lum = np.clip(result_lum + (fine * 0.12 + medium * 0.20) * detail_gate, 0.0, 1.0)
    result = np.clip(result * (enhanced_lum / np.maximum(result_lum, 1e-6))[..., None], 0.0, 1.0)

    # Soft highlight compression prevents white clipping while retaining star
    # color. It does not enlarge the stellar footprint.
    peak = np.max(result, axis=2)
    hot = _smoothstep(peak, 0.82, 1.0)
    compressed = result / (1.0 + result * 0.32) * 1.14
    result = np.clip(result * (1.0 - hot[..., None] * 0.55) + compressed * (hot[..., None] * 0.55), 0.0, 1.0)

    if log:
        log(
            "Narrowband Color: PixInsight-style linked stretch, chroma-only denoise, "
            "continuous measured HOO separation, protected multiscale contrast, and star preservation applied "
            f"(black={black:.6f}, sky={sky_median:.6f}, highlight={highlight:.6f}, "
            f"signal_mean={float(np.mean(signal)):.5f}, color_gate_mean={float(np.mean(color_gate)):.5f})."
        )
    return np.clip(np.rint(result * 65535.0), 0, 65535).astype(np.uint16)


def apply_processed_narrowband_color_finish(
    processed_image: np.ndarray,
    log: LogCallback | None = None,
    *,
    star_layer: np.ndarray | None = None,
) -> np.ndarray:
    """Apply Narrowband Color after the normal finished-nebula pipeline.

    The incoming display-referred luminance remains authoritative. Only
    coherent, low-frequency measured chroma is graded; sky and stellar pixels
    are protected continuously.
    """
    rgb = _to_float01(processed_image)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return np.asarray(processed_image)
    rgb = rgb[..., :3]
    height, width = rgb.shape[:2]
    support = _edge_support((height, width))
    safe = support > 0.97
    luminance = _luminance(rgb)
    safe_lum = luminance[safe] if np.any(safe) else luminance.reshape(-1)

    detection = cv2.GaussianBlur(luminance, (0, 0), 0.72)
    local = cv2.GaussianBlur(detection, (0, 0), 2.2)
    stellar_detail = np.maximum(detection - local, 0.0)
    detail_scale = max(1e-6, float(np.percentile(stellar_detail[safe], 99.55)))
    star_core = np.clip(stellar_detail / detail_scale, 0.0, 1.0) ** 0.58
    star_core = np.maximum(
        star_core,
        _smoothstep(
            luminance,
            float(np.percentile(safe_lum, 97.2)),
            float(np.percentile(safe_lum, 99.92)),
        ) * 0.72,
    )
    if star_layer is not None:
        stars = _to_float01(star_layer)
        if stars.shape == rgb.shape:
            star_lum = np.max(stars, axis=2)
            scale = max(1e-6, float(np.percentile(star_lum, 99.90)))
            star_core = np.maximum(
                star_core,
                _smoothstep(star_lum, scale * 0.035, scale * 0.72),
            )
    star_protect = np.clip(
        cv2.GaussianBlur(star_core.astype(np.float32), (0, 0), 3.4) * 1.65,
        0.0,
        1.0,
    )

    broad = cv2.GaussianBlur(
        luminance,
        (0, 0),
        max(8.0, min(height, width) * 0.010),
    )
    broad_safe = broad[safe] if np.any(safe) else broad.reshape(-1)
    signal_low = float(np.percentile(broad_safe, 43.0))
    signal_high = max(signal_low + 1e-6, float(np.percentile(broad_safe, 98.6)))
    signal = cv2.GaussianBlur(
        (_smoothstep(broad, signal_low, signal_high) ** 0.66).astype(np.float32),
        (0, 0),
        1.8,
    ) * support

    chroma = rgb - luminance[..., None]
    chroma_smooth = cv2.GaussianBlur(chroma.astype(np.float32), (0, 0), 1.15)
    cleanup = np.clip(
        (1.0 - star_protect) * (0.90 - signal * 0.28),
        0.38,
        0.90,
    )
    cleaned = np.clip(
        luminance[..., None]
        + chroma * (1.0 - cleanup[..., None])
        + chroma_smooth * cleanup[..., None],
        0.0,
        1.0,
    )
    cleaned_lum = _luminance(cleaned)
    cleaned = np.clip(
        cleaned * (luminance / np.maximum(cleaned_lum, 1e-6))[..., None],
        0.0,
        1.0,
    )

    color_base = cv2.GaussianBlur(cleaned.astype(np.float32), (0, 0), 2.3)
    red, green, blue = color_base[..., 0], color_base[..., 1], color_base[..., 2]
    ha_raw = np.maximum(red - (green + blue) * 0.50, 0.0)
    oiii_raw = np.maximum((green + blue) * 0.50 - red, 0.0)
    quiet = (signal < 0.10) & (star_protect < 0.06) & safe
    active = (signal > 0.17) & (star_protect < 0.25) & safe
    if np.count_nonzero(quiet) < 256:
        quiet = safe
    if np.count_nonzero(active) < 256:
        active = safe
    ha = _smoothstep(
        ha_raw,
        float(np.percentile(ha_raw[quiet], 92.0)),
        max(1e-6, float(np.percentile(ha_raw[active], 99.0))),
    )
    oiii = _smoothstep(
        oiii_raw,
        float(np.percentile(oiii_raw[quiet], 92.0)),
        max(1e-6, float(np.percentile(oiii_raw[active], 99.0))),
    )
    coherent = cv2.GaussianBlur(
        np.maximum(ha, oiii).astype(np.float32),
        (0, 0),
        1.5,
    )
    color_gate = np.clip(
        signal * coherent * (1.0 - star_protect * 0.98),
        0.0,
        0.68,
    )

    warm = np.array([1.00, 0.47, 0.15], dtype=np.float32)
    cool = np.array([0.13, 0.69, 1.00], dtype=np.float32)
    total = ha + oiii + 1e-5
    palette = (ha[..., None] * warm + oiii[..., None] * cool) / total[..., None]
    palette_lum = _luminance(palette)
    palette = np.clip(
        palette * (luminance / np.maximum(palette_lum, 1e-6))[..., None],
        0.0,
        1.0,
    )
    result = np.clip(
        cleaned * (1.0 - color_gate[..., None])
        + palette * color_gate[..., None],
        0.0,
        1.0,
    )

    background = cv2.GaussianBlur(
        np.clip((0.16 - signal) / 0.14, 0.0, 1.0).astype(np.float32),
        (0, 0),
        2.0,
    ) * (1.0 - star_protect)
    result = np.clip(
        luminance[..., None]
        + (result - _luminance(result)[..., None])
        * (1.0 - background[..., None] * 0.985),
        0.0,
        1.0,
    )
    result = np.clip(
        result * (1.0 - star_protect[..., None] * 0.96)
        + rgb * (star_protect[..., None] * 0.96),
        0.0,
        1.0,
    )
    result_lum = _luminance(result)
    result = np.clip(
        result * (luminance / np.maximum(result_lum, 1e-6))[..., None],
        0.0,
        1.0,
    )

    if log:
        log(
            "Narrowband Color: applied after DeepSNR, StarNet, detail recovery, and star recombination; "
            f"finished luminance and stellar profiles preserved (signal_mean={float(np.mean(signal)):.5f}, "
            f"color_gate_mean={float(np.mean(color_gate)):.5f})."
        )
    return np.clip(np.rint(result * 65535.0), 0, 65535).astype(np.uint16)