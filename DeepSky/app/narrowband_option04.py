from __future__ import annotations

from typing import Callable

import cv2
import numpy as np


LogCallback = Callable[[str], None]


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    ).astype(np.float32)


def _smoothstep(values: np.ndarray, low: float, high: float) -> np.ndarray:
    scaled = np.clip((values - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def apply_option04_very_heavy_finish(
    polished: np.ndarray,
    display_source: np.ndarray,
    starnet_starless: np.ndarray,
    support: np.ndarray,
    log: LogCallback | None = None,
) -> np.ndarray:
    """Apply the validated option 04 star reduction and soft-sky finish.

    StarNet is used only to measure compact stellar residuals and broad nebula
    support. No pixels from the StarNet starless canvas enter the output.
    """
    result = np.clip(np.asarray(polished, dtype=np.float32), 0.0, 1.0)
    source = np.clip(np.asarray(display_source, dtype=np.float32), 0.0, 1.0)
    starless = np.clip(np.asarray(starnet_starless, dtype=np.float32), 0.0, 1.0)
    edge_support = np.clip(np.asarray(support, dtype=np.float32), 0.0, 1.0)
    safe = edge_support > 0.97

    source_lum = _luminance(source)
    starless_lum = _luminance(starless)
    residual = np.maximum(source_lum - starless_lum, 0.0)
    compact = np.maximum(
        cv2.GaussianBlur(residual, (0, 0), 0.68)
        - cv2.GaussianBlur(residual, (0, 0), 3.0),
        0.0,
    )
    compact_values = compact[safe] if np.any(safe) else compact.reshape(-1)
    residual_values = residual[safe] if np.any(safe) else residual.reshape(-1)
    compact_low = float(np.percentile(compact_values, 78.0))
    compact_high = max(compact_low + 1e-6, float(np.percentile(compact_values, 99.76)))
    residual_low = float(np.percentile(residual_values, 70.0))
    residual_high = max(residual_low + 1e-6, float(np.percentile(residual_values, 99.82)))
    core = np.sqrt(
        np.clip(
            _smoothstep(compact, compact_low, compact_high)
            * _smoothstep(residual, residual_low, residual_high),
            0.0,
            1.0,
        )
    )
    core = np.clip(
        cv2.GaussianBlur(core.astype(np.float32), (0, 0), 0.72) * edge_support,
        0.0,
        1.0,
    )
    halo = np.clip(cv2.GaussianBlur(core, (0, 0), 1.8) * 1.25, 0.0, 1.0)
    source_values = source_lum[safe] if np.any(safe) else source_lum.reshape(-1)
    bright_low = float(np.percentile(source_values, 98.6))
    bright_high = max(bright_low + 1e-6, float(np.percentile(source_values, 99.92)))
    bright_core = _smoothstep(source_lum, bright_low, bright_high)
    bright_halo = np.maximum(
        bright_core,
        cv2.GaussianBlur(bright_core.astype(np.float32), (0, 0), 2.2) * 3.0,
    )
    bright_halo = np.clip(
        bright_halo * edge_support,
        0.0,
        1.0,
    )
    shrink = np.clip(core * 0.78 + halo * 0.78 * 0.36, 0.0, 0.88)

    before_lum = _luminance(result)
    eroded_lum = cv2.erode(
        before_lum,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    local_floor = cv2.GaussianBlur(before_lum, (0, 0), 2.5) * 0.76
    reduced_lum = np.maximum(
        before_lum * (1.0 - shrink) + eroded_lum * shrink,
        local_floor,
    )
    result = np.clip(
        result * (reduced_lum / np.maximum(before_lum, 1e-6))[..., None],
        0.0,
        1.0,
    )

    broad_starless = cv2.GaussianBlur(starless, (0, 0), 7.0)
    broad_lum = _luminance(broad_starless)
    broad_chroma = np.max(broad_starless, axis=2) - np.min(broad_starless, axis=2)
    safe_lum = broad_lum[safe] if np.any(safe) else broad_lum.reshape(-1)
    safe_chroma = broad_chroma[safe] if np.any(safe) else broad_chroma.reshape(-1)
    signal = np.maximum(
        _smoothstep(
            broad_lum,
            float(np.percentile(safe_lum, 58.0)),
            float(np.percentile(safe_lum, 98.2)),
        ),
        _smoothstep(
            broad_chroma,
            float(np.percentile(safe_chroma, 65.0)),
            float(np.percentile(safe_chroma, 99.0)),
        )
        * 0.90,
    )
    signal = np.clip(
        cv2.GaussianBlur(signal.astype(np.float32), (0, 0), 2.0) * edge_support,
        0.0,
        1.0,
    )
    stellar_protection = np.maximum(halo, bright_halo)
    sky = np.clip(
        (1.0 - signal) * (1.0 - stellar_protection) * edge_support,
        0.0,
        1.0,
    )
    smooth_sky = cv2.GaussianBlur(result, (0, 0), 1.77)
    soft_mix = np.clip(sky * 0.62, 0.0, 0.72)[..., None]
    result = result * (1.0 - soft_mix) + smooth_sky * soft_mix

    result_lum = _luminance(result)
    rolloff = np.square(np.clip(1.0 - result_lum / 0.11, 0.0, 1.0))
    target_lum = result_lum + 0.012 * rolloff
    neutral = np.asarray([0.96, 1.00, 1.05], dtype=np.float32)
    neutral /= float(_luminance(neutral.reshape(1, 1, 3))[0, 0])
    neutral_rgb = target_lum[..., None] * neutral.reshape(1, 1, 3)
    lift_mix = np.clip(sky * 0.72, 0.0, 0.72)[..., None]
    result = np.clip(
        result * (1.0 - lift_mix) + neutral_rgb * lift_mix,
        0.0,
        1.0,
    )

    # Restore restrained filament-scale contrast after the soft-sky pass.
    # Work only in measured extended signal and exclude stellar footprints, so
    # the added clarity cannot amplify empty-sky noise or create hard star rings.
    clarity_lum = _luminance(result)
    fine_structure = (
        cv2.GaussianBlur(clarity_lum, (0, 0), 0.65)
        - cv2.GaussianBlur(clarity_lum, (0, 0), 2.0)
    )
    medium_structure = (
        cv2.GaussianBlur(clarity_lum, (0, 0), 1.4)
        - cv2.GaussianBlur(clarity_lum, (0, 0), 5.0)
    )
    clarity_gate = np.clip(signal * np.square(1.0 - stellar_protection), 0.0, 1.0)
    clarity_delta = (
        fine_structure * 0.55 + medium_structure * 0.32
    ) * clarity_gate * 0.48
    clarified_lum = np.clip(clarity_lum + clarity_delta, 0.0, 1.0)
    result = np.clip(
        result * (clarified_lum / np.maximum(clarity_lum, 1e-6))[..., None],
        0.0,
        1.0,
    )

    if log:
        output_lum = _luminance(result)
        before_peak = float(np.percentile(before_lum[safe], 99.95)) if np.any(safe) else float(np.max(before_lum))
        after_peak = float(np.percentile(output_lum[safe], 99.95)) if np.any(safe) else float(np.max(output_lum))
        log(
            "Narrowband Color: option 04 very-heavy star reduction and soft-sky finish applied "
            f"(shrink_mean={float(np.mean(shrink)):.5f}, "
            f"sky_soft_mix_mean={float(np.mean(soft_mix)):.5f}, "
            f"clarity_gate_mean={float(np.mean(clarity_gate)):.5f}, "
            f"stellar_peak_p99.95={before_peak:.5f}->{after_peak:.5f})."
        )
    return result.astype(np.float32)
