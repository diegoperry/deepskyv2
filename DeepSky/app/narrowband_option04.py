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

    # The morphological pass above shrinks bright cores, but a dense field can
    # still read as a white pointillist background because thousands of modest
    # stars never reach ``compact_high``. Remove a measured fraction of that
    # wider stellar population directly from luminance. StarNet remains a
    # detector only: the starless RGB canvas is never copied into the result.
    population_gate = _smoothstep(residual, residual_low * 0.20, residual_high * 0.42) * (1.0 - bright_core * 0.965)
    population_gate = np.clip(
        cv2.GaussianBlur(population_gate.astype(np.float32), (0, 0), 0.58)
        * edge_support,
        0.0,
        1.0,
    )
    population_lum = _luminance(result)
    measured_reduction = np.minimum(
        residual * population_gate * 1.12,
        population_lum * population_gate * 0.90,
    )
    population_reduced_lum = np.maximum(
        population_lum - measured_reduction,
        cv2.GaussianBlur(population_lum, (0, 0), 2.2) * 0.42,
    )
    result = np.clip(
        result
        * (population_reduced_lum / np.maximum(population_lum, 1e-6))[..., None],
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

    # Remove faint neutral pinpoints left behind by stellar reduction. A small
    # grayscale opening supplies the local sky level; the continuous mask
    # excludes extended nebula signal and protected bright stars.
    dot_lum = _luminance(result)
    opened_sky_lum = cv2.morphologyEx(
        dot_lum,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    dot_response = np.maximum(dot_lum - opened_sky_lum, 0.0)
    dot_samples = dot_response[sky > 0.55]
    dot_low = float(np.percentile(dot_samples, 48.0)) if dot_samples.size else 0.0
    dot_high = float(np.percentile(dot_samples, 99.5)) if dot_samples.size else max(dot_low + 1e-6, 0.01)
    neutral_dot = _smoothstep(dot_response, dot_low, max(dot_low + 1e-6, dot_high))
    result_chroma = np.max(result, axis=2) - np.min(result, axis=2)
    gray_confirmation = 1.0 - _smoothstep(result_chroma, 0.018, 0.072)
    dot_candidate = neutral_dot * gray_confirmation * sky * (1.0 - bright_halo)
    dot_seed = (
        (dot_candidate > 0.015)
        & (gray_confirmation > 0.15)
        & (sky > 0.25)
        & (bright_halo < 0.10)
    ).astype(np.float32)
    dot_footprint = cv2.dilate(
        dot_seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    dot_footprint = cv2.GaussianBlur(dot_footprint.astype(np.float32), (0, 0), 0.82)
    dot_mask = np.clip(
        np.maximum.reduce(
            [population_gate * sky, dot_candidate * 1.65, dot_footprint * 0.995]
        )
        * (1.0 - bright_halo * 0.995)
        * (1.0 - bright_core * 0.995),
        0.0,
        0.999,
    )
    cleaned_dot_lum = dot_lum * (1.0 - dot_mask) + np.minimum(dot_lum, opened_sky_lum) * dot_mask
    result = np.clip(
        result * (cleaned_dot_lum / np.maximum(dot_lum, 1e-6))[..., None],
        0.0,
        1.0,
    )
    result_lum = _luminance(result)
    # Anchor quiet sky to a dark warm-charcoal floor.
    target_lum = result_lum * 0.65 + 0.0060
    neutral = np.asarray([1.10, 0.91, 1.00], dtype=np.float32)
    neutral /= float(_luminance(neutral.reshape(1, 1, 3))[0, 0])
    neutral_rgb = target_lum[..., None] * neutral.reshape(1, 1, 3)
    lift_mix = np.clip(sky * 0.78, 0.0, 0.78)[..., None]
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
    ) * clarity_gate * 0.96
    clarified_lum = np.clip(clarity_lum + clarity_delta, 0.0, 1.0)
    result = np.clip(
        result * (clarified_lum / np.maximum(clarity_lum, 1e-6))[..., None],
        0.0,
        1.0,
    )

    # Give compact stars a clean photographic profile after the heavy reduction.
    # Smooth only the transition zone and colored fringe; retain the measured
    # central color so naturally blue and orange stars do not become white dots.
    profile_rgb = cv2.GaussianBlur(result, (0, 0), 0.62)
    profile_lum = _luminance(profile_rgb)
    current_lum = _luminance(result)
    profile_mix = np.clip(halo * 0.30, 0.0, 0.30)
    clean_lum = current_lum * (1.0 - profile_mix) + profile_lum * profile_mix
    profiled = np.clip(
        result * (clean_lum / np.maximum(current_lum, 1e-6))[..., None],
        0.0,
        1.0,
    )
    profiled_lum = _luminance(profiled)
    chroma = profiled - profiled_lum[..., None]
    smooth_chroma = cv2.GaussianBlur(chroma.astype(np.float32), (0, 0), 1.05)
    fringe = np.clip(halo * np.square(1.0 - core), 0.0, 1.0)
    fringe_mix = np.clip(fringe * 0.82, 0.0, 0.82)
    clean_chroma = (
        chroma * (1.0 - fringe_mix[..., None])
        + smooth_chroma * fringe_mix[..., None]
    )
    result = np.clip(profiled_lum[..., None] + clean_chroma, 0.0, 1.0)

    # Run the neutral-dot removal after every stellar profiling operation. This
    # is intentionally a terminal sky cleanup: compact stellar local maxima and
    # their halos are replaced by surrounding measured sky luminance, while
    # only the brightest stars and extended signal remain untouched.
    terminal_lum = _luminance(result)
    terminal_open = cv2.morphologyEx(
        terminal_lum,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
    )
    terminal_response = np.maximum(terminal_lum - terminal_open, 0.0)
    terminal_chroma = np.max(result, axis=2) - np.min(result, axis=2)
    terminal_seed = (
        (terminal_response > 0.0012)
        & (sky > 0.20)
        & (bright_halo < 0.12)
        & (bright_core < 0.10)
    ).astype(np.float32)
    terminal_mask = cv2.dilate(
        terminal_seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    terminal_mask = np.clip(
        cv2.GaussianBlur(terminal_mask.astype(np.float32), (0, 0), 0.78)
        * sky
        * (1.0 - bright_halo)
        * (1.0 - bright_core),
        0.0,
        0.999,
    )
    terminal_clean_lum = terminal_lum * (1.0 - terminal_mask) + np.minimum(terminal_lum, terminal_open) * terminal_mask
    result = np.clip(
        result * (terminal_clean_lum / np.maximum(terminal_lum, 1e-6))[..., None],
        0.0,
        1.0,
    )
    if log:
        output_lum = _luminance(result)
        before_peak = float(np.percentile(before_lum[safe], 99.95)) if np.any(safe) else float(np.max(before_lum))
        after_peak = float(np.percentile(output_lum[safe], 99.95)) if np.any(safe) else float(np.max(output_lum))
        log(
            "Narrowband Color: option 04 warm-charcoal sky and clean compact-star finish applied "
            f"(shrink_mean={float(np.mean(shrink)):.5f}, "
            f"sky_soft_mix_mean={float(np.mean(soft_mix)):.5f}, "
            f"clarity_gate_mean={float(np.mean(clarity_gate)):.5f}, "
            f"stellar_peak_p99.95={before_peak:.5f}->{after_peak:.5f})."
        )
    return result.astype(np.float32)
