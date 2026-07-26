from __future__ import annotations

from math import ceil
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


def keep_largest_narrowband_stars(
    polished: np.ndarray,
    display_source: np.ndarray,
    starnet_starless: np.ndarray,
    support: np.ndarray,
    *,
    keep_fraction: float = 0.10,
    log: LogCallback | None = None,
) -> np.ndarray:
    """Suppress measured compact stars except for the largest flux-ranked set.

    The positive display-minus-StarNet residual supplies detection and removal
    strength. StarNet RGB pixels are never copied into the output.
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
        cv2.GaussianBlur(residual, (0, 0), 0.65)
        - cv2.GaussianBlur(residual, (0, 0), 3.0),
        0.0,
    )
    compact_values = compact[safe] if np.any(safe) else compact.reshape(-1)
    residual_values = residual[safe] if np.any(safe) else residual.reshape(-1)
    compact_low = float(np.percentile(compact_values, 72.0))
    compact_high = max(compact_low + 1e-6, float(np.percentile(compact_values, 99.75)))
    residual_low = float(np.percentile(residual_values, 68.0))
    residual_high = max(residual_low + 1e-6, float(np.percentile(residual_values, 99.82)))
    confidence = np.sqrt(
        np.clip(
            _smoothstep(compact, compact_low, compact_high)
            * _smoothstep(residual, residual_low, residual_high),
            0.0,
            1.0,
        )
    )
    seed = (
        (confidence > 0.085)
        & (residual > max(residual_low, 0.0012))
        & (edge_support > 0.72)
    ).astype(np.uint8)
    seed = cv2.morphologyEx(
        seed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        seed,
        connectivity=8,
    )
    if component_count <= 2:
        if log:
            log("Narrowband sparse-star finish skipped: too few reliable stellar components.")
        return result

    component_ids = np.arange(1, component_count, dtype=np.int32)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    flux = np.bincount(
        labels.reshape(-1),
        weights=residual.reshape(-1),
        minlength=component_count,
    )[1:].astype(np.float32)
    # Integrated residual flux is the primary apparent-size measurement; the
    # footprint term resolves similarly bright compact stars.
    scores = flux * np.sqrt(np.maximum(areas, 1.0))
    keep_count = max(1, int(ceil(component_ids.size * np.clip(keep_fraction, 0.01, 1.0))))
    keep_ids = component_ids[np.argsort(scores)[-keep_count:]]
    keep_lookup = np.zeros(component_count, dtype=np.uint8)
    keep_lookup[keep_ids] = 1
    remove_lookup = np.ones(component_count, dtype=np.uint8)
    remove_lookup[0] = 0
    remove_lookup[keep_ids] = 0

    remove_seed = remove_lookup[labels].astype(np.float32)
    keep_seed = keep_lookup[labels].astype(np.float32)
    remove_footprint = cv2.dilate(
        remove_seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    remove_footprint = np.clip(
        cv2.GaussianBlur(remove_footprint, (0, 0), 1.40) * 1.85,
        0.0,
        1.0,
    )
    keep_protection = cv2.dilate(
        keep_seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    keep_protection = np.clip(
        cv2.GaussianBlur(keep_protection, (0, 0), 1.8) * 2.2,
        0.0,
        1.0,
    )
    remove_footprint *= (1.0 - keep_protection) * edge_support

    before_lum = _luminance(result)
    local_floor = cv2.morphologyEx(
        before_lum,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    local_floor = np.maximum(
        local_floor * 0.92,
        cv2.GaussianBlur(before_lum, (0, 0), 3.8) * 0.52,
    )
    removal_mix = np.clip(remove_footprint * 0.985, 0.0, 0.985)
    reduced_lum = (
        before_lum * (1.0 - removal_mix) + local_floor * removal_mix
    )
    output = np.clip(
        result * (reduced_lum / np.maximum(before_lum, 1e-6))[..., None],
        0.0,
        1.0,
    )

    if log:
        removed_count = int(component_ids.size - keep_count)
        before_peak = float(np.percentile(before_lum[safe], 99.9)) if np.any(safe) else float(np.max(before_lum))
        after_lum = _luminance(output)
        after_peak = float(np.percentile(after_lum[safe], 99.9)) if np.any(safe) else float(np.max(after_lum))
        log(
            "Narrowband sparse-star finish kept the largest 10% of measured stars "
            f"(detected={int(component_ids.size)}, kept={keep_count}, removed={removed_count}, "
            f"keep_fraction={keep_count / max(1, component_ids.size):.3f}, "
            f"stellar_peak_p99.9={before_peak:.5f}->{after_peak:.5f})."
        )
    return output.astype(np.float32)
