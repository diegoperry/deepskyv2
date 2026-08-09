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
) -> np.ndarray:
    """Apply a signal-aware warm/cyan galaxy palette without changing luminance.

    The caller supplies a StarNet starless image so stellar color is handled by
    recomposition, not this function. Color evidence comes from the calibrated
    reference and is accepted only inside continuous extended-galaxy masks.
    """
    base = _float01(starless_image)
    reference = _float01(color_reference)
    if base.ndim != 3 or base.shape[-1] < 3 or reference.shape != base.shape:
        return np.asarray(starless_image)

    lum = _luminance(base).astype(np.float32)
    reference_lum = _luminance(reference).astype(np.float32)
    extended = (
        cv2.GaussianBlur(lum, (0, 0), 5.0) * 0.48
        + cv2.GaussianBlur(lum, (0, 0), 18.0) * 0.34
        + cv2.GaussianBlur(lum, (0, 0), 42.0) * 0.18
    )
    low = float(np.percentile(extended, 58.0))
    high = float(np.percentile(extended, 99.65))
    signal = np.clip((extended - low) / max(high - low, 1e-6), 0.0, 1.0) ** 0.72
    signal = cv2.GaussianBlur(signal.astype(np.float32), (0, 0), 5.0)

    core_low = float(np.percentile(extended, 94.0))
    core_high = float(np.percentile(extended, 99.92))
    core = np.clip((extended - core_low) / max(core_high - core_low, 1e-6), 0.0, 1.0) ** 1.25
    core = cv2.GaussianBlur(core.astype(np.float32), (0, 0), 3.0)

    reference_chroma = reference - reference_lum[..., None]
    chroma_size = np.max(reference, axis=2) - np.min(reference, axis=2)
    chroma_floor = float(np.percentile(chroma_size, 48.0))
    chroma_high = float(np.percentile(chroma_size, 99.2))
    chroma_confidence = np.clip(
        (chroma_size - chroma_floor) / max(chroma_high - chroma_floor, 1e-6),
        0.0,
        1.0,
    ) ** 0.70
    chroma_confidence = cv2.GaussianBlur(chroma_confidence.astype(np.float32), (0, 0), 1.4)

    # A continuous calibrated-RGB axis chooses warm versus cool placement. It
    # never creates hard regions and has zero influence in unmeasured sky.
    axis = reference[..., 0] - 0.5 * (reference[..., 1] + reference[..., 2])
    axis = cv2.GaussianBlur(axis.astype(np.float32), (0, 0), 1.2)
    axis_scale = float(np.percentile(np.abs(axis), 99.2))
    signed = np.tanh(axis / max(axis_scale * 0.52, 1e-5))
    warm = np.clip(signed, 0.0, 1.0)
    cool = np.clip(-signed, 0.0, 1.0)

    color_mask = np.clip(signal * chroma_confidence * (1.0 - core * 0.55), 0.0, 0.88)
    warm_rgb = np.stack((lum * 1.18, lum * 0.88, lum * 0.58), axis=2)
    cool_rgb = np.stack((lum * 0.56, lum * 1.02, lum * 1.18), axis=2)
    mapped = np.clip(
        warm_rgb * warm[..., None]
        + cool_rgb * cool[..., None]
        + (lum[..., None] + reference_chroma * 1.35) * (1.0 - np.maximum(warm, cool))[..., None],
        0.0,
        1.0,
    )
    colored = np.clip(base * (1.0 - color_mask[..., None]) + mapped * color_mask[..., None], 0.0, 1.0)

    # Neutralize only true low-signal background. Preserve the input luminance
    # after all chroma work so deconvolution detail and the nucleus are unchanged.
    background = np.clip((1.0 - signal * 2.25) * (1.0 - chroma_confidence * 0.75), 0.0, 1.0)
    neutral = lum[..., None] + (colored - lum[..., None]) * (1.0 - background[..., None] * 0.72)
    colored_lum = _luminance(neutral).astype(np.float32)
    output = np.clip(neutral * (lum / np.maximum(colored_lum, 1e-5))[..., None], 0.0, 1.0)

    # Do not let an already colorful broadband finish turn the palette into a
    # painted surface. The accepted chroma is bounded by the calibrated source
    # at each pixel as well as a conservative absolute ceiling.
    output_chroma = output - lum[..., None]
    measured_chroma = np.max(np.abs(reference_chroma), axis=2)
    chroma_limit = np.minimum(0.075, 0.014 + measured_chroma * 1.45)
    bounded_chroma = np.clip(
        output_chroma, -chroma_limit[..., None], chroma_limit[..., None]
    )
    bounded_chroma -= _luminance(bounded_chroma)[..., None]
    chroma_extent = np.max(np.abs(bounded_chroma), axis=2)
    headroom = np.minimum(lum, 1.0 - lum)
    headroom_scale = np.minimum(1.0, headroom / np.maximum(chroma_extent, 1e-6))
    output = np.clip(lum[..., None] + bounded_chroma * headroom_scale[..., None], 0.0, 1.0)

    if log:
        log(
            "Applied galaxy Narrowband Color to the StarNet starless layer: "
            f"signal_mean={float(np.mean(signal)):.5f}, "
            f"color_mask_mean={float(np.mean(color_mask)):.5f}, "
            f"color_mask_max={float(np.max(color_mask)):.5f}, "
            "deconvolved luminance preserved; stars excluded from grading."
        )
    return np.clip(output * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
