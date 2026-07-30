from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .astrosharp_v2 import (
    AstroSharpPsfEstimate,
    blend_astrosharp_dual_structure,
    estimate_astrosharp_psf,
)


LogCallback = Callable[[str], None]
MODEL_BUNDLE = (
    Path(__file__).resolve().parent
    / "models"
    / "astrosharp_dualpsf_weights.npz"
)

# Values returned by the official GetMatrixFun9x9.R for the first valid pixel
# of an 11x11 matrix filled in R's column-major order. This preserves the
# upstream model's exact, non-row-major 81-feature order.
_REFERENCE_VALUES = np.asarray(
    [
        37, 48, 59, 38, 60, 39, 50, 61, 49,
        25, 36, 47, 58, 69, 26, 70, 27, 71, 28, 72, 29, 40, 51, 62, 73,
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        12, 13, 14, 15, 16, 17, 18, 19, 20,
        23, 24, 30, 31, 34, 35, 41, 42, 45, 46, 52, 53,
        97, 56, 57, 63, 64, 67, 68, 74, 75, 78, 79, 80, 81, 82, 83, 84,
        85, 86, 89, 90, 91, 92, 93, 94, 95, 96,
    ],
    dtype=np.int32,
)
_FEATURE_ROWS = ((_REFERENCE_VALUES - 1) % 11) + 1
_FEATURE_COLS = ((_REFERENCE_VALUES - 1) // 11) + 1
_FEATURE_OFFSETS = tuple(
    zip(
        (_FEATURE_ROWS - 5).tolist(),
        (_FEATURE_COLS - 5).tolist(),
    )
)


@dataclass(frozen=True)
class AstroSharpNativeResult:
    dso_donor: np.ndarray
    star_donor: np.ndarray
    psf: AstroSharpPsfEstimate
    manifest: dict[str, object]


def _to_float01(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.float32) / float(np.iinfo(array.dtype).max)
    return np.clip(
        np.nan_to_num(
            array.astype(np.float32),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ),
        0.0,
        1.0,
    )


def _model_key(psf: float) -> str:
    value = round(float(psf) * 4.0) / 4.0
    text = f"{value:g}".replace(".", "p")
    return f"psf_{text}"


def _load_weights(
    bundle: np.lib.npyio.NpzFile,
    psf: float,
) -> tuple[np.ndarray, ...]:
    prefix = _model_key(psf)
    weights = []
    for index in range(1, 6):
        matrix = bundle[f"{prefix}_w{index}"].astype(
            np.float32,
            copy=False,
        )
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        weights.append(matrix)
    return tuple(weights)


def _infer_network(
    features: np.ndarray,
    weights: tuple[np.ndarray, ...],
) -> np.ndarray:
    values = features
    for index, matrix in enumerate(weights):
        values = np.concatenate(
            (
                np.ones((values.shape[0], 1), dtype=np.float32),
                values,
            ),
            axis=1,
        ) @ matrix
        if index < len(weights) - 1:
            values = np.maximum(values, 0.0)
        else:
            values = 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))
    return values[:, 0].astype(np.float32)


def _predict_luminance(
    luminance: np.ndarray,
    weights: tuple[np.ndarray, ...],
    *,
    chunk_size: int,
) -> np.ndarray:
    height, width = luminance.shape
    output = luminance.copy()
    for row_start in range(4, height - 4, chunk_size):
        row_end = min(height - 4, row_start + chunk_size)
        for col_start in range(4, width - 4, chunk_size):
            col_end = min(width - 4, col_start + chunk_size)
            features = np.stack(
                [
                    luminance[
                        row_start + row_offset : row_end + row_offset,
                        col_start + col_offset : col_end + col_offset,
                    ].reshape(-1)
                    for row_offset, col_offset in _FEATURE_OFFSETS
                ],
                axis=1,
            ).astype(np.float32)
            predicted = _infer_network(features, weights)
            output[row_start:row_end, col_start:col_end] = predicted.reshape(
                row_end - row_start,
                col_end - col_start,
            )
    return output


def _restore_luv(luv: np.ndarray, luminance: np.ndarray) -> np.ndarray:
    restored = luv.copy()
    restored[..., 0] = np.clip(luminance * 100.0, 0.0, 100.0)
    return np.clip(
        cv2.cvtColor(restored, cv2.COLOR_Luv2RGB),
        0.0,
        1.0,
    ).astype(np.float32)


def _array_sha256(image: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(image).view(np.uint8)
    ).hexdigest()


def run_astrosharp_native_dual(
    image: np.ndarray,
    log: LogCallback | None = None,
    *,
    psf: AstroSharpPsfEstimate | None = None,
    aggressiveness: float = 1.0,
    chunk_size: int = 325,
    manifest_path: Path | None = None,
) -> AstroSharpNativeResult:
    """Run AstroSharp DualPSF natively with the official bundled weights."""
    if not MODEL_BUNDLE.is_file():
        raise RuntimeError(f"AstroSharp model bundle is missing: {MODEL_BUNDLE}")
    source = _to_float01(image)
    if source.ndim != 3 or source.shape[-1] < 3:
        raise ValueError("AstroSharp native inference requires an RGB image.")
    source = source[..., :3]
    estimate = psf or estimate_astrosharp_psf(source)
    aggressiveness = float(np.clip(aggressiveness, 0.01, 1.0))
    chunk_size = int(np.clip(chunk_size, 50, 750))
    if log:
        log(
            "ASTROSHARP_NATIVE_START "
            f"measured_fwhm={estimate.measured_fwhm_px:.3f}px "
            f"dso_psf={estimate.dso_psf:g} star_psf={estimate.star_psf:g} "
            f"stars={estimate.accepted_stars} confidence={estimate.confidence} "
            f"aggressiveness={aggressiveness:.2f} chunk_size={chunk_size}"
        )
    started = time.perf_counter()
    luv = cv2.cvtColor(source.astype(np.float32), cv2.COLOR_RGB2Luv)
    original_luminance = luv[..., 0] / 100.0
    with np.load(MODEL_BUNDLE) as bundle:
        dso_predicted = _predict_luminance(
            original_luminance,
            _load_weights(bundle, estimate.dso_psf),
            chunk_size=chunk_size,
        )
        star_predicted = _predict_luminance(
            original_luminance,
            _load_weights(bundle, estimate.star_psf),
            chunk_size=chunk_size,
        )
    dso_luminance = np.clip(
        dso_predicted * aggressiveness
        + original_luminance * (1.0 - aggressiveness),
        0.0,
        1.0,
    )
    star_luminance = np.clip(
        star_predicted * aggressiveness
        + original_luminance * (1.0 - aggressiveness),
        0.0,
        1.0,
    )
    dso_donor = _restore_luv(luv, dso_luminance)
    star_donor = _restore_luv(luv, star_luminance)
    elapsed = time.perf_counter() - started
    dso_delta = float(np.mean(np.abs(dso_luminance - original_luminance)))
    star_delta = float(np.mean(np.abs(star_luminance - original_luminance)))
    manifest: dict[str, object] = {
        "engine": "AstroSharp DualPSF native",
        "upstream": "https://github.com/deepskydetail/AstroSharp",
        "model_bundle": str(MODEL_BUNDLE),
        "model_bundle_sha256": hashlib.sha256(
            MODEL_BUNDLE.read_bytes()
        ).hexdigest(),
        "input_array_sha256": _array_sha256(source),
        "dso_donor_sha256": _array_sha256(dso_donor),
        "star_donor_sha256": _array_sha256(star_donor),
        "psf": asdict(estimate),
        "aggressiveness": aggressiveness,
        "chunk_size": chunk_size,
        "dso_delta_mean": dso_delta,
        "star_delta_mean": star_delta,
        "elapsed_seconds": elapsed,
        "actually_executed": True,
    }
    if manifest_path is not None:
        path = Path(manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if log:
        log(
            "ASTROSHARP_NATIVE_COMPLETE "
            f"elapsed={elapsed:.2f}s dso_delta_mean={dso_delta:.6f} "
            f"star_delta_mean={star_delta:.6f} "
            f"dso_sha256={manifest['dso_donor_sha256']} "
            f"star_sha256={manifest['star_donor_sha256']}"
        )
    return AstroSharpNativeResult(
        dso_donor=dso_donor,
        star_donor=star_donor,
        psf=estimate,
        manifest=manifest,
    )


def apply_astrosharp_native_dual(
    image: np.ndarray,
    log: LogCallback | None = None,
    *,
    psf: AstroSharpPsfEstimate | None = None,
    aggressiveness: float = 1.0,
    maximum_mix: float = 1.0,
    chunk_size: int = 325,
    manifest_path: Path | None = None,
) -> tuple[np.ndarray, AstroSharpNativeResult]:
    inference = run_astrosharp_native_dual(
        image,
        log,
        psf=psf,
        aggressiveness=aggressiveness,
        chunk_size=chunk_size,
        manifest_path=manifest_path,
    )
    result = blend_astrosharp_dual_structure(
        image,
        inference.dso_donor,
        inference.star_donor,
        log,
        maximum_mix=maximum_mix,
    )
    accepted_delta = float(
        np.mean(np.abs(_to_float01(result) - _to_float01(image)))
    )
    inference.manifest["maximum_mix"] = float(maximum_mix)
    inference.manifest["accepted_delta_mean"] = accepted_delta
    inference.manifest["accepted_output_sha256"] = _array_sha256(result)
    if manifest_path is not None:
        Path(manifest_path).write_text(
            json.dumps(inference.manifest, indent=2),
            encoding="utf-8",
        )
    return result, inference
