from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


LogCallback = Callable[[str], None]


@dataclass(frozen=True)
class AstroSharpRuntime:
    app_dir: Path
    rscript: Path
    runner: Path


def _to_float01(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(np.float32) / float(np.iinfo(array.dtype).max)
    result = np.nan_to_num(
        array.astype(np.float32),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    finite = result[np.isfinite(result)]
    if finite.size and float(np.max(finite)) > 1.5:
        result /= float(np.max(finite))
    return np.clip(result, 0.0, 1.0)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    ).astype(np.float32)


def _smoothstep(values: np.ndarray, low: float, high: float) -> np.ndarray:
    scaled = np.clip(
        (values - low) / max(float(high - low), 1e-7),
        0.0,
        1.0,
    )
    return scaled * scaled * (3.0 - 2.0 * scaled)


def _edge_support(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    border = max(8, int(min(height, width) * 0.018))
    support = np.ones((height, width), dtype=np.float32)
    support[:border] = 0.0
    support[-border:] = 0.0
    support[:, :border] = 0.0
    support[:, -border:] = 0.0
    return cv2.GaussianBlur(
        support,
        (0, 0),
        max(2.0, border * 0.28),
    )


def discover_astrosharp_runtime(
    app_dir: Path | None = None,
    *,
    runner: Path | None = None,
) -> AstroSharpRuntime | None:
    configured = app_dir or (
        Path(os.environ["DEEPSKY_ASTROSHARP_APP_DIR"])
        if os.environ.get("DEEPSKY_ASTROSHARP_APP_DIR")
        else None
    )
    if configured is None:
        return None
    configured = Path(configured).expanduser().resolve()
    rscript_override = os.environ.get("DEEPSKY_ASTROSHARP_RSCRIPT")
    rscript = (
        Path(rscript_override).expanduser().resolve()
        if rscript_override
        else configured / "R-Portable-Win" / "bin" / "x64" / "Rscript.exe"
    )
    runner_path = (
        Path(runner).expanduser().resolve()
        if runner is not None
        else Path(__file__).resolve().parents[1]
        / "tools"
        / "astrosharp"
        / "headless_astrosharp.R"
    )
    required = (
        configured / "GetMatrixFun9x9.R",
        configured / "PSF" / "81_1_FWHM_4.RDS",
        rscript,
        runner_path,
    )
    if not all(path.is_file() for path in required):
        return None
    return AstroSharpRuntime(
        app_dir=configured,
        rscript=rscript,
        runner=runner_path,
    )


def run_astrosharp_model(
    input_path: Path,
    output_path: Path,
    runtime: AstroSharpRuntime,
    log: LogCallback | None = None,
    *,
    psf: float = 4.0,
    strength: float = 0.50,
    chunk_size: int = 256,
) -> None:
    psf = round(float(psf) * 4.0) / 4.0
    strength = float(np.clip(strength, 0.0, 1.0))
    chunk_size = int(np.clip(chunk_size, 32, 750))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(runtime.rscript),
        str(runtime.runner),
        str(Path(input_path).resolve()),
        str(output_path.resolve()),
        str(runtime.app_dir),
        f"{psf:g}",
        f"{strength:g}",
        str(chunk_size),
    ]
    if log:
        log(
            "AstroSharp experimental inference: "
            f"psf={psf:g}, model_strength={strength:.2f}, chunk_size={chunk_size}."
        )
    process = subprocess.Popen(
        command,
        cwd=runtime.app_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    output_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        cleaned = line.rstrip()
        output_lines.append(cleaned)
        if log and (
            "complete:" in cleaned.lower()
            or "error" in cleaned.lower()
        ):
            log(cleaned)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            "AstroSharp failed with exit code "
            f"{return_code}: {' | '.join(output_lines[-8:])}"
        )
    if not output_path.is_file():
        raise RuntimeError("AstroSharp completed without creating its output TIFF.")


def blend_astrosharp_structure(
    base_image: np.ndarray,
    astrosharp_image: np.ndarray,
    log: LogCallback | None = None,
    *,
    maximum_mix: float = 0.76,
) -> np.ndarray:
    """Use AstroSharp as a bounded luminance donor inside measured nebula signal.

    Color always comes from ``base_image``. Empty sky, compact stars, and image
    borders continuously reject the donor, preventing noise, stellar rings,
    color drift, and the four-pixel unprocessed border from reaching the result.
    """
    base_array = np.asarray(base_image)
    base = _to_float01(base_array)
    donor = _to_float01(np.asarray(astrosharp_image))
    if (
        base.ndim != 3
        or base.shape[-1] < 3
        or donor.shape != base.shape
    ):
        raise ValueError("AstroSharp donor must match the RGB base image shape.")
    base = base[..., :3]
    donor = donor[..., :3]
    height, width = base.shape[:2]
    support = _edge_support((height, width))
    safe = support > 0.97

    base_lum = _luminance(base)
    donor_lum = _luminance(donor)
    broad_lum = cv2.GaussianBlur(
        base_lum,
        (0, 0),
        max(6.0, min(height, width) * 0.0065),
    )
    broad_rgb = cv2.GaussianBlur(base, (0, 0), 2.4)
    broad_chroma = np.max(broad_rgb, axis=2) - np.min(broad_rgb, axis=2)
    safe_lum = broad_lum[safe] if np.any(safe) else broad_lum.reshape(-1)
    safe_chroma = (
        broad_chroma[safe]
        if np.any(safe)
        else broad_chroma.reshape(-1)
    )
    luminance_signal = _smoothstep(
        broad_lum,
        float(np.percentile(safe_lum, 48.0)),
        float(np.percentile(safe_lum, 98.4)),
    )
    color_signal = _smoothstep(
        broad_chroma,
        float(np.percentile(safe_chroma, 60.0)),
        float(np.percentile(safe_chroma, 98.4)),
    )
    signal = np.clip(
        np.maximum(luminance_signal, color_signal * 0.86) * support,
        0.0,
        1.0,
    )
    signal = cv2.GaussianBlur(signal.astype(np.float32), (0, 0), 1.8)

    detection = cv2.GaussianBlur(base_lum, (0, 0), 0.72)
    compact = np.maximum(
        detection - cv2.GaussianBlur(detection, (0, 0), 2.8),
        0.0,
    )
    safe_compact = compact[safe] if np.any(safe) else compact.reshape(-1)
    star_low = float(np.percentile(safe_compact, 96.5))
    star_high = max(
        star_low + 1e-7,
        float(np.percentile(safe_compact, 99.78)),
    )
    star_protect = _smoothstep(compact, star_low, star_high)
    star_protect = np.clip(
        cv2.GaussianBlur(star_protect.astype(np.float32), (0, 0), 2.3) * 1.65,
        0.0,
        1.0,
    )

    maximum_mix = float(np.clip(maximum_mix, 0.0, 0.85))
    gate = np.clip(
        (signal**0.58)
        * np.square(1.0 - star_protect)
        * support
        * maximum_mix,
        0.0,
        maximum_mix,
    )
    raw_delta = donor_lum - base_lum
    delta_limit = 0.026 + base_lum * 0.105
    bounded_delta = np.clip(raw_delta, -delta_limit, delta_limit)
    result_lum = np.clip(base_lum + bounded_delta * gate, 0.0, 1.0)
    result = np.clip(
        base * (result_lum / np.maximum(base_lum, 1e-6))[..., None],
        0.0,
        1.0,
    )
    if log:
        log(
            "AstroSharp experimental structure donor blended through continuous "
            "nebula, stellar, and edge protection "
            f"(gate_mean={float(np.mean(gate)):.5f}, "
            f"gate_p95={float(np.percentile(gate, 95.0)):.5f}, "
            f"raw_delta_mean={float(np.mean(np.abs(raw_delta))):.6f}, "
            f"accepted_delta_mean={float(np.mean(np.abs(result_lum - base_lum))):.6f})."
        )
    if np.issubdtype(base_array.dtype, np.integer):
        maximum = float(np.iinfo(base_array.dtype).max)
        return np.clip(np.rint(result * maximum), 0, maximum).astype(base_array.dtype)
    return result.astype(np.float32)
