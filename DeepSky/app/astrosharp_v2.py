from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


LogCallback = Callable[[str], None]


@dataclass(frozen=True)
class AstroSharpDualRuntime:
    app_dir: Path
    rscript: Path
    runner: Path


@dataclass(frozen=True)
class AstroSharpPsfEstimate:
    measured_fwhm_px: float
    dso_psf: float
    star_psf: float
    accepted_stars: int
    confidence: str


@dataclass(frozen=True)
class AstroSharpDualResult:
    dso_output: Path
    star_output: Path
    manifest: Path
    psf: AstroSharpPsfEstimate
    elapsed_seconds: float


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


def _quarter_step(value: float) -> float:
    return float(np.clip(round(float(value) * 4.0) / 4.0, 1.0, 8.0))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def discover_astrosharp_dual_runtime(
    app_dir: Path | None = None,
    *,
    runner: Path | None = None,
) -> AstroSharpDualRuntime | None:
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
        / "headless_astrosharp_dual.R"
    )
    model_dir = configured / "PSF"
    model_count = len(list(model_dir.glob("81_1_FWHM_*.RDS")))
    required = (
        configured / "GetMatrixFun9x9.R",
        rscript,
        runner_path,
    )
    if model_count < 20 or not all(path.is_file() for path in required):
        return None
    return AstroSharpDualRuntime(
        app_dir=configured,
        rscript=rscript,
        runner=runner_path,
    )


def estimate_astrosharp_psf(image: np.ndarray) -> AstroSharpPsfEstimate:
    """Estimate AstroSharp's PSF control from measured compact-star FWHM.

    AstroSharp instructs users to measure stellar FWHM and divide by 2.35.
    Robust second moments provide the corresponding Gaussian sigma here.
    """
    rgb = _to_float01(image)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError("AstroSharp PSF estimation requires an RGB image.")
    lum = _luminance(rgb[..., :3])
    local = cv2.GaussianBlur(lum, (0, 0), 4.0)
    compact = np.maximum(lum - local, 0.0)
    threshold = max(
        float(np.percentile(compact, 99.35)),
        float(np.median(compact) + np.std(compact) * 2.5),
    )
    seeds = (compact > threshold).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        seeds,
        connectivity=8,
    )
    sigmas: list[float] = []
    height, width = lum.shape
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        center_x, center_y = centroids[label]
        x = int(round(center_x))
        y = int(round(center_y))
        if (
            area < 2
            or area > 90
            or x < 7
            or y < 7
            or x >= width - 7
            or y >= height - 7
        ):
            continue
        patch = lum[y - 6 : y + 7, x - 6 : x + 7]
        background = float(np.percentile(patch, 20.0))
        weights = np.maximum(patch - background, 0.0)
        total = float(np.sum(weights))
        if total <= 1e-5 or float(np.max(patch)) >= 0.995:
            continue
        yy, xx = np.mgrid[-6:7, -6:7].astype(np.float32)
        centroid_x = float(np.sum(weights * xx) / total)
        centroid_y = float(np.sum(weights * yy) / total)
        variance = float(
            np.sum(
                weights
                * (
                    np.square(xx - centroid_x)
                    + np.square(yy - centroid_y)
                )
            )
            / (2.0 * total)
        )
        sigma = float(np.sqrt(max(variance, 0.0)))
        if 0.70 <= sigma <= 6.5:
            sigmas.append(sigma)
    if len(sigmas) < 5:
        return AstroSharpPsfEstimate(
            measured_fwhm_px=7.05,
            dso_psf=3.0,
            star_psf=3.0,
            accepted_stars=len(sigmas),
            confidence="fallback",
        )
    values = np.asarray(sigmas, dtype=np.float32)
    low, high = np.percentile(values, [18.0, 72.0])
    clipped = values[(values >= low) & (values <= high)]
    sigma = float(np.median(clipped if clipped.size else values))
    dso_psf = _quarter_step(sigma)
    # The stellar donor is intentionally one quarter-step lower. This follows
    # DualPSF's separate stellar model design while avoiding enlarged cores.
    star_psf = _quarter_step(dso_psf - 0.25)
    return AstroSharpPsfEstimate(
        measured_fwhm_px=sigma * 2.35,
        dso_psf=dso_psf,
        star_psf=star_psf,
        accepted_stars=len(sigmas),
        confidence="measured" if len(sigmas) >= 18 else "limited",
    )


def run_astrosharp_dual(
    input_path: Path,
    dso_output: Path,
    star_output: Path,
    runtime: AstroSharpDualRuntime,
    psf: AstroSharpPsfEstimate,
    log: LogCallback | None = None,
    *,
    aggressiveness: float = 1.0,
    chunk_size: int = 325,
) -> AstroSharpDualResult:
    aggressiveness = float(np.clip(aggressiveness, 0.01, 1.0))
    chunk_size = int(np.clip(chunk_size, 50, 750))
    dso_output = Path(dso_output)
    star_output = Path(star_output)
    dso_output.parent.mkdir(parents=True, exist_ok=True)
    star_output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(runtime.rscript),
        str(runtime.runner),
        str(Path(input_path).resolve()),
        str(dso_output.resolve()),
        str(star_output.resolve()),
        str(runtime.app_dir),
        f"{psf.dso_psf:g}",
        f"{psf.star_psf:g}",
        f"{aggressiveness:g}",
        str(chunk_size),
    ]
    if log:
        log(
            "AstroSharp DualPSF inference started: "
            f"measured_fwhm={psf.measured_fwhm_px:.3f}px, "
            f"dso_psf={psf.dso_psf:g}, star_psf={psf.star_psf:g}, "
            f"stars={psf.accepted_stars}, confidence={psf.confidence}, "
            f"aggressiveness={aggressiveness:.2f}, chunk_size={chunk_size}."
        )
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=runtime.app_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        lines = (process.stdout + "\n" + process.stderr).splitlines()
        raise RuntimeError(
            "AstroSharp DualPSF failed with exit code "
            f"{process.returncode}: {' | '.join(lines[-8:])}"
        )
    if not dso_output.is_file() or not star_output.is_file():
        raise RuntimeError(
            "AstroSharp DualPSF completed without both donor TIFFs."
        )
    manifest_path = dso_output.parent / "astrosharp_manifest.json"
    manifest = {
        "engine": "AstroSharp DualPSF",
        "upstream": "https://github.com/deepskydetail/AstroSharp",
        "runner": str(runtime.runner),
        "input": str(Path(input_path).resolve()),
        "input_sha256": _sha256(Path(input_path)),
        "dso_output": str(dso_output.resolve()),
        "dso_output_sha256": _sha256(dso_output),
        "star_output": str(star_output.resolve()),
        "star_output_sha256": _sha256(star_output),
        "psf": asdict(psf),
        "aggressiveness": aggressiveness,
        "chunk_size": chunk_size,
        "elapsed_seconds": elapsed,
        "return_code": process.returncode,
        "actually_executed": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    if log:
        log(
            "AstroSharp DualPSF verified: both donor files and SHA-256 "
            f"manifest created in {elapsed:.2f}s."
        )
    return AstroSharpDualResult(
        dso_output=dso_output,
        star_output=star_output,
        manifest=manifest_path,
        psf=psf,
        elapsed_seconds=elapsed,
    )


def blend_astrosharp_dual_structure(
    base_image: np.ndarray,
    dso_image: np.ndarray,
    star_image: np.ndarray,
    log: LogCallback | None = None,
    *,
    maximum_mix: float = 1.0,
) -> np.ndarray:
    """Compose the official DualPSF donors without discarding quiet-frame output.

    AstroSharp applies its DSO model across the entire image and continuously
    switches to its separate stellar model around detected stars. Earlier
    DeepSky builds added a nebula-signal gate and delta limiter here, which
    rejected most of the neural-network result and made activation invisible.
    """
    base_array = np.asarray(base_image)
    base = _to_float01(base_array)
    dso = _to_float01(dso_image)
    stars = _to_float01(star_image)
    if (
        base.ndim != 3
        or base.shape[-1] < 3
        or dso.shape != base.shape
        or stars.shape != base.shape
    ):
        raise ValueError("Both AstroSharp donors must match the RGB base.")
    base = base[..., :3]
    base_lum = _luminance(base)
    dso_lum = _luminance(dso)
    star_lum = _luminance(stars)

    # Approximate the upstream multiscale stellar mask continuously. Compact
    # positive structure is measured at several scales so small and moderately
    # bloated stars select the stellar PSF without hard masks or neon cores.
    stellar_responses = []
    for sigma in (0.8, 1.4, 2.4, 4.0):
        narrow = cv2.GaussianBlur(base_lum, (0, 0), sigma)
        broad = cv2.GaussianBlur(base_lum, (0, 0), sigma * 1.9)
        stellar_responses.append(np.maximum(narrow - broad, 0.0))
    compact = np.maximum.reduce(stellar_responses)
    low = float(np.percentile(compact, 97.2))
    high = max(low + 1e-7, float(np.percentile(compact, 99.92)))
    stellar = _smoothstep(compact, low, high)
    stellar = cv2.GaussianBlur(stellar.astype(np.float32), (0, 0), 1.45)
    stellar = np.clip(stellar * 1.65, 0.0, 1.0)

    dual_lum = dso_lum * (1.0 - stellar) + star_lum * stellar
    maximum_mix = float(np.clip(maximum_mix, 0.0, 1.0))
    result_lum = np.clip(
        base_lum + (dual_lum - base_lum) * maximum_mix,
        0.0,
        1.0,
    )
    result = np.clip(
        base * (result_lum / np.maximum(base_lum, 1e-6))[..., None],
        0.0,
        1.0,
    )
    if log:
        log(
            "AstroSharp DualPSF accepted as the full-frame structure stage: "
            f"dso_delta_mean={float(np.mean(np.abs(dso_lum - base_lum))):.6f}, "
            f"star_delta_mean={float(np.mean(np.abs(star_lum - base_lum))):.6f}, "
            f"accepted_delta_mean={float(np.mean(np.abs(result_lum - base_lum))):.6f}, "
            f"stellar_mask_mean={float(np.mean(stellar)):.5f}, "
            f"maximum_mix={maximum_mix:.2f}."
        )
    if np.issubdtype(base_array.dtype, np.integer):
        maximum = float(np.iinfo(base_array.dtype).max)
        return np.clip(
            np.rint(result * maximum),
            0,
            maximum,
        ).astype(base_array.dtype)
    return result.astype(np.float32)