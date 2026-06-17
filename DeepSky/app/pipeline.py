from __future__ import annotations

import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .cli_tools import find_executable, run_deepsnr, run_starnet
from .goal_look import (
    apply_broadband_look,
    apply_cosmos_style_nebula_finish,
    apply_goal_look,
    apply_pixinsight_style_nebula_finish,
    apply_prestretched_broadband_look,
    apply_prestretched_nebula_rgb_reveal,
    apply_small_galaxy_darkroom_look,
    apply_starless_nebula_detail,
    blend_galaxy_deconvolution_detail,
    blend_broadband_background_denoise,
    chroma_percentile,
    red_emission_dominance,
)
from .image_io import (
    convert_to_working_tiff,
    describe_array,
    is_supported_input,
    load_image,
    make_preview,
    save_png,
    save_tiff,
)
from .input_analysis import analyze_input_stretch, detect_telescope_profile
from .image_math import add_bright_star_fraction, add_images, add_weighted_star_layer, subtract_images
from .python_color_calibration import python_fallback_color_calibration
from .settings import AppSettings
from .siril_cli import (
    build_siril_pcc_command,
    create_basic_color_script,
    create_photometric_color_script,
    find_siril_executable,
    run_siril_script,
)
from .stretch import astrophotography_stretch


LogCallback = Callable[[str], None]


class PipelineMode(str, Enum):
    FULL = "full"
    STRETCH = "stretch"
    DEEPSNR = "deepsnr"
    STARNET = "starnet"
    SIRIL = "siril"


def create_job_folder(output_root: Path, input_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_folder = output_root / f"{stamp}_{input_path.stem}"
    job_folder.mkdir(parents=True, exist_ok=False)
    return job_folder


def _log_existing_image(path: Path, write_log: LogCallback, label: str) -> None:
    image = load_image(path, write_log)
    write_log(f"{label}: {describe_array(path, image)}")


def _normalized_object_type(settings: AppSettings) -> str:
    value = getattr(settings, "object_type", "Nebula").strip().lower()
    if value in {"galaxy", "star cluster"}:
        return value
    return "nebula"


def _normalized_input_mode(settings: AppSettings) -> str:
    value = getattr(settings, "input_processing_mode", "Auto").strip().lower()
    if value in {"pre-stretched", "pre_stretched", "prestretched"}:
        return "pre_stretched"
    if value == "linear":
        return "linear"
    return "auto"


def _normalized_stretch_level(settings: AppSettings) -> str:
    value = getattr(settings, "stretch_level", "Standard").strip().lower()
    if value in {"subtle", "slightly aggressive", "slight", "slightly_aggressive"}:
        return "subtle"
    if value == "aggressive":
        return "aggressive"
    return "standard"


def _stretch_strength_for(base: str, stretch_level: str) -> str:
    return _adjust_stretch_strength(base, stretch_level)


def _adjust_stretch_strength(base_strength: str, stretch_level: str) -> str:
    ladders = (
        ["gentle", "slight", "normal", "aggressive", "extra_aggressive"],
        ["seestar_slight", "seestar", "seestar_aggressive", "seestar_extra_aggressive"],
    )
    for ladder in ladders:
        if base_strength in ladder:
            index = ladder.index(base_strength)
            if stretch_level == "subtle":
                index -= 1
            elif stretch_level == "aggressive":
                index += 1
            return ladder[max(0, min(len(ladder) - 1, index))]

    if base_strength == "seestar":
        if stretch_level == "subtle":
            return "seestar_slight"
        if stretch_level == "aggressive":
            return "seestar_extra_aggressive"
        return "seestar_aggressive"
    if base_strength == "gentle":
        if stretch_level == "subtle":
            return "slight"
        if stretch_level == "aggressive":
            return "aggressive"
        return "gentle"
    if base_strength == "normal":
        if stretch_level == "subtle":
            return "slight"
        if stretch_level == "aggressive":
            return "aggressive"
        return "normal"
    return base_strength


def _looks_like_green_duoband_raw(image: np.ndarray, analysis: object | None) -> bool:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return False

    rgb = arr[..., :3].astype(np.float32)
    if np.issubdtype(arr.dtype, np.integer):
        max_value = float(np.iinfo(arr.dtype).max)
        if max_value > 0:
            rgb /= max_value
    elif rgb.size and float(np.nanmax(rgb)) > 1.0:
        rgb /= 65535.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)

    pixels = rgb.reshape(-1, 3)
    medians = np.percentile(pixels, 50.0, axis=0)
    highs = np.percentile(pixels, 97.5, axis=0)
    metrics = getattr(analysis, "metrics", {}) if analysis is not None else {}
    raw_p50 = float(metrics.get("raw_p50", np.percentile(rgb, 50.0)))
    raw_p999 = float(metrics.get("raw_p999", np.percentile(rgb, 99.9)))

    compressed_raw = raw_p50 > 0.045 and raw_p999 < 0.16
    green_median = medians[1] > max(medians[0], medians[2]) * 1.10 + 0.004
    green_high = highs[1] > max(highs[0], highs[2]) * 1.02 + 0.004
    return bool(compressed_raw and green_median and green_high)


def _to_float01(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    rgb = arr[..., :3].astype(np.float32) if arr.ndim == 3 else arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.integer):
        max_value = float(np.iinfo(arr.dtype).max)
        if max_value > 0:
            rgb /= max_value
    elif rgb.size and float(np.nanmax(rgb)) > 1.0:
        rgb /= 65535.0
    return np.nan_to_num(np.clip(rgb, 0.0, 1.0), nan=0.0, posinf=0.0, neginf=0.0)


def _crop_edge_artifacts(image: np.ndarray, fraction: float = 0.06) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim < 2:
        return arr
    height, width = arr.shape[:2]
    base_y = int(round(height * fraction))
    base_x = int(round(width * fraction))
    if base_y < 1 or base_x < 1:
        return arr

    rgb = _to_float01(arr)
    lum = _rgb_luminance(rgb)
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)

    inset_y = max(base_y, int(round(height * 0.12)))
    inset_x = max(base_x, int(round(width * 0.12)))
    interior = lum[inset_y : height - inset_y, inset_x : width - inset_x]
    interior_chroma = chroma[inset_y : height - inset_y, inset_x : width - inset_x]
    if interior.size < 1024:
        return arr

    interior_p10 = float(np.percentile(interior, 10.0))
    interior_median = float(np.median(interior))
    interior_p95 = float(np.percentile(interior, 95.0))
    interior_std = float(np.std(interior))
    interior_chroma_p95 = float(np.percentile(interior_chroma, 95.0))
    max_y = int(round(height * 0.14))
    max_x = int(round(width * 0.14))
    step_y = max(4, int(round(height * 0.01)))
    step_x = max(4, int(round(width * 0.01)))
    stripe_y = max(6, int(round(height * 0.018)))
    stripe_x = max(6, int(round(width * 0.018)))

    top = bottom = base_y
    left = right = base_x

    def bad_luminance(stripe: np.ndarray, stripe_chroma: np.ndarray, vertical: bool) -> bool:
        if stripe.size < 16:
            return False
        axis_len = stripe.shape[0] if vertical else stripe.shape[1]
        tile_count = 8
        darkest_p10 = 1.0
        darkest_median = 1.0
        worst_std = 0.0
        worst_chroma = 0.0
        for index in range(tile_count):
            start = index * axis_len // tile_count
            stop = (index + 1) * axis_len // tile_count
            tile = stripe[start:stop, :] if vertical else stripe[:, start:stop]
            tile_chroma = stripe_chroma[start:stop, :] if vertical else stripe_chroma[:, start:stop]
            if tile.size < 16:
                continue
            darkest_p10 = min(darkest_p10, float(np.percentile(tile, 10.0)))
            darkest_median = min(darkest_median, float(np.median(tile)))
            worst_std = max(worst_std, float(np.std(tile)))
            worst_chroma = max(worst_chroma, float(np.percentile(tile_chroma, 95.0)))
        dark_and_noisy = (
            (darkest_p10 < interior_p10 * 0.72 - 0.004 or darkest_median < interior_median * 0.74 - 0.006)
            and (worst_std > interior_std * 1.45 + 0.003 or worst_chroma > interior_chroma_p95 * 1.28 + 0.008)
        )
        extreme_noise = (
            worst_std > interior_std * 2.05 + 0.006
            and worst_chroma > interior_chroma_p95 * 1.55 + 0.010
            and darkest_median < interior_p95 * 0.92
        )
        return bool(
            dark_and_noisy or extreme_noise
        )

    while top < max_y and bad_luminance(lum[top : top + stripe_y, :], chroma[top : top + stripe_y, :], vertical=False):
        top += step_y
    while bottom < max_y and bad_luminance(lum[height - bottom - stripe_y : height - bottom, :], chroma[height - bottom - stripe_y : height - bottom, :], vertical=False):
        bottom += step_y
    while left < max_x and bad_luminance(lum[:, left : left + stripe_x], chroma[:, left : left + stripe_x], vertical=True):
        left += step_x
    while right < max_x and bad_luminance(lum[:, width - right - stripe_x : width - right], chroma[:, width - right - stripe_x : width - right], vertical=True):
        right += step_x

    max_side_crop_y = int(round(height * 0.05))
    max_side_crop_x = int(round(width * 0.05))
    max_total_crop_y = int(round(height * 0.08))
    max_total_crop_x = int(round(width * 0.08))
    if (
        top > max_side_crop_y
        or bottom > max_side_crop_y
        or left > max_side_crop_x
        or right > max_side_crop_x
        or top + bottom > max_total_crop_y
        or left + right > max_total_crop_x
    ):
        return arr
    if height - top - bottom < 96 or width - left - right < 96:
        return arr
    return arr[top : height - bottom, left : width - right].copy()


def _rgb_luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _apply_duoband_nebula_finish(
    image: np.ndarray,
    write_log: LogCallback,
    palette: str = "warm",
) -> np.ndarray:
    rgb = _to_float01(image)
    lum = _rgb_luminance(rgb).astype(np.float32)

    black = float(np.percentile(lum, 10.0))
    white = float(np.percentile(lum, 99.75))
    base_lum = np.clip((lum - black) / max(1e-6, white - black), 0.0, 1.0)
    base_lum = np.clip(base_lum ** 0.78, 0.0, 1.0)

    large = cv2.GaussianBlur(base_lum, (0, 0), 16.0)
    small = cv2.GaussianBlur(base_lum, (0, 0), 1.2)
    detail = np.clip(base_lum + (small - large) * 0.42, 0.0, 1.0)

    green_signal = np.clip(rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2]) * 0.72, 0.0, 1.0)
    sky_anchor = float(np.percentile(base_lum, 54.0))
    signal = np.clip(
        (base_lum - sky_anchor)
        / max(1e-6, np.percentile(base_lum, 99.35) - sky_anchor),
        0.0,
        1.0,
    )
    chroma_signal = np.clip(green_signal / max(1e-6, float(np.percentile(green_signal, 99.6))), 0.0, 1.0)
    signal = np.clip(signal * 0.78 + chroma_signal * 0.26, 0.0, 1.0)
    signal = cv2.GaussianBlur((signal ** 0.92).astype(np.float32), (0, 0), 1.35)

    star_core = np.clip(
        (base_lum - np.percentile(base_lum, 98.2))
        / max(1e-6, np.percentile(base_lum, 99.96) - np.percentile(base_lum, 98.2)),
        0.0,
        1.0,
    )
    star_core = cv2.GaussianBlur(star_core.astype(np.float32), (0, 0), 0.7)
    broad_haze = cv2.GaussianBlur(signal.astype(np.float32), (0, 0), 18.0)
    haze_reject = np.clip((broad_haze - 0.16) / 0.40, 0.0, 1.0)
    nebula = np.clip(signal * (1.0 - star_core * 0.82) * (0.66 + 0.34 * haze_reject), 0.0, 1.0)
    nebula = np.clip((nebula - 0.055) / 0.945, 0.0, 1.0)

    palettes = {
        "gold": (
            np.array([1.68, 0.82, 0.38], dtype=np.float32),
            np.array([1.18, 0.72, 0.48], dtype=np.float32),
            0.16,
        ),
        "warm": (
            np.array([1.46, 0.74, 0.42], dtype=np.float32),
            np.array([1.05, 0.70, 0.54], dtype=np.float32),
            0.20,
        ),
        "deep": (
            np.array([1.58, 0.66, 0.32], dtype=np.float32),
            np.array([0.70, 0.88, 1.12], dtype=np.float32),
            0.23,
        ),
    }
    warm_color, cool_color, sky_floor = palettes.get(palette, palettes["warm"])

    sky_mask = np.clip(1.0 - nebula * 1.35 - star_core * 0.75, 0.0, 1.0)
    sky_mask = cv2.GaussianBlur(sky_mask.astype(np.float32), (0, 0), 2.0)
    neutral_sky = np.clip(detail[..., None] * np.array([0.50, 0.51, 0.50], dtype=np.float32), 0.0, 1.0)
    warm_nebula = np.clip(detail[..., None] * warm_color.reshape(1, 1, 3), 0.0, 1.0)
    cool_shadow = np.clip(detail[..., None] * cool_color.reshape(1, 1, 3), 0.0, 1.0)
    shadow = np.clip((large - detail) / max(1e-6, float(np.percentile(large, 98.0))), 0.0, 1.0)
    warm_signal = np.clip(rgb[..., 0] - np.maximum(rgb[..., 1], rgb[..., 2]) * 0.76, 0.0, 1.0)
    cool_signal = np.clip(np.maximum(rgb[..., 1], rgb[..., 2]) - rgb[..., 0] * 0.84, 0.0, 1.0)
    warm_norm = np.clip(warm_signal / max(1e-6, float(np.percentile(warm_signal, 99.5))), 0.0, 1.0)
    cool_norm = np.clip(cool_signal / max(1e-6, float(np.percentile(cool_signal, 99.5))), 0.0, 1.0)
    broad_halo = np.clip(cv2.GaussianBlur(nebula.astype(np.float32), (0, 0), 9.0) * (1.0 - signal * 0.52), 0.0, 1.0)
    pink_halo = np.clip(detail[..., None] * np.array([1.10, 0.72, 1.08], dtype=np.float32).reshape(1, 1, 3), 0.0, 1.0)
    warm_gold = np.clip(detail[..., None] * np.array([1.24, 0.86, 0.46], dtype=np.float32).reshape(1, 1, 3), 0.0, 1.0)
    cool_cyan = np.clip(detail[..., None] * np.array([0.58, 0.98, 1.28], dtype=np.float32).reshape(1, 1, 3), 0.0, 1.0)
    bright_core = np.clip(
        (detail - np.percentile(detail, 85.0))
        / max(1e-6, np.percentile(detail, 99.7) - np.percentile(detail, 85.0)),
        0.0,
        1.0,
    ) * nebula
    core_white = np.clip(detail[..., None] * np.array([1.06, 1.00, 0.94], dtype=np.float32).reshape(1, 1, 3), 0.0, 1.0)

    finished = np.clip(neutral_sky - sky_floor * sky_mask[..., None], 0.0, 1.0)
    finished = np.clip(finished * (1.0 - shadow[..., None] * 0.34) + cool_shadow * (shadow[..., None] * nebula[..., None] * 0.20), 0.0, 1.0)
    finished = np.clip(finished * (1.0 - broad_halo[..., None] * 0.16) + pink_halo * (broad_halo[..., None] * 0.16), 0.0, 1.0)
    finished = np.clip(finished * (1.0 - nebula[..., None] * 0.86) + warm_nebula * (nebula[..., None] * 0.86), 0.0, 1.0)
    finished = np.clip(finished * (1.0 - warm_norm[..., None] * nebula[..., None] * 0.20) + warm_gold * (warm_norm[..., None] * nebula[..., None] * 0.20), 0.0, 1.0)
    finished = np.clip(finished * (1.0 - cool_norm[..., None] * nebula[..., None] * 0.28) + cool_cyan * (cool_norm[..., None] * nebula[..., None] * 0.28), 0.0, 1.0)
    finished = np.clip(finished * (1.0 - bright_core[..., None] * 0.12) + core_white * (bright_core[..., None] * 0.12), 0.0, 1.0)
    finished = np.clip(finished * (1.0 - sky_mask[..., None] * 0.36), 0.0, 1.0)

    star_neutral = np.clip(base_lum[..., None] * np.array([1.10, 1.02, 0.94], dtype=np.float32), 0.0, 1.0)
    finished = np.clip(finished * (1.0 - star_core[..., None] * 0.78) + star_neutral * (star_core[..., None] * 0.78), 0.0, 1.0)

    final_lum = _rgb_luminance(finished)
    saturation = 0.78 + nebula[..., None] * (0.92 if palette != "gold" else 1.05) + star_core[..., None] * 0.15
    finished = np.clip(final_lum[..., None] + (finished - final_lum[..., None]) * saturation, 0.0, 1.0)
    finished = np.clip(finished * (1.0 - sky_mask[..., None] * 0.18) + final_lum[..., None] * (sky_mask[..., None] * 0.18), 0.0, 1.0)

    write_log(
        "Applied duo-band nebula color finish: "
        f"palette={palette}; black={black:.5f}; white={white:.5f}; "
        f"signal_mean={float(np.mean(signal)):.5f}; nebula_mean={float(np.mean(nebula)):.5f}; "
        f"warm_mean={float(np.mean(warm_norm * nebula)):.5f}; cool_mean={float(np.mean(cool_norm * nebula)):.5f}; "
        f"sky_mean={float(np.mean(sky_mask)):.5f}; "
        f"star_mean={float(np.mean(star_core)):.5f}"
    )
    return np.clip(finished * 65535.0, 0.0, 65535.0).round().astype(np.uint16)


def _working_background_spread(image: np.ndarray) -> float:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return 0.0
    rgb = arr[..., :3].astype(np.float32)
    if np.issubdtype(arr.dtype, np.integer):
        max_value = float(np.iinfo(arr.dtype).max)
        if max_value > 0:
            rgb /= max_value
    elif rgb.size and float(np.nanmax(rgb)) > 1.0:
        rgb /= 65535.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    lum = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    height, width = lum.shape
    tile_values: list[float] = []
    rows = cols = 6
    for row in range(rows):
        y0 = row * height // rows
        y1 = (row + 1) * height // rows
        for col in range(cols):
            x0 = col * width // cols
            x1 = (col + 1) * width // cols
            tile = lum[y0:y1, x0:x1]
            if tile.size:
                tile_values.append(float(np.percentile(tile, 50.0)))
    if len(tile_values) < 4:
        return 0.0
    values = np.asarray(tile_values, dtype=np.float32)
    median = float(np.percentile(values, 50.0))
    return float((np.percentile(values, 90.0) - np.percentile(values, 10.0)) / max(median, 1e-6))


def _needs_siril_for_gradient_galaxy(
    image: np.ndarray,
    analysis: object | None,
    detected_telescope: str,
    object_type: str,
) -> tuple[bool, float]:
    if object_type != "galaxy" or detected_telescope != "seestar":
        return False, 0.0
    metrics = getattr(analysis, "metrics", {}) if analysis is not None else {}
    raw_p999 = float(metrics.get("raw_p999", 1.0))
    bright_fraction = float(metrics.get("bright_fraction", 1.0))
    spread = _working_background_spread(image)
    small_faint_galaxy = raw_p999 < 0.012 and bright_fraction < 0.00025
    return bool(small_faint_galaxy and spread > 0.25), spread


def _is_compact_siril_galaxy(analysis: object | None) -> bool:
    metrics = getattr(analysis, "metrics", {}) if analysis is not None else {}
    raw_p999 = float(metrics.get("raw_p999", 1.0))
    bright_fraction = float(metrics.get("bright_fraction", 1.0))
    return raw_p999 < 0.025 and bright_fraction < 0.00025


def _needs_gentle_nebula_star_reduction(analysis: object | None, image: np.ndarray) -> bool:
    metrics = getattr(analysis, "metrics", {}) if analysis is not None else {}
    raw_p999 = float(metrics.get("raw_p999", 0.0))
    emission_score = red_emission_dominance(image)
    return raw_p999 >= 0.055 and emission_score < 2.4


def _auto_baseline_stretch_strength(
    analysis: object | None,
    detected_telescope: str,
    object_type: str,
    *,
    gentle_recommended: bool,
) -> tuple[str, str]:
    metrics = getattr(analysis, "metrics", {}) if analysis is not None else {}
    raw_p99 = float(metrics.get("raw_p99", 0.0))
    raw_p999 = float(metrics.get("raw_p999", 0.0))
    shadow_fraction = float(metrics.get("shadow_fraction", 0.0))
    midtone_fraction = float(metrics.get("midtone_fraction", 0.0))
    bright_fraction = float(metrics.get("bright_fraction", 0.0))
    recommended_mode = getattr(analysis, "recommended_mode", "")

    if recommended_mode == "pre_stretched":
        return "gentle", "histogram already looks stretched"
    if object_type == "galaxy":
        if raw_p999 < 0.012:
            return "seestar_aggressive", f"very faint protected raw galaxy signal raw_p999={raw_p999:.5f}"
        if raw_p999 < 0.020 and bright_fraction < 0.00020:
            return (
                "seestar_slight",
                f"compact bright galaxy protected from over-stretch raw_p999={raw_p999:.5f}, bright_fraction={bright_fraction:.5f}",
            )
        return "seestar", f"protected raw galaxy baseline raw_p999={raw_p999:.5f}"
    if raw_p999 < 0.012:
        return "seestar_extra_aggressive", f"very faint protected raw signal raw_p999={raw_p999:.5f}"
    if raw_p999 < 0.032:
        return "seestar_aggressive", f"faint protected raw signal raw_p999={raw_p999:.5f}"
    if detected_telescope == "seestar":
        return "seestar", f"moderate SeeStar signal raw_p999={raw_p999:.5f}"
    if gentle_recommended:
        return "seestar_slight", "soft-stretched protected raw histogram"
    if shadow_fraction > 0.98 and midtone_fraction < 0.003:
        return "seestar_aggressive", f"very dark protected raw histogram shadow_fraction={shadow_fraction:.5f}"
    if raw_p99 > 0.085 and raw_p999 < 0.14:
        return "seestar_slight", f"bright low-dynamic-range protected raw signal raw_p99={raw_p99:.5f}"
    if raw_p999 < 0.08:
        return "seestar_aggressive", f"faint protected raw signal raw_p999={raw_p999:.5f}"
    return "seestar", "normal protected raw histogram"


def _run_local_stretch_calibration(
    working: Path,
    stretched: Path,
    calibrated: Path,
    write_log: LogCallback,
    strength: str = "normal",
) -> Path:
    stretched_image = astrophotography_stretch(load_image(working, write_log), strength=strength)
    save_tiff(stretched, stretched_image, write_log)
    _log_existing_image(stretched, write_log, "stretched.tif")
    shutil.copy2(stretched, calibrated)
    _log_existing_image(calibrated, write_log, "calibrated.tif")
    return calibrated


def _apply_broadband_background_cleanup(
    image: np.ndarray,
    job_folder: Path,
    settings: AppSettings,
    write_log: LogCallback,
    label: str,
) -> np.ndarray:
    finished_image = apply_broadband_look(image, write_log)
    deepsnr_exe = find_executable(Path(settings.deepsnr_folder))
    if not deepsnr_exe:
        write_log(f"DeepSNR {label} background cleanup skipped; executable not found.")
        return finished_image

    pre_denoise = job_folder / f"{label}_pre_deepsnr.tif"
    deepsnr_background = job_folder / f"{label}_deepsnr.tif"
    save_tiff(pre_denoise, finished_image, write_log)
    write_log(f"DeepSNR {label} background cleanup executable: {deepsnr_exe}")
    try:
        run_deepsnr(pre_denoise, deepsnr_background, deepsnr_exe, write_log)
        _log_existing_image(deepsnr_background, write_log, f"{label}_deepsnr.tif")
        return blend_broadband_background_denoise(
            finished_image,
            load_image(deepsnr_background, write_log),
            settings.galaxy_background_smoothness,
            settings.galaxy_background_darkness,
            settings.galaxy_chroma_noise_reduction,
            settings.galaxy_protect_detail,
            write_log,
        )
    except Exception as exc:
        write_log(f"DeepSNR {label} background cleanup failed; keeping broadband finish. Error: {exc}")
        return finished_image


def _run_siril_calibration(
    original: Path,
    working: Path,
    stretched: Path,
    calibrated: Path,
    job_folder: Path,
    settings: AppSettings,
    write_log: LogCallback,
    *,
    darkroom_small_galaxy: bool = False,
) -> Path:
    mode = settings.color_calibration_mode
    object_type = _normalized_object_type(settings)
    use_deconvolution_layer = (
        bool(getattr(settings, "siril_deconvolution_enabled", False))
        and mode == "Basic"
        and object_type == "galaxy"
    )
    if mode == "Off":
        write_log("Color calibration is off; applying local stretch only.")
        return _run_local_stretch_calibration(working, stretched, calibrated, write_log)

    siril_exe = find_siril_executable(Path(settings.siril_folder))
    if not siril_exe:
        write_log("Siril executable not found; using Python fallback color calibration.")
        return _run_python_fallback_calibration(working, stretched, calibrated, settings, write_log)

    pcc_command = build_siril_pcc_command(original) if mode == "Siril Photometric" else None
    if mode == "Siril Photometric" and pcc_command:
        siril_input = job_folder / original.name
        if siril_input.resolve() != original.resolve():
            shutil.copy2(original, siril_input)
        write_log(f"Siril PCC metadata command: {pcc_command}")
    elif mode == "Siril Photometric":
        write_log("Siril PCC metadata unavailable; using Python fallback color calibration.")
        return _run_python_fallback_calibration(working, stretched, calibrated, settings, write_log)
    else:
        siril_input = job_folder / "siril_input.tif"
        shutil.copy2(working, siril_input)
    siril_output_fit = job_folder / "siril_output.fit"

    write_log(f"Siril executable: {siril_exe}")
    write_log(
        "Siril Color Settings: "
        f"mode={mode}; "
        f"object_name={settings.siril_object_name or '<empty>'}; "
        f"ra_dec={settings.siril_ra_dec or '<empty>'}; "
        f"focal_length={settings.siril_focal_length or '<empty>'}; "
        f"pixel_size={settings.siril_pixel_size or '<empty>'}; "
        f"apply_scnr={settings.siril_apply_scnr}; "
        f"color_saturation={settings.siril_color_saturation}; "
        f"deconvolution_enabled={getattr(settings, 'siril_deconvolution_enabled', False)}; "
        f"deconvolution_layer={use_deconvolution_layer}; "
        f"debug_mode={settings.siril_debug_mode}"
    )
    _log_existing_image(siril_input, write_log, "siril_input.tif")

    if mode == "Siril Photometric":
        script_path = create_photometric_color_script(
            siril_input,
            siril_output_fit,
            job_folder,
            optional_object_name=settings.siril_object_name.strip() or None,
            optional_ra_dec=settings.siril_ra_dec.strip() or None,
            optional_focal_length=settings.siril_focal_length.strip() or None,
            optional_pixel_size=settings.siril_pixel_size.strip() or None,
            apply_scnr=settings.siril_apply_scnr,
            color_saturation=settings.siril_color_saturation,
        )
    else:
        script_path = create_basic_color_script(
            siril_input,
            siril_output_fit,
            job_folder,
            apply_scnr=settings.siril_apply_scnr,
            color_saturation=settings.siril_color_saturation,
        )

    write_log(f"Siril script: {script_path}")
    try:
        run_siril_script(siril_exe, script_path, job_folder, write_log)
    except Exception as exc:
        if mode in {"Basic", "Siril Photometric"}:
            write_log(f"Siril {mode} failed; using Python fallback color calibration. Error: {exc}")
            return _run_python_fallback_calibration(working, stretched, calibrated, settings, write_log)
        raise
    if not siril_output_fit.exists():
        raise RuntimeError(f"Siril completed but did not create {siril_output_fit}")
    write_log("Siril color calibration succeeded.")
    _log_existing_image(siril_output_fit, write_log, "siril_output.fit")

    deconvolved_output_fit = job_folder / "siril_deconvolved.fit"
    if use_deconvolution_layer:
        write_log("Siril Richardson-Lucy deconvolution test enabled as a galaxy-detail layer.")
        deconvolution_script = create_basic_color_script(
            siril_input,
            deconvolved_output_fit,
            job_folder,
            apply_scnr=settings.siril_apply_scnr,
            color_saturation=settings.siril_color_saturation,
            enable_deconvolution=True,
            deconvolution_iterations=14,
            deconvolution_alpha=1800,
        )
        write_log(f"Siril deconvolution layer script: {deconvolution_script}")
        try:
            run_siril_script(siril_exe, deconvolution_script, job_folder, write_log)
        except Exception as exc:
            write_log(f"Siril deconvolution layer failed; continuing with normal Siril output. Error: {exc}")
        else:
            if deconvolved_output_fit.exists():
                write_log("Siril Richardson-Lucy deconvolution layer succeeded.")
                _log_existing_image(deconvolved_output_fit, write_log, "siril_deconvolved.fit")
            else:
                write_log("Siril deconvolution layer completed but did not create output; continuing without it.")

    siril_image = load_image(siril_output_fit, write_log)
    siril_image = np.flipud(siril_image)
    write_log("Corrected Siril FITS orientation with vertical flip.")
    deconvolution_image = None
    if use_deconvolution_layer and deconvolved_output_fit.exists():
        deconvolution_image = load_image(deconvolved_output_fit, write_log)
        deconvolution_image = np.flipud(deconvolution_image)
    raw_siril = job_folder / "siril_calibrated.tif"
    save_tiff(raw_siril, siril_image, write_log)
    _log_existing_image(raw_siril, write_log, "siril_calibrated.tif")

    if mode == "Basic":
        chroma_95 = chroma_percentile(siril_image, 95.0)
        emission_score = red_emission_dominance(siril_image)
        write_log(
            f"Siril Basic object type: {object_type}; "
            f"chroma p95={chroma_95:.5f}; red_emission_dominance={emission_score:.3f}"
        )
        if object_type == "galaxy":
            if darkroom_small_galaxy:
                write_log("Object type is Galaxy; applying raw Siril small-galaxy darkroom finish.")
                finished_image = apply_small_galaxy_darkroom_look(siril_image, write_log)
            elif deconvolution_image is not None:
                write_log("Object type is Galaxy with Siril deconvolution; using broadband finish without heavy background cleanup.")
                finished_image = apply_broadband_look(siril_image, write_log)
            else:
                write_log("Object type is Galaxy; using neutral broadband finish with protected background cleanup.")
                finished_image = _apply_broadband_background_cleanup(siril_image, job_folder, settings, write_log, "galaxy")
            if deconvolution_image is not None:
                write_log("Applying Siril deconvolution detail after galaxy finish.")
                finished_image = blend_galaxy_deconvolution_detail(finished_image, deconvolution_image, write_log)
        elif object_type == "star cluster":
            write_log("Object type is Star Cluster; using neutral star-preserving broadband finish.")
            finished_image = _apply_broadband_background_cleanup(siril_image, job_folder, settings, write_log, "star_cluster")
        elif emission_score < 3.0:
            write_log("Nebula mode selected, but broadband-like color detected; using neutral broadband finish.")
            finished_image = _apply_broadband_background_cleanup(siril_image, job_folder, settings, write_log, "broadband")
        else:
            write_log("Object type is Nebula; using emission nebula color finish.")
            finished_image = apply_goal_look(siril_image, write_log, stretch=False)
    else:
        write_log("Siril PCC succeeded; preserving Siril photometric color without manual color shaping.")
        finished_image = siril_image

    save_tiff(calibrated, finished_image, write_log)
    _log_existing_image(calibrated, write_log, "calibrated.tif")
    shutil.copy2(calibrated, stretched)
    _log_existing_image(stretched, write_log, "stretched.tif")
    return calibrated


def _run_python_fallback_calibration(
    working: Path,
    stretched: Path,
    calibrated: Path,
    settings: AppSettings,
    write_log: LogCallback,
) -> Path:
    source = load_image(working, write_log)
    python_color = python_fallback_color_calibration(source, write_log)
    emission_score = red_emission_dominance(python_color)
    object_type = _normalized_object_type(settings)
    write_log(f"Python fallback object type: {object_type}; red_emission_dominance={emission_score:.3f}")
    if object_type == "nebula":
        calibrated_image = apply_goal_look(python_color, write_log, stretch=False)
    else:
        calibrated_image = apply_broadband_look(python_color, write_log)
    save_tiff(calibrated, calibrated_image, write_log)
    _log_existing_image(calibrated, write_log, "calibrated.tif")
    shutil.copy2(calibrated, stretched)
    _log_existing_image(stretched, write_log, "stretched.tif")
    return calibrated


def run_pipeline(input_path: Path, settings: AppSettings, mode: PipelineMode, log: LogCallback) -> dict[str, Path]:
    input_path = Path(input_path)
    if not is_supported_input(input_path):
        raise ValueError(f"Unsupported input file: {input_path.suffix}")

    output_root = Path(settings.output_folder)
    output_root.mkdir(parents=True, exist_ok=True)
    job_folder = create_job_folder(output_root, input_path)
    log_file = job_folder / "processing_log.txt"

    def write_log(message: str) -> None:
        log(message)
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    write_log(f"DeepSky job: {job_folder}")
    original = job_folder / input_path.name
    shutil.copy2(input_path, original)
    write_log(f"Copied original: {original.name}")
    _log_existing_image(original, write_log, "original")
    analysis = None
    detected_telescope = "generic"
    try:
        detected_telescope = detect_telescope_profile(original)
        write_log(f"Detected telescope profile: {detected_telescope}.")
        analysis = analyze_input_stretch(original)
        metrics = ", ".join(f"{key}={value:.4f}" for key, value in analysis.metrics.items())
        if analysis.likely_stretched:
            write_log(f"WARNING: Pre-stretched input suspected ({analysis.confidence} confidence). {analysis.message}")
            write_log(f"Input stretch analysis: {metrics}")
        else:
            write_log(f"Input stretch analysis: {analysis.message} ({metrics})")
        write_log(f"Auto input recommendation: {analysis.recommended_mode} ({analysis.recommended_reason}).")
    except Exception as exc:
        write_log(f"Input stretch analysis skipped: {exc}")

    before_preview = job_folder / "before_preview.png"
    make_preview(original, before_preview, log=write_log)

    working = job_folder / "working.tif"
    stretched = job_folder / "stretched.tif"
    calibrated = job_folder / "calibrated.tif"
    denoised = job_folder / "denoised.tif"
    starless = job_folder / "starless.tif"
    stars = job_folder / "stars.tif"
    starless_test = job_folder / "starless_test.tif"
    starless_test_stars = job_folder / "starless_test_stars.tif"
    final = job_folder / "final.tif"
    final_png = job_folder / "final.png"

    write_log("Creating 16-bit working TIFF.")
    convert_to_working_tiff(original, working, write_log)
    _log_existing_image(working, write_log, "working.tif")

    input_mode = _normalized_input_mode(settings)
    stretch_level = _normalized_stretch_level(settings)
    object_type = _normalized_object_type(settings)
    siril_deconvolution_requested = bool(getattr(settings, "siril_deconvolution_enabled", False)) and object_type == "galaxy"
    star_setting_raw = str(getattr(settings, "star_handling_mode", "") or "").strip().lower()
    if star_setting_raw in {"starless", "zero stars", "no stars"}:
        star_handling_mode = "starless"
    elif star_setting_raw in {"slight", "slight star reduction", "star reduction", "reduced", "reduce"}:
        star_handling_mode = "slight"
    elif bool(getattr(settings, "starless_test_enabled", False)):
        star_handling_mode = "slight"
    else:
        star_handling_mode = "standard"
    starless_test_requested = star_handling_mode in {"slight", "starless"}
    starless_only_requested = star_handling_mode == "starless"
    write_log(f"Selected stretch level: {stretch_level}.")
    write_log(f"Star settings mode: {star_handling_mode}.")
    use_prestretched = bool(getattr(settings, "prestretched_input", False)) or input_mode == "pre_stretched"
    use_protected_raw_finish = False
    if input_mode == "auto" and analysis is not None:
        use_prestretched = analysis.recommended_mode == "pre_stretched"
        use_protected_raw_finish = analysis.recommended_mode == "gentle_stretch"
        write_log(f"Auto input mode selected: {analysis.recommended_mode}.")
    elif input_mode == "linear":
        use_prestretched = False
        write_log("Manual input mode selected: linear.")
    elif use_prestretched:
        write_log("Manual input mode selected: pre-stretched.")

    use_protected_raw_path = not use_prestretched
    if use_protected_raw_path:
        use_protected_raw_finish = True
        if detected_telescope == "seestar":
            write_log("SeeStar metadata detected; using protected raw baseline stretch path.")
        else:
            write_log("Raw input detected; using protected SeeStar-style baseline stretch path.")

    working_image_for_routing = load_image(working, write_log)
    green_duoband_raw = False

    gradient_galaxy_siril = False
    gradient_galaxy_spread = 0.0
    if mode == PipelineMode.FULL and settings.color_calibration_mode != "Off":
        gradient_galaxy_siril, gradient_galaxy_spread = _needs_siril_for_gradient_galaxy(
            working_image_for_routing,
            analysis,
            detected_telescope,
            object_type,
        )
        if gradient_galaxy_siril:
            write_log(
                "Small faint SeeStar galaxy with strong working-background spread detected; "
                f"using Siril galaxy cleanup path. spread={gradient_galaxy_spread:.3f}"
            )

    should_use_siril_calibration = settings.color_calibration_mode != "Off" and (
        mode == PipelineMode.SIRIL
        or (mode == PipelineMode.FULL and use_prestretched)
        or gradient_galaxy_siril
        or (mode == PipelineMode.FULL and siril_deconvolution_requested)
    )
    if should_use_siril_calibration:
        if siril_deconvolution_requested and not gradient_galaxy_siril and not use_prestretched and mode == PipelineMode.FULL:
            write_log("Siril deconvolution requested; routing galaxy run through Siril calibration path.")
        write_log("Siril calibration path enabled for this run; applying it to the working TIFF.")
        _run_siril_calibration(
            original,
            working,
            stretched,
            calibrated,
            job_folder,
            settings,
            write_log,
            darkroom_small_galaxy=gradient_galaxy_siril,
        )
    elif use_prestretched:
        write_log("Pre-stretched input mode enabled; skipping DeepSky/Siril initial stretch.")
        write_log(f"Applying pre-stretched object finish for: {object_type}")
        source = load_image(working, write_log)
        if object_type == "galaxy":
            calibrated_image = apply_prestretched_broadband_look(source, write_log)
            save_tiff(calibrated, calibrated_image, write_log)
        elif object_type == "star cluster":
            calibrated_image = apply_prestretched_broadband_look(source, write_log)
            save_tiff(calibrated, calibrated_image, write_log)
        else:
            write_log("Pre-stretched nebula selected; revealing existing RGB pixels without raw re-stretch.")
            calibrated_image = apply_prestretched_nebula_rgb_reveal(source, write_log)
            save_tiff(calibrated, calibrated_image, write_log)
        shutil.copy2(calibrated, stretched)
        _log_existing_image(stretched, write_log, "stretched.tif")
        _log_existing_image(calibrated, write_log, "calibrated.tif")
    elif use_protected_raw_finish:
        working_image = working_image_for_routing
        green_duoband_raw = object_type == "nebula" and _looks_like_green_duoband_raw(working_image, analysis)
        base_strength, baseline_reason = _auto_baseline_stretch_strength(
            analysis,
            detected_telescope,
            object_type,
            gentle_recommended=False,
        )
        stretch_source_image = working_image
        if green_duoband_raw:
            base_strength = "gentle"
            baseline_reason = "green-dominant low-dynamic duo-band raw frame; avoiding SeeStar-style background lift"
            stretch_source_image = load_image(original, write_log)
        stretch_strength = _adjust_stretch_strength(base_strength, stretch_level)
        write_log(f"Auto stretch baseline: {base_strength} ({baseline_reason}).")
        write_log(f"Applying {stretch_strength} stretch after user adjustment: {stretch_level}.")
        stretched_image = astrophotography_stretch(stretch_source_image, strength=stretch_strength)
        save_tiff(stretched, stretched_image, write_log)
        _log_existing_image(stretched, write_log, "stretched.tif")
        if green_duoband_raw:
            palette = str(getattr(settings, "duoband_palette", "warm") or "warm").strip().lower()
            write_log(f"Green-dominant duo-band raw finish: applying {palette} color lift.")
            calibrated_image = _apply_duoband_nebula_finish(stretched_image, write_log, palette)
            save_tiff(calibrated, calibrated_image, write_log)
        elif object_type in {"galaxy", "star cluster"}:
            write_log(f"Applying protected raw broadband finish for: {object_type}.")
            calibrated_image = apply_prestretched_broadband_look(stretched_image, write_log)
            save_tiff(calibrated, calibrated_image, write_log)
        else:
            write_log("Applying protected raw nebula finish.")
            calibrated_image = apply_goal_look(stretched_image, write_log, stretch=False)
            save_tiff(calibrated, calibrated_image, write_log)
        _log_existing_image(calibrated, write_log, "calibrated.tif")
    elif mode == PipelineMode.STRETCH:
        base_strength, baseline_reason = _auto_baseline_stretch_strength(
            analysis,
            detected_telescope,
            object_type,
            gentle_recommended=False,
        )
        stretch_strength = _adjust_stretch_strength(base_strength, stretch_level)
        write_log(f"Auto stretch baseline: {base_strength} ({baseline_reason}).")
        write_log(f"Applying local astrophotography stretch: {stretch_strength} after user adjustment: {stretch_level}.")
        _run_local_stretch_calibration(working, stretched, calibrated, write_log, strength=stretch_strength)
    else:
        _run_siril_calibration(original, working, stretched, calibrated, job_folder, settings, write_log)

    current = calibrated
    preserve_siril_galaxy_finish = gradient_galaxy_siril or (
        siril_deconvolution_requested and mode == PipelineMode.FULL
    )
    skip_siril_galaxy_star_reduction = (
        preserve_siril_galaxy_finish
        and object_type == "galaxy"
        and not starless_only_requested
        and _is_compact_siril_galaxy(analysis)
    )

    if mode in {PipelineMode.FULL, PipelineMode.DEEPSNR} and not preserve_siril_galaxy_finish:
        deepsnr_exe = find_executable(Path(settings.deepsnr_folder))
        if not deepsnr_exe:
            raise FileNotFoundError("DeepSNR executable not found. Update the DeepSNR path in settings.")
        write_log(f"DeepSNR executable: {deepsnr_exe}")
        run_deepsnr(current, denoised, deepsnr_exe, write_log)
        _log_existing_image(denoised, write_log, "denoised.tif")
        current = denoised
    elif preserve_siril_galaxy_finish and mode == PipelineMode.FULL:
        write_log("Skipping generic DeepSNR stage; Siril finish already applied.")

    if mode in {PipelineMode.FULL, PipelineMode.STARNET} and not preserve_siril_galaxy_finish:
        if mode == PipelineMode.STARNET and not denoised.exists():
            shutil.copy2(current, denoised)
            _log_existing_image(denoised, write_log, "denoised.tif")
            current = denoised
        starnet_exe = find_executable(Path(settings.starnet_folder))
        if not starnet_exe:
            raise FileNotFoundError("StarNet executable not found. Update the StarNet path in settings.")
        write_log(f"StarNet executable: {starnet_exe}")
        run_starnet(current, starless, starnet_exe, write_log)
        _log_existing_image(starless, write_log, "starless.tif")
        subtract_images(current, starless, stars)
        _log_existing_image(stars, write_log, "stars.tif")
        gentle_nebula_star_reduction = False
        if starless_test_requested and object_type == "nebula" and not green_duoband_raw:
            gentle_nebula_star_reduction = _needs_gentle_nebula_star_reduction(analysis, load_image(current, write_log))
        if starless_test_requested and object_type == "nebula" and not green_duoband_raw and not gentle_nebula_star_reduction:
            write_log("Enhancing starless nebula dust/detail before star recombination.")
            enhanced_starless = apply_starless_nebula_detail(load_image(starless, write_log), write_log)
            save_tiff(starless, enhanced_starless, write_log)
            _log_existing_image(starless, write_log, "enhanced starless.tif")
        elif gentle_nebula_star_reduction:
            write_log("Reflection-style nebula detected; using gentle star reduction without starless dust enhancer.")
        elif green_duoband_raw:
            write_log("Skipping starless nebula dust/detail enhancer for green-dominant duo-band raw frame.")
        if starless_test_requested:
            if starless_only_requested:
                write_log("Starless enabled; keeping StarNet starless image without recombining the star layer.")
                shutil.copy2(starless, final)
            else:
                keep_fraction = 0.40 if gentle_nebula_star_reduction else (0.10 if object_type == "nebula" else 0.60)
                write_log(f"Slight Star Reduction enabled; recombining starless image with brightest {keep_fraction:.0%} of stars.")
                threshold = add_bright_star_fraction(starless, stars, final, keep_fraction=keep_fraction)
                write_log(f"Star reduction kept bright stars with layer threshold {threshold:.1f}.")
            if object_type == "nebula" and not gentle_nebula_star_reduction:
                write_log("Applying PixInsight-style nebula finish.")
                pixinsight_nebula = apply_pixinsight_style_nebula_finish(load_image(final, write_log), write_log)
                save_tiff(final, pixinsight_nebula, write_log)
        else:
            if object_type == "nebula" and not green_duoband_raw:
                write_log("Reducing faint nebula star/noise layer before recombination.")
                low, high = add_weighted_star_layer(starless, stars, final)
                write_log(f"Weighted nebula star layer: low={low:.1f}, high={high:.1f}.")
                write_log("Applying PixInsight-style nebula finish.")
                pixinsight_nebula = apply_pixinsight_style_nebula_finish(load_image(final, write_log), write_log)
                save_tiff(final, pixinsight_nebula, write_log)
            else:
                add_images(starless, stars, final)
        _log_existing_image(final, write_log, "final.tif")
        current = final
    elif preserve_siril_galaxy_finish and mode == PipelineMode.FULL:
        write_log("Skipping StarNet stage for Siril finish.")

    if mode in {PipelineMode.STRETCH, PipelineMode.DEEPSNR, PipelineMode.SIRIL} or (
        preserve_siril_galaxy_finish and mode == PipelineMode.FULL
    ):
        shutil.copy2(current, final)
        _log_existing_image(final, write_log, "final.tif")

    if (
        starless_test_requested
        and (preserve_siril_galaxy_finish or mode not in {PipelineMode.FULL, PipelineMode.STARNET})
        and not skip_siril_galaxy_star_reduction
    ):
        starnet_exe = find_executable(Path(settings.starnet_folder))
        if not starnet_exe:
            raise FileNotFoundError("StarNet executable not found. Update the StarNet path in settings.")
        write_log(f"{'Starless' if starless_only_requested else 'Slight Star Reduction'} enabled; running StarNet on final image.")
        write_log(f"StarNet executable: {starnet_exe}")
        run_starnet(final, starless_test, starnet_exe, write_log)
        _log_existing_image(starless_test, write_log, "starless_test.tif")
        subtract_images(final, starless_test, starless_test_stars)
        _log_existing_image(starless_test_stars, write_log, "starless_test_stars.tif")
        if starless_only_requested:
            write_log("Starless enabled; keeping final StarNet starless image without recombining the star layer.")
            shutil.copy2(starless_test, final)
        else:
            keep_fraction = 0.10 if object_type == "nebula" or preserve_siril_galaxy_finish else 0.60
            threshold = add_bright_star_fraction(starless_test, starless_test_stars, final, keep_fraction=keep_fraction)
            write_log(f"Star reduction kept bright stars with layer threshold {threshold:.1f}.")
            if object_type == "nebula":
                write_log("Applying PixInsight-style nebula finish.")
                pixinsight_nebula = apply_pixinsight_style_nebula_finish(load_image(final, write_log), write_log)
                save_tiff(final, pixinsight_nebula, write_log)
        _log_existing_image(final, write_log, "final.tif")
    elif starless_test_requested and skip_siril_galaxy_star_reduction:
        write_log("Star reduction skipped for compact Siril deconvolution galaxy finish to preserve detail.")

    if object_type == "nebula" and final.exists():
        write_log("Cropping nebula stacking edges before export.")
        edge_cropped = _crop_edge_artifacts(load_image(final, write_log), fraction=0.025)
        save_tiff(final, edge_cropped, write_log)
        _log_existing_image(final, write_log, "edge-cropped final.tif")

    save_png(final_png, load_image(final, write_log), write_log)
    after_preview = job_folder / "after_preview.png"
    preview_source = calibrated if mode == PipelineMode.SIRIL else final
    make_preview(preview_source, after_preview, log=write_log, stretch_for_display=False)
    calibrated_preview = job_folder / "calibrated_preview.png"
    make_preview(calibrated, calibrated_preview, log=write_log, stretch_for_display=False)
    write_log(f"Final image: {final}")
    write_log("Done.")

    return {
        "job_folder": job_folder,
        "before_preview": before_preview,
        "after_preview": after_preview,
        "final": final,
        "png": final_png,
        "log": log_file,
    }
