from __future__ import annotations

import shutil
import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .cli_tools import find_executable, run_deepsnr, run_starnet
from .goal_look import (
    apply_broadband_look,
    apply_pcc_galaxy_look,
    apply_cosmos_style_nebula_finish,
    apply_goal_look,
    apply_measured_color_to_nebula_detail,
    apply_natural_nebula_rgb_look,
    apply_pixinsight_style_nebula_finish,
    apply_prestretched_broadband_look,
    apply_prestretched_nebula_rgb_reveal,
    apply_small_galaxy_darkroom_look,
    apply_starless_nebula_detail,
    blend_galaxy_deconvolution_detail,
    blend_broadband_background_denoise,
    chroma_percentile,
    compose_pixinsight_nebula_layers,
    reflection_nebula_bias,
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
from .plate_solver import find_astap_database, find_astap_executable, solve_image, write_plate_solve_debug, write_wcs_enriched_fits
from .python_color_calibration import python_fallback_color_calibration
from .settings import AppSettings
from .siril_cli import (
    build_siril_pcc_command,
    create_basic_color_script,
    create_nebula_local_color_script,
    create_photometric_color_script,
    find_local_spcc_catalog,
    find_siril_executable,
    run_siril_script,
    siril_catalog_calibration_path,
)
from .stretch import astrophotography_stretch
from .target_identifier import identify_target


LogCallback = Callable[[str], None]


class PipelineMode(str, Enum):
    FULL = "full"
    STRETCH = "stretch"
    DEEPSNR = "deepsnr"
    STARNET = "starnet"
    SIRIL = "siril"


class PccCalibrationFailed(RuntimeError):
    """Raised when catalog-backed Siril PCC fails and the web UI should ask the user."""


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
        ["seestar_weak_nebula", "seestar_slight", "seestar", "seestar_aggressive", "seestar_extra_aggressive"],
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


def _looks_like_green_duoband_raw(
    image: np.ndarray,
    analysis: object | None,
    source_name: str = "",
) -> bool:
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
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    metrics = getattr(analysis, "metrics", {}) if analysis is not None else {}
    raw_p50 = float(metrics.get("raw_p50", np.percentile(rgb, 50.0)))
    raw_p999 = float(metrics.get("raw_p999", np.percentile(rgb, 99.9)))

    compressed_raw = raw_p50 > 0.045 and raw_p999 < 0.16
    green_median = medians[1] > max(medians[0], medians[2]) * 1.10 + 0.004
    green_high = highs[1] > max(highs[0], highs[2]) * 1.02 + 0.004

    normalized_name = source_name.lower().replace("_", " ").replace("-", " ")
    named_duoband = (
        ("duo" in normalized_name and "band" in normalized_name)
        or "dual band" in normalized_name
        or "dualband" in normalized_name
    )
    lifted_gradient_duoband = (
        named_duoband
        and _working_background_spread(arr) > 0.55
        and raw_p50 > 0.055
        and raw_p999 < 0.42
        and float(np.percentile(chroma, 95.0)) > 0.07
    )
    return bool((compressed_raw and green_median and green_high) or lifted_gradient_duoband)


def _looks_like_weak_snr_nebula_raw(analysis: object | None) -> bool:
    metrics = getattr(analysis, "metrics", {}) if analysis is not None else {}
    raw_p99 = float(metrics.get("raw_p99", 0.0))
    raw_p999 = float(metrics.get("raw_p999", 0.0))
    bright_fraction = float(metrics.get("bright_fraction", 0.0))
    shadow_fraction = float(metrics.get("shadow_fraction", 0.0))
    midtone_fraction = float(metrics.get("midtone_fraction", 0.0))
    return bool(
        raw_p999 < 0.018
        and raw_p99 < 0.014
        and bright_fraction < 0.00022
        and shadow_fraction > 0.990
        and midtone_fraction < 0.0015
    )


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


def _flatten_lifted_duoband_gradient(image: np.ndarray, write_log: LogCallback) -> np.ndarray:
    rgb = _to_float01(image)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        return image

    height, width = rgb.shape[:2]
    if height < 64 or width < 64:
        return image

    lum = _rgb_luminance(rgb)
    before_top = float(np.median(lum[: max(1, height // 4), :]))
    before_bottom = float(np.median(lum[height - max(1, height // 4) :, :]))

    row_background = np.percentile(rgb, 18.0, axis=1).astype(np.float32)
    sigma_y = max(20.0, height * 0.055)
    kernel_y = max(3, int(round(sigma_y * 6.0)) | 1)
    row_background = cv2.GaussianBlur(
        row_background.reshape(height, 1, 3),
        (1, kernel_y),
        0.0,
        sigma_y,
        borderType=cv2.BORDER_REPLICATE,
    ).reshape(height, 3)
    column_background = np.percentile(rgb, 18.0, axis=0).astype(np.float32)
    sigma_x = max(20.0, width * 0.050)
    kernel_x = max(3, int(round(sigma_x * 6.0)) | 1)
    column_background = cv2.GaussianBlur(
        column_background.reshape(1, width, 3),
        (kernel_x, 1),
        sigma_x,
        0.0,
        borderType=cv2.BORDER_REPLICATE,
    ).reshape(width, 3)

    grid_rows = 10
    grid_cols = 14
    coarse = np.zeros((grid_rows, grid_cols, 3), dtype=np.float32)
    for row in range(grid_rows):
        y0 = row * height // grid_rows
        y1 = (row + 1) * height // grid_rows
        for col in range(grid_cols):
            x0 = col * width // grid_cols
            x1 = (col + 1) * width // grid_cols
            tile = rgb[y0:y1, x0:x1, :]
            coarse[row, col, :] = np.percentile(tile.reshape(-1, 3), 14.0, axis=0)
    coarse_background = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    coarse_background = cv2.GaussianBlur(
        coarse_background,
        (0, 0),
        max(16.0, width * 0.018),
        max(16.0, height * 0.018),
        borderType=cv2.BORDER_REPLICATE,
    )

    anchor = np.percentile(coarse, 8.0, axis=(0, 1)).astype(np.float32)
    row_excess = np.clip(row_background - anchor.reshape(1, 3), 0.0, 1.0)[:, None, :]
    column_excess = np.clip(column_background - anchor.reshape(1, 3), 0.0, 1.0)[None, :, :]
    coarse_excess = np.clip(coarse_background - anchor.reshape(1, 1, 3), 0.0, 1.0)
    excess = np.maximum(coarse_excess, np.maximum(row_excess * 0.86, column_excess * 0.82))

    flattened = np.clip(rgb - excess * 0.92, 0.0, 1.0)

    flattened_lum = _rgb_luminance(flattened)
    detail_floor = float(np.percentile(lum, 2.0))
    flattened += np.clip(detail_floor - np.percentile(flattened_lum, 2.0), 0.0, 0.04)
    flattened = np.clip(flattened, 0.0, 1.0)

    after_lum = _rgb_luminance(flattened)
    after_top = float(np.median(after_lum[: max(1, height // 4), :]))
    after_bottom = float(np.median(after_lum[height - max(1, height // 4) :, :]))
    write_log(
        "Flattened lifted duo-band gradient before stretch: "
        f"top_median={before_top:.5f}->{after_top:.5f}; "
        f"bottom_median={before_bottom:.5f}->{after_bottom:.5f}; "
        f"row_excess_p95={float(np.percentile(excess, 95.0)):.5f}"
    )
    return np.clip(flattened * 65535.0, 0.0, 65535.0).round().astype(np.uint16)


def _crop_edge_artifacts(
    image: np.ndarray,
    fraction: float = 0.06,
    max_side_fraction: float = 0.12,
    max_total_fraction: float = 0.18,
) -> np.ndarray:
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

    top = bottom = 0
    left = right = 0

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
        bright_color_noise = (
            worst_std > interior_std * 1.34 + 0.004
            and worst_chroma > interior_chroma_p95 * 1.38 + 0.010
            and darkest_median < interior_p95 * 1.08
        )
        extreme_noise = (
            worst_std > interior_std * 2.05 + 0.006
            and worst_chroma > interior_chroma_p95 * 1.55 + 0.010
            and darkest_median < interior_p95 * 0.92
        )
        return bool(
            dark_and_noisy or bright_color_noise or extreme_noise
        )

    while top < max_y and bad_luminance(lum[top : top + stripe_y, :], chroma[top : top + stripe_y, :], vertical=False):
        top += step_y
    while bottom < max_y and bad_luminance(lum[height - bottom - stripe_y : height - bottom, :], chroma[height - bottom - stripe_y : height - bottom, :], vertical=False):
        bottom += step_y
    while left < max_x and bad_luminance(lum[:, left : left + stripe_x], chroma[:, left : left + stripe_x], vertical=True):
        left += step_x
    while right < max_x and bad_luminance(lum[:, width - right - stripe_x : width - right], chroma[:, width - right - stripe_x : width - right], vertical=True):
        right += step_x

    max_side_crop_y = int(round(height * max_side_fraction))
    max_side_crop_x = int(round(width * max_side_fraction))
    max_total_crop_y = int(round(height * max_total_fraction))
    max_total_crop_x = int(round(width * max_total_fraction))
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


def _crop_lifted_duoband_artifacts(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim < 2:
        return arr
    height, width = arr.shape[:2]
    if height < 128 or width < 128:
        return arr

    rgb = _to_float01(arr)
    lum = _rgb_luminance(rgb)
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    inset_y = int(round(height * 0.22))
    inset_x = int(round(width * 0.22))
    interior = lum[inset_y : height - inset_y, inset_x : width - inset_x]
    interior_chroma = chroma[inset_y : height - inset_y, inset_x : width - inset_x]
    if interior.size < 1024:
        return arr

    interior_lum = float(np.percentile(interior, 72.0))
    interior_chroma = float(np.percentile(interior_chroma, 88.0))
    step_y = max(8, int(round(height * 0.012)))
    step_x = max(8, int(round(width * 0.012)))
    stripe_y = max(12, int(round(height * 0.030)))
    stripe_x = max(12, int(round(width * 0.030)))
    max_top = int(round(height * 0.08))
    max_bottom = int(round(height * 0.18))
    max_left = int(round(width * 0.18))
    max_right = int(round(width * 0.08))

    def bad_horizontal(start: int, stop: int) -> bool:
        stripe_lum = lum[start:stop, :]
        stripe_chroma = chroma[start:stop, :]
        return bool(
            float(np.percentile(stripe_lum, 82.0)) > interior_lum * 1.20 + 0.010
            or float(np.percentile(stripe_chroma, 92.0)) > interior_chroma * 1.35 + 0.012
        )

    def bad_vertical(start: int, stop: int) -> bool:
        stripe_lum = lum[:, start:stop]
        stripe_chroma = chroma[:, start:stop]
        return bool(
            float(np.percentile(stripe_lum, 82.0)) > interior_lum * 1.20 + 0.010
            or float(np.percentile(stripe_chroma, 92.0)) > interior_chroma * 1.35 + 0.012
        )

    top = bottom = left = right = 0
    while top < max_top and bad_horizontal(top, min(height, top + stripe_y)):
        top += step_y
    while bottom < max_bottom and bad_horizontal(max(0, height - bottom - stripe_y), height - bottom):
        bottom += step_y
    while left < max_left and bad_vertical(left, min(width, left + stripe_x)):
        left += step_x
    while right < max_right and bad_vertical(max(0, width - right - stripe_x), width - right):
        right += step_x

    top = max(top, int(round(height * 0.018)))
    bottom = max(bottom, int(round(height * 0.130)))
    left = max(left, int(round(width * 0.075)))
    right = max(right, int(round(width * 0.025)))
    if height - top - bottom < 96 or width - left - right < 96:
        return arr
    return arr[top : height - bottom, left : width - right].copy()


def _clean_lifted_nebula_color_borders(image: np.ndarray, write_log: LogCallback) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim < 3 or arr.shape[-1] < 3:
        return arr
    height, width = arr.shape[:2]
    if height < 180 or width < 180:
        return arr

    rgb = _to_float01(arr)
    lum = _rgb_luminance(rgb)
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    green_excess = np.clip(rgb[..., 1] - (rgb[..., 0] * 0.58 + rgb[..., 2] * 0.42), 0.0, 1.0)

    inset_y = int(round(height * 0.24))
    inset_x = int(round(width * 0.24))
    interior = np.s_[inset_y : height - inset_y, inset_x : width - inset_x]
    if lum[interior].size < 1024:
        return arr
    interior_chroma = float(np.percentile(chroma[interior], 88.0))
    interior_green = float(np.percentile(green_excess[interior], 88.0))
    interior_lum = float(np.percentile(lum[interior], 82.0))

    step_y = max(8, int(round(height * 0.012)))
    step_x = max(8, int(round(width * 0.012)))
    stripe_y = max(14, int(round(height * 0.025)))
    stripe_x = max(14, int(round(width * 0.025)))
    max_top = int(round(height * 0.10))
    max_bottom = int(round(height * 0.18))
    max_left = int(round(width * 0.22))
    max_right = int(round(width * 0.22))

    def bad_slice(region: tuple[slice, slice]) -> bool:
        region_lum = lum[region]
        region_chroma = chroma[region]
        region_green = green_excess[region]
        if region_lum.size < 64:
            return False
        bright_noisy = float(np.percentile(region_lum, 82.0)) > interior_lum * 1.10 + 0.008
        colored_noisy = float(np.percentile(region_chroma, 92.0)) > interior_chroma * 1.22 + 0.010
        green_wall = float(np.percentile(region_green, 90.0)) > interior_green * 1.35 + 0.006
        return bool((colored_noisy and bright_noisy) or (green_wall and colored_noisy))

    top = bottom = left = right = 0
    while top < max_top and bad_slice((slice(top, min(height, top + stripe_y)), slice(None))):
        top += step_y
    while bottom < max_bottom and bad_slice((slice(max(0, height - bottom - stripe_y), height - bottom), slice(None))):
        bottom += step_y
    while left < max_left and bad_slice((slice(None), slice(left, min(width, left + stripe_x)))):
        left += step_x
    while right < max_right and bad_slice((slice(None), slice(max(0, width - right - stripe_x), width - right))):
        right += step_x

    if top or bottom or left or right:
        write_log(
            "Detected lifted nebula color borders; preserving image size and applying color cleanup only: "
            f"detected_top={top}, detected_bottom={bottom}, detected_left={left}, detected_right={right}"
        )

    lum = _rgb_luminance(rgb)
    red_signal = np.clip(
        (rgb[..., 0] - (rgb[..., 1] * 0.62 + rgb[..., 2] * 0.38))
        / max(1e-6, float(np.percentile(rgb[..., 0], 99.2) - np.percentile(rgb[..., 0], 35.0))),
        0.0,
        1.0,
    )
    red_signal = cv2.GaussianBlur(red_signal.astype(np.float32), (0, 0), 3.0)
    broad_red = cv2.GaussianBlur(red_signal.astype(np.float32), (0, 0), 13.0)
    red_protect = np.clip(red_signal * (0.16 + broad_red * 1.65), 0.0, 1.0)
    star_signal = np.clip(
        (lum - np.percentile(lum, 97.2))
        / max(1e-6, np.percentile(lum, 99.95) - np.percentile(lum, 97.2)),
        0.0,
        1.0,
    )
    green_target = rgb[..., 0] * 0.60 + rgb[..., 2] * 0.40
    green_excess = np.maximum(0.0, rgb[..., 1] - green_target)
    green_hi = max(1e-6, float(np.percentile(green_excess, 99.2)))
    green_noise = np.clip(green_excess / green_hi, 0.0, 1.0)
    sky_weight = np.clip(
        (np.percentile(lum, 82.0) - lum)
        / max(1e-6, np.percentile(lum, 82.0) - np.percentile(lum, 6.0)),
        0.0,
        1.0,
    )
    reduction = np.clip(green_noise * sky_weight * (1.0 - red_protect * 0.88) * (1.0 - star_signal * 0.86), 0.0, 0.88)
    if float(np.mean(reduction)) > 0.002:
        rgb[..., 1] = np.clip(rgb[..., 1] - green_excess * reduction, 0.0, 1.0)
        write_log(f"Suppressed lifted nebula green speckle noise: reduction_mean={float(np.mean(reduction)):.5f}")

    warm_target = rgb[..., 1] * 0.66 + rgb[..., 2] * 0.34
    warm_excess = np.maximum(0.0, rgb[..., 0] - warm_target)
    warm_hi = max(1e-6, float(np.percentile(warm_excess, 99.2)))
    warm_noise = np.clip(warm_excess / warm_hi, 0.0, 1.0)
    local_lum = cv2.medianBlur((lum * 255.0).astype(np.uint8), 5).astype(np.float32) / 255.0
    point_like = np.clip(
        (lum - local_lum)
        / max(1e-6, float(np.percentile(lum - local_lum, 99.4) - np.percentile(lum - local_lum, 65.0))),
        0.0,
        1.0,
    )
    warm_reduction = np.clip(
        warm_noise
        * (0.34 + point_like * 0.76)
        * sky_weight
        * (1.0 - red_protect * 0.93)
        * (1.0 - star_signal * 0.88),
        0.0,
        0.82,
    )
    if float(np.mean(warm_reduction)) > 0.002:
        neutral_warm = np.clip(lum[..., None] * np.array([0.95, 0.96, 1.00], dtype=np.float32).reshape(1, 1, 3), 0.0, 1.0)
        rgb = np.clip(rgb * (1.0 - warm_reduction[..., None] * 0.72) + neutral_warm * (warm_reduction[..., None] * 0.72), 0.0, 1.0)
        write_log(f"Suppressed lifted nebula warm speckle noise: reduction_mean={float(np.mean(warm_reduction)):.5f}")

    yy, xx = np.indices((height, width), dtype=np.float32)
    edge_distance = np.minimum.reduce([xx, yy, width - 1 - xx, height - 1 - yy])
    edge = np.clip((min(height, width) * 0.12 - edge_distance) / max(1.0, min(height, width) * 0.12), 0.0, 1.0)
    edge = cv2.GaussianBlur(np.clip(edge, 0.0, 1.0).astype(np.float32), (0, 0), 5.0)
    lum = _rgb_luminance(rgb)
    neutral = lum[..., None] * np.array([1.02, 0.96, 0.90], dtype=np.float32).reshape(1, 1, 3)
    mix = np.clip(edge[..., None] * 0.34, 0.0, 0.34)
    cleaned = np.clip(rgb * (1.0 - mix) + neutral * mix, 0.0, 1.0)
    return np.clip(cleaned * 65535.0, 0.0, 65535.0).round().astype(np.uint16)


def _apply_reflection_nebula_color_preserve(image: np.ndarray, write_log: LogCallback) -> np.ndarray:
    """Gentle IC63-style finish: keep real red/brown target color, only tame green cast."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return arr

    rgb = _to_float01(arr)
    lum = _rgb_luminance(rgb).astype(np.float32)
    low = float(np.percentile(lum, 1.0))
    high = float(np.percentile(lum, 99.72))
    rgb = np.clip((rgb - low) / max(1e-6, high - low), 0.0, 1.0)
    rgb = np.clip(rgb, 0.0, 1.0) ** 0.74

    lum = _rgb_luminance(rgb).astype(np.float32)
    star = np.clip(
        (lum - np.percentile(lum, 97.1))
        / max(1e-6, np.percentile(lum, 99.96) - np.percentile(lum, 97.1)),
        0.0,
        1.0,
    ) ** 1.6
    star = cv2.GaussianBlur(star.astype(np.float32), (0, 0), 0.9)
    star_protect = np.clip(
        (lum - np.percentile(lum, 99.15))
        / max(1e-6, np.percentile(lum, 99.985) - np.percentile(lum, 99.15)),
        0.0,
        1.0,
    ) ** 1.35
    star_protect = cv2.GaussianBlur(star_protect.astype(np.float32), (0, 0), 0.8)

    red_brown = np.clip(
        (rgb[..., 0] * 0.72 + rgb[..., 1] * 0.18 - rgb[..., 2] * 0.46)
        / max(1e-6, np.percentile(lum, 99.3) - np.percentile(lum, 24.0)),
        0.0,
        1.0,
    )
    signal = np.clip(
        (lum - np.percentile(lum, 18.0))
        / max(1e-6, np.percentile(lum, 98.8) - np.percentile(lum, 18.0)),
        0.0,
        1.0,
    ) ** 0.72
    nebula = np.clip(red_brown * signal * (1.0 - star * 0.82), 0.0, 1.0)
    nebula = cv2.GaussianBlur(nebula.astype(np.float32), (0, 0), 2.8)
    broad = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 18.0)
    broad_signal = np.clip(
        (broad - np.percentile(broad, 42.0))
        / max(1e-6, np.percentile(broad, 98.7) - np.percentile(broad, 42.0)),
        0.0,
        1.0,
    )
    nebula = np.clip(np.maximum(nebula, broad_signal * red_brown * 0.62) * (1.0 - star * 0.78), 0.0, 1.0)

    lum = _rgb_luminance(rgb).astype(np.float32)
    chroma = rgb - lum[..., None]
    warm = np.clip(lum[..., None] + chroma * (1.0 + nebula[..., None] * 0.72), 0.0, 1.0)
    warm_lum = _rgb_luminance(warm).astype(np.float32)
    warm = np.clip(warm * (lum / np.maximum(warm_lum, 1e-5))[..., None], 0.0, 1.0)
    rgb = np.clip(rgb * (1.0 - nebula[..., None] * 0.42) + warm * (nebula[..., None] * 0.42), 0.0, 1.0)
    write_log("Reflection nebula fallback: preserved measured RGB chroma; no fixed warm/red color vector applied.")

    lum = _rgb_luminance(rgb).astype(np.float32)
    green_excess = np.maximum(0.0, rgb[..., 1] - (rgb[..., 0] * 0.58 + rgb[..., 2] * 0.42))
    sky = np.clip(
        (np.percentile(lum, 58.0) - lum)
        / max(1e-6, np.percentile(lum, 58.0) - np.percentile(lum, 4.0)),
        0.0,
        1.0,
    ) * (1.0 - nebula * 0.92) * (1.0 - star_protect * 0.86)
    reduction = np.clip(sky * 0.82 + green_excess * 2.8 * (1.0 - nebula * 0.72), 0.0, 0.78)
    rgb[..., 1] = np.clip(rgb[..., 1] - green_excess * reduction, 0.0, 1.0)

    lum = _rgb_luminance(rgb).astype(np.float32)
    neutral_sky = np.clip(
        lum[..., None] * np.array([0.94, 0.95, 0.98], dtype=np.float32).reshape(1, 1, 3),
        0.0,
        1.0,
    )
    sky_color_mix = np.clip(sky[..., None] * (1.0 - nebula[..., None] * 0.88) * (1.0 - star[..., None] * 0.92) * 0.62, 0.0, 0.62)
    rgb = np.clip(rgb * (1.0 - sky_color_mix) + neutral_sky * sky_color_mix, 0.0, 1.0)

    lum = _rgb_luminance(rgb).astype(np.float32)
    chroma = rgb - lum[..., None]
    quiet_sky = np.clip(sky * (1.0 - nebula * 0.82) * (1.0 - star_protect * 0.95), 0.0, 1.0)
    subdued_chroma = np.clip(lum[..., None] + chroma * 0.035, 0.0, 1.0)
    chroma_quiet_mix = np.clip(quiet_sky[..., None] * 0.985, 0.0, 0.985)
    rgb = np.clip(rgb * (1.0 - chroma_quiet_mix) + subdued_chroma * chroma_quiet_mix, 0.0, 1.0)

    # Reflection targets like IC 63 often have real red-brown signal sitting in a very
    # noisy star field. Suppress isolated warm chroma in the sky without flattening
    # the nebula mask or touching bright star cores.
    lum = _rgb_luminance(rgb).astype(np.float32)
    chroma = rgb - lum[..., None]
    red_speckle = np.clip(
        (rgb[..., 0] - (rgb[..., 1] * 0.72 + rgb[..., 2] * 0.28))
        / max(1e-6, float(np.percentile(rgb[..., 0], 99.4) - np.percentile(rgb[..., 0], 35.0))),
        0.0,
        1.0,
    )
    red_speckle = np.clip(red_speckle * quiet_sky * (1.0 - nebula * 0.94) * (1.0 - star_protect * 0.96), 0.0, 1.0)
    red_speckle = cv2.GaussianBlur(red_speckle.astype(np.float32), (0, 0), 0.45)
    if float(np.mean(red_speckle)) > 0.001:
        neutral = np.clip(lum[..., None] * np.array([0.94, 0.95, 0.98], dtype=np.float32).reshape(1, 1, 3), 0.0, 1.0)
        rgb = np.clip(rgb * (1.0 - red_speckle[..., None] * 0.82) + neutral * (red_speckle[..., None] * 0.82), 0.0, 1.0)
        write_log(f"Suppressed reflection nebula warm sky speckle: mean={float(np.mean(red_speckle)):.5f}")

    lum = _rgb_luminance(rgb).astype(np.float32)
    chroma = rgb - lum[..., None]
    sky_blur = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), 1.9)
    sky_blur_lum = _rgb_luminance(sky_blur).astype(np.float32)
    sky_blur = np.clip(sky_blur * (lum / np.maximum(sky_blur_lum, 1e-5))[..., None], 0.0, 1.0)
    sky_blur = np.clip(_rgb_luminance(sky_blur)[..., None] + (sky_blur - _rgb_luminance(sky_blur)[..., None]) * 0.18, 0.0, 1.0)
    sky_smooth_mix = np.clip(quiet_sky[..., None] * (1.0 - nebula[..., None] * 0.94) * (1.0 - star_protect[..., None] * 0.96) * 0.90, 0.0, 0.90)
    rgb = np.clip(rgb * (1.0 - sky_smooth_mix) + sky_blur * sky_smooth_mix, 0.0, 1.0)

    smooth = cv2.bilateralFilter((rgb * 255.0).astype(np.uint8), 7, 50, 9).astype(np.float32) / 255.0
    blur_smooth = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), 1.35)
    smooth = np.clip(smooth * 0.62 + blur_smooth * 0.38, 0.0, 1.0)
    denoise_mix = np.clip((1.0 - star_protect[..., None] * 0.98) * (1.0 - nebula[..., None] * 0.84) * (0.42 + quiet_sky[..., None] * 0.84), 0.0, 0.94)
    rgb = np.clip(rgb * (1.0 - denoise_mix) + smooth * denoise_mix, 0.0, 1.0)

    local_rgb = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), 0.85)
    local_noise = np.mean(np.abs(rgb - local_rgb), axis=2)
    noise_weight = np.clip(
        (local_noise - np.percentile(local_noise, 52.0))
        / max(1e-6, np.percentile(local_noise, 97.0) - np.percentile(local_noise, 52.0)),
        0.0,
        1.0,
    )
    median_smooth = cv2.medianBlur((rgb * 255.0).astype(np.uint8), 3).astype(np.float32) / 255.0
    deep_smooth = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), 2.25)
    sky_smooth = np.clip(median_smooth * 0.72 + deep_smooth * 0.28, 0.0, 1.0)
    sky_despeckle_mix = np.clip(
        quiet_sky[..., None]
        * (1.0 - nebula[..., None] * 0.92)
        * (1.0 - star_protect[..., None] * 0.985)
        * (0.58 + noise_weight[..., None] * 0.50),
        0.0,
        0.92,
    )
    rgb = np.clip(rgb * (1.0 - sky_despeckle_mix) + sky_smooth * sky_despeckle_mix, 0.0, 1.0)

    nlm = cv2.fastNlMeansDenoisingColored(
        (rgb * 255.0).astype(np.uint8),
        None,
        9,
        14,
        7,
        21,
    ).astype(np.float32) / 255.0
    nlm_lum = _rgb_luminance(nlm).astype(np.float32)
    nlm_chroma = nlm - nlm_lum[..., None]
    rgb_lum = _rgb_luminance(rgb).astype(np.float32)
    nlm = np.clip(rgb_lum[..., None] + nlm_chroma * 0.82, 0.0, 1.0)
    nlm_mix = np.clip(
        quiet_sky[..., None]
        * (1.0 - nebula[..., None] * 0.86)
        * (1.0 - star_protect[..., None] * 0.94)
        * (0.46 + noise_weight[..., None] * 0.40),
        0.0,
        0.76,
    )
    rgb = np.clip(rgb * (1.0 - nlm_mix) + nlm * nlm_mix, 0.0, 1.0)

    heavy_nlm = cv2.fastNlMeansDenoisingColored(
        (rgb * 255.0).astype(np.uint8),
        None,
        15,
        28,
        7,
        27,
    ).astype(np.float32) / 255.0
    heavy_lum = _rgb_luminance(heavy_nlm).astype(np.float32)
    heavy_chroma = heavy_nlm - heavy_lum[..., None]
    rgb_lum = _rgb_luminance(rgb).astype(np.float32)
    heavy_nlm = np.clip(rgb_lum[..., None] + heavy_chroma * 0.16, 0.0, 1.0)
    background_cleanup = np.clip(
        quiet_sky[..., None]
        * (1.0 - nebula[..., None] * 0.96)
        * (1.0 - star_protect[..., None] * 0.985)
        * (0.56 + noise_weight[..., None] * 0.38),
        0.0,
        0.92,
    )
    rgb = np.clip(rgb * (1.0 - background_cleanup) + heavy_nlm * background_cleanup, 0.0, 1.0)

    lum = _rgb_luminance(rgb).astype(np.float32)
    point_floor = cv2.medianBlur((lum * 255.0).astype(np.uint8), 5).astype(np.float32) / 255.0
    point_excess = np.maximum(0.0, lum - point_floor)
    point_weight = np.clip(
        (point_excess - np.percentile(point_excess, 72.0))
        / max(1e-6, np.percentile(point_excess, 99.55) - np.percentile(point_excess, 72.0)),
        0.0,
        1.0,
    ) ** 0.72
    tiny_speckle = np.clip(
        point_weight
        * quiet_sky
        * (1.0 - nebula * 0.94)
        * (1.0 - star_protect * 0.98),
        0.0,
        1.0,
    )
    point_clean = cv2.medianBlur((rgb * 255.0).astype(np.uint8), 5).astype(np.float32) / 255.0
    point_mix = np.clip(tiny_speckle[..., None] * 0.72, 0.0, 0.72)
    rgb = np.clip(rgb * (1.0 - point_mix) + point_clean * point_mix, 0.0, 1.0)

    lum = _rgb_luminance(rgb).astype(np.float32)
    max_ch = np.max(rgb, axis=2)
    min_ch = np.min(rgb, axis=2)
    saturation = np.clip((max_ch - min_ch) / np.maximum(lum + 0.025, 1e-5), 0.0, 1.0)
    warm_or_green = np.clip(
        np.maximum(rgb[..., 0] - rgb[..., 2] * 0.82, rgb[..., 1] - rgb[..., 2] * 0.74),
        0.0,
        1.0,
    )
    color_speckle = np.clip(
        point_weight
        * saturation
        * (warm_or_green / max(1e-6, float(np.percentile(warm_or_green, 99.2))))
        * quiet_sky
        * (1.0 - nebula * 0.97)
        * (1.0 - star_protect * 0.98),
        0.0,
        1.0,
    )
    color_speckle = cv2.GaussianBlur(color_speckle.astype(np.float32), (0, 0), 0.55)
    if float(np.mean(color_speckle)) > 0.001:
        cleaner_lum = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 1.05)
        cleaner = np.clip(cleaner_lum[..., None] * np.array([0.92, 0.93, 0.98], dtype=np.float32).reshape(1, 1, 3), 0.0, 1.0)
        color_speckle_mix = np.clip(color_speckle[..., None] * 0.82, 0.0, 0.82)
        rgb = np.clip(rgb * (1.0 - color_speckle_mix) + cleaner * color_speckle_mix, 0.0, 1.0)
        write_log(f"Suppressed reflection nebula colored sky speckle: mean={float(np.mean(color_speckle)):.5f}")

    sky_darken = np.clip(
        quiet_sky
        * (1.0 - nebula * 0.84)
        * (1.0 - star_protect * 0.92)
        * 0.74,
        0.0,
        0.74,
    )
    rgb = np.clip(rgb * (1.0 - sky_darken[..., None]), 0.0, 1.0)

    sky_sample = quiet_sky > 0.52
    if int(np.count_nonzero(sky_sample)) >= 2048:
        sky_medians = np.array([float(np.median(rgb[..., channel][sky_sample])) for channel in range(3)], dtype=np.float32)
        neutral_target = float(np.median(sky_medians))
        sky_gains = np.clip(neutral_target / np.maximum(sky_medians, 1e-5), 0.72, 1.18)
        sky_gains = sky_gains * np.array([0.96, 0.98, 1.02], dtype=np.float32)
        sky_neutralized = np.clip(rgb * sky_gains.reshape(1, 1, 3), 0.0, 1.0)
        neutral_mix = np.clip(quiet_sky[..., None] * (1.0 - nebula[..., None] * 0.94) * 0.88, 0.0, 0.88)
        rgb = np.clip(rgb * (1.0 - neutral_mix) + sky_neutralized * neutral_mix, 0.0, 1.0)
        write_log(
            "Reflection nebula background neutralization: "
            f"sky_rgb={sky_medians[0]:.5f},{sky_medians[1]:.5f},{sky_medians[2]:.5f}; "
            f"gains={sky_gains[0]:.3f},{sky_gains[1]:.3f},{sky_gains[2]:.3f}"
        )

    lum = _rgb_luminance(rgb).astype(np.float32)
    sky_floor = float(np.percentile(lum[sky > 0.45], 18.0)) if int(np.count_nonzero(sky > 0.45)) >= 512 else float(np.percentile(lum, 5.0))
    rgb = np.clip((rgb - sky_floor * 0.64) / max(1e-6, 1.0 - sky_floor * 0.64), 0.0, 1.0)
    lum = _rgb_luminance(rgb).astype(np.float32)
    star_neutral = lum[..., None] + np.clip(rgb - lum[..., None], -0.075, 0.075)
    rgb = np.clip(rgb * (1.0 - star_protect[..., None] * 0.24) + star_neutral * (star_protect[..., None] * 0.24), 0.0, 1.0)

    write_log(
        "Applied reflection nebula color-preserve finish: "
        f"low={low:.5f}; high={high:.5f}; nebula_mean={float(np.mean(nebula)):.5f}; "
        f"green_excess_mean={float(np.mean(green_excess)):.5f}; sky_floor={sky_floor:.5f}"
    )
    return np.clip(rgb * 65535.0, 0.0, 65535.0).round().astype(np.uint16)


def _clean_reflection_nebula_starless_layer(image: np.ndarray, write_log: LogCallback) -> np.ndarray:
    """Suppress IC63-style starless chroma grain while preserving red reflection signal."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        return arr

    rgb = _to_float01(arr)
    lum = _rgb_luminance(rgb).astype(np.float32)
    red_signal = np.clip(
        (rgb[..., 0] - (rgb[..., 1] * 0.68 + rgb[..., 2] * 0.32))
        / max(1e-6, float(np.percentile(rgb[..., 0], 99.2) - np.percentile(rgb[..., 0], 35.0))),
        0.0,
        1.0,
    )
    broad = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 22.0)
    broad_signal = np.clip(
        (broad - np.percentile(broad, 44.0))
        / max(1e-6, np.percentile(broad, 98.5) - np.percentile(broad, 44.0)),
        0.0,
        1.0,
    )
    nebula = np.clip(red_signal * 0.72 + broad_signal * red_signal * 0.65, 0.0, 1.0)
    nebula = cv2.GaussianBlur(nebula.astype(np.float32), (0, 0), 3.4)
    residual_points = np.clip(
        (lum - np.percentile(lum, 96.8))
        / max(1e-6, np.percentile(lum, 99.92) - np.percentile(lum, 96.8)),
        0.0,
        1.0,
    )
    residual_points = cv2.GaussianBlur(residual_points.astype(np.float32), (0, 0), 0.75)
    sky = np.clip((1.0 - nebula * 1.25) * (1.0 - residual_points * 0.88), 0.0, 1.0)

    rgb8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    nlm = cv2.fastNlMeansDenoisingColored(rgb8, None, 18, 34, 7, 31).astype(np.float32) / 255.0
    median = cv2.medianBlur(rgb8, 5).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), 2.4)
    smooth = np.clip(nlm * 0.58 + median * 0.24 + blur * 0.18, 0.0, 1.0)

    smooth_lum = _rgb_luminance(smooth).astype(np.float32)
    smooth_chroma = smooth - smooth_lum[..., None]
    sky_smooth = np.clip(smooth_lum[..., None] + smooth_chroma * 0.22, 0.0, 1.0)
    sky_mix = np.clip(sky[..., None] * 0.92, 0.0, 0.92)
    rgb = np.clip(rgb * (1.0 - sky_mix) + sky_smooth * sky_mix, 0.0, 1.0)

    lum = _rgb_luminance(rgb).astype(np.float32)
    smooth_lum = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), 3.2)
    luma_mix = np.clip(sky * (1.0 - nebula * 0.92) * 0.72, 0.0, 0.72)
    cleaned_lum = np.clip(lum * (1.0 - luma_mix) + smooth_lum * luma_mix, 0.0, 1.0)
    rgb = np.clip(rgb * (cleaned_lum / np.maximum(lum, 1e-5))[..., None], 0.0, 1.0)

    lum = _rgb_luminance(rgb).astype(np.float32)
    chroma = rgb - lum[..., None]
    neutral_sky = np.clip(lum[..., None] * np.array([0.94, 0.95, 0.99], dtype=np.float32).reshape(1, 1, 3), 0.0, 1.0)
    neutral_mix = np.clip(sky[..., None] * (1.0 - nebula[..., None] * 0.86) * 0.82, 0.0, 0.82)
    rgb = np.clip(rgb * (1.0 - neutral_mix) + neutral_sky * neutral_mix, 0.0, 1.0)

    nebula_chroma = np.clip(lum[..., None] + chroma * (1.0 + nebula[..., None] * 0.55), 0.0, 1.0)
    rgb = np.clip(rgb * (1.0 - nebula[..., None] * 0.28) + nebula_chroma * (nebula[..., None] * 0.28), 0.0, 1.0)

    write_log(
        "Cleaned reflection nebula starless layer: "
        f"nebula_mean={float(np.mean(nebula)):.5f}; sky_mean={float(np.mean(sky)):.5f}"
    )
    return np.clip(rgb * 65535.0, 0.0, 65535.0).round().astype(np.uint16)


def _rgb_luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _repair_nebula_starless_base(
    starless_image: np.ndarray,
    stars_image: np.ndarray,
    reference_image: np.ndarray,
    write_log: LogCallback,
) -> tuple[np.ndarray, float]:
    """Build a cleaner nebula base without letting StarNet scars drive the image."""
    starless_rgb = _to_float01(starless_image)
    stars_rgb = _to_float01(stars_image)
    reference_rgb = _to_float01(reference_image)
    if (
        starless_rgb.ndim != 3
        or stars_rgb.ndim != 3
        or reference_rgb.ndim != 3
        or starless_rgb.shape[:2] != stars_rgb.shape[:2]
        or starless_rgb.shape[:2] != reference_rgb.shape[:2]
    ):
        return starless_image, 0.0

    height, width = starless_rgb.shape[:2]
    support = np.ones((height, width), dtype=np.float32)
    inset_y = max(1, int(round(height * 0.025)))
    inset_x = max(1, int(round(width * 0.025)))
    support[:inset_y, :] = 0.0
    support[-inset_y:, :] = 0.0
    support[:, :inset_x] = 0.0
    support[:, -inset_x:] = 0.0
    support = cv2.GaussianBlur(support, (0, 0), max(1.5, min(height, width) * 0.006))

    star_lum = np.max(stars_rgb, axis=2).astype(np.float32)
    star_high = max(float(np.percentile(star_lum[support > 0.90], 99.35)) if np.any(support > 0.90) else float(np.percentile(star_lum, 99.35)), 1e-6)
    star_norm = np.clip(star_lum / star_high, 0.0, 1.0)
    star_density = float(np.mean((star_norm > 0.030) & (support > 0.90)))
    if star_density < 0.020:
        write_log(f"Nebula StarNet base audit: star_density={star_density:.5f}; starless base kept unchanged.")
        return starless_image, star_density

    starless_lum = _rgb_luminance(starless_rgb).astype(np.float32)
    reference_lum = _rgb_luminance(reference_rgb).astype(np.float32)
    removal_core = np.clip((star_norm - 0.0015) / 0.9985, 0.0, 1.0) ** 0.38
    faint_star_support = np.clip((star_norm - 0.0006) / 0.9994, 0.0, 1.0) ** 0.34
    removal_halo = cv2.GaussianBlur(removal_core.astype(np.float32), (0, 0), 2.2)
    removal_wide = cv2.GaussianBlur(faint_star_support.astype(np.float32), (0, 0), 7.0)
    repair_mask = np.clip((removal_halo * 0.58 + removal_wide * 0.42) * support, 0.0, 1.0)
    repair_strength = float(np.clip(0.68 + star_density * 2.1, 0.68, 0.96))
    repair_mask = np.clip(repair_mask * repair_strength, 0.0, 0.995)

    quiet = (support > 0.90) & (repair_mask < 0.04) & (reference_lum > 0)
    if np.count_nonzero(quiet) > 512:
        scale = float(np.median(starless_lum[quiet]) / max(float(np.median(reference_lum[quiet])), 1e-5))
    else:
        scale = float(np.median(starless_lum) / max(float(np.median(reference_lum)), 1e-5))
    scale = float(np.clip(scale, 0.40, 2.20))
    reference_repair = np.clip(reference_rgb * scale, 0.0, 1.0)

    inpaint_seed = np.clip(
        (
            (star_norm > 0.032).astype(np.float32) * 0.70
            + (repair_mask > 0.145).astype(np.float32) * 0.30
        )
        * support,
        0.0,
        1.0,
    )
    inpaint_seed = cv2.dilate(
        inpaint_seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1 + int(star_density > 0.12),
    )
    inpaint_mask = (inpaint_seed > 0).astype(np.uint8) * 255
    if np.count_nonzero(inpaint_mask) > 0:
        reference8 = np.clip(reference_repair * 255.0, 0, 255).astype(np.uint8)
        inpainted8 = cv2.inpaint(reference8, inpaint_mask, 3.0, cv2.INPAINT_TELEA)
        inpainted_reference = inpainted8.astype(np.float32) / 255.0
    else:
        inpainted_reference = reference_repair

    # StarNet's starless layer is useful as a mask, but dense smart-telescope
    # fields leave thousands of tiny removal scars.  In those cases, suppress
    # stars in the calibrated RGB frame and blend that back in as the texture
    # master instead of enhancing the scarred StarNet pixels.
    local_reference = cv2.GaussianBlur(inpainted_reference.astype(np.float32), (0, 0), 2.4 + min(star_density, 0.22) * 4.6)
    wide_reference = cv2.GaussianBlur(inpainted_reference.astype(np.float32), (0, 0), 8.5 + min(star_density, 0.22) * 10.5)
    suppressed_reference = np.clip(
        reference_repair * (1.0 - repair_mask[..., None])
        + (inpainted_reference * 0.62 + local_reference * 0.28 + wide_reference * 0.10) * repair_mask[..., None],
        0.0,
        1.0,
    )

    local_starless = cv2.GaussianBlur(starless_rgb.astype(np.float32), (0, 0), 2.0 + min(star_density, 0.22) * 4.0)
    wide_starless = cv2.GaussianBlur(starless_rgb.astype(np.float32), (0, 0), 7.0 + min(star_density, 0.22) * 9.0)
    starless_repair = np.clip(local_starless * 0.14 + wide_starless * 0.06 + suppressed_reference * 0.80, 0.0, 1.0)
    repaired_starless = np.clip(
        starless_rgb * (1.0 - repair_mask[..., None]) + starless_repair * repair_mask[..., None],
        0.0,
        1.0,
    )

    # Use calibrated RGB as the master only where broad nebula structure exists.
    # In clean sky, the StarNet layer is the better master because it has already
    # removed the dense star field. This keeps real nebula hue/detail without
    # reintroducing thousands of stars into the background.
    ref_smooth = cv2.GaussianBlur(reference_lum.astype(np.float32), (0, 0), 22.0)
    starless_smooth = cv2.GaussianBlur(starless_lum.astype(np.float32), (0, 0), 18.0)
    broad_signal = np.maximum(ref_smooth, starless_smooth)
    signal_low = float(np.percentile(broad_signal[support > 0.70], 52.0)) if np.any(support > 0.70) else float(np.percentile(broad_signal, 52.0))
    signal_high = float(np.percentile(broad_signal[support > 0.70], 99.15)) if np.any(support > 0.70) else float(np.percentile(broad_signal, 99.15))
    nebula_structure = np.clip((broad_signal - signal_low) / max(1e-6, signal_high - signal_low), 0.0, 1.0)
    nebula_structure = cv2.GaussianBlur((nebula_structure * support).astype(np.float32), (0, 0), 8.0)
    weak_dense_nebula = bool(star_density > 0.055 and float(np.mean(nebula_structure)) < 0.14)
    if weak_dense_nebula:
        dense_blend = float(np.clip(0.28 + star_density * 0.80, 0.28, 0.46))
        dense_blend_map = np.clip(
            0.035
            + nebula_structure * dense_blend
            + cv2.GaussianBlur(repair_mask.astype(np.float32), (0, 0), 2.4) * 0.045,
            0.025,
            0.54,
        )
    else:
        dense_blend = float(np.clip(0.70 + star_density * 1.25, 0.70, 0.96))
        dense_blend_map = np.clip(
            0.14
            + nebula_structure * dense_blend
            + cv2.GaussianBlur(repair_mask.astype(np.float32), (0, 0), 2.4) * 0.14,
            0.12,
            0.96,
        )
    repaired = np.clip(
        repaired_starless * (1.0 - dense_blend_map[..., None])
        + suppressed_reference * dense_blend_map[..., None],
        0.0,
        1.0,
    )

    repaired_lum = _rgb_luminance(repaired).astype(np.float32)
    local_wide = cv2.GaussianBlur(repaired_lum, (0, 0), 5.2)
    residual_scale = max(
        1e-6,
        float(np.percentile(np.abs(repaired_lum - local_wide)[support > 0.70], 98.3))
        if np.any(support > 0.70)
        else float(np.percentile(np.abs(repaired_lum - local_wide), 98.3)),
    )
    residual_points = np.clip((repaired_lum - cv2.GaussianBlur(repaired_lum, (0, 0), 1.0)) / residual_scale, 0.0, 1.0)
    residual_points = np.clip(
        cv2.GaussianBlur(residual_points.astype(np.float32), (0, 0), 0.75)
        * (0.34 + star_density * 2.4)
        * support
        * (1.0 - cv2.GaussianBlur(repair_mask.astype(np.float32), (0, 0), 6.0) * 0.35),
        0.0,
        0.70,
    )
    residual_color = cv2.GaussianBlur(repaired.astype(np.float32), (0, 0), 1.9 + min(star_density, 0.22) * 4.5)
    repaired = np.clip(
        repaired * (1.0 - residual_points[..., None])
        + residual_color * residual_points[..., None],
        0.0,
        1.0,
    )

    if star_density > 0.055:
        repaired8 = np.clip(repaired * 255.0, 0, 255).astype(np.uint8)
        denoised = cv2.fastNlMeansDenoisingColored(
            repaired8,
            None,
            int(round(10 + min(star_density, 0.20) * 54)),
            int(round(18 + min(star_density, 0.20) * 64)),
            5,
            17,
        ).astype(np.float32) / 255.0
        sky_like = np.clip(
            (1.0 - cv2.GaussianBlur(repair_mask.astype(np.float32), (0, 0), 8.0) * 0.45)
            * (1.0 - np.clip((reference_lum - np.percentile(reference_lum, 72.0)) / max(1e-6, np.percentile(reference_lum, 99.4) - np.percentile(reference_lum, 72.0)), 0.0, 1.0) * 0.25)
            * support,
            0.0,
            1.0,
        )
        repaired = np.clip(
            repaired * (1.0 - sky_like[..., None] * 0.32)
            + denoised * (sky_like[..., None] * 0.40),
            0.0,
            1.0,
        )

    write_log(
        "Nebula StarNet base audit: "
        f"star_density={star_density:.5f}; "
        f"repair_strength={repair_strength:.2f}; "
        f"dense_calibrated_blend={dense_blend:.2f}; "
        f"weak_dense_nebula={weak_dense_nebula}; "
        f"nebula_structure_mean={float(np.mean(nebula_structure)):.5f}; "
        f"repair_mask_mean={float(np.mean(repair_mask)):.5f}; "
        "using one nebula pipeline with calibrated RGB as dense-field texture reference."
    )
    return np.clip(repaired * 65535.0, 0.0, 65535.0).round().astype(np.uint16), star_density


def _apply_mild_nebula_star_core_reduction(image: np.ndarray, write_log: LogCallback) -> np.ndarray:
    """Mildly compress compact star cores without using StarNet's starless reconstruction."""
    rgb = _to_float01(image)
    if rgb.ndim != 3:
        return image

    lum = _rgb_luminance(rgb).astype(np.float32)
    low = float(np.percentile(lum, 96.5))
    high = float(np.percentile(lum, 99.92))
    if high <= low + 1e-6:
        write_log("Mild nebula star reduction skipped: insufficient stellar contrast.")
        return image

    star_signal = np.clip((lum - low) / max(1e-6, high - low), 0.0, 1.0)
    local_bg_lum = cv2.GaussianBlur(lum, (0, 0), 2.2)
    contrast = np.clip((lum - local_bg_lum) / max(1e-6, high - low), 0.0, 1.0)
    compact = np.clip(star_signal * (contrast ** 0.55), 0.0, 1.0)

    seeds = (compact > 0.11).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)
    if count <= 1:
        write_log("Mild nebula star reduction skipped: no compact star cores found.")
        return image

    areas = stats[:, cv2.CC_STAT_AREA]
    valid = (np.arange(count) > 0) & (areas <= 220)
    valid[0] = False
    core_mask = valid[labels].astype(np.float32)
    core_mask = cv2.GaussianBlur(core_mask * compact, (0, 0), 0.85)
    core_mask = np.clip(core_mask, 0.0, 1.0)

    smooth_rgb = cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), 1.15)
    smooth_lum = _rgb_luminance(smooth_rgb).astype(np.float32)
    smooth_chroma = smooth_rgb - smooth_lum[..., None]
    replacement = np.clip(smooth_lum[..., None] + smooth_chroma * 0.70, 0.0, 1.0)

    mix = np.clip(core_mask[..., None] * 0.38, 0.0, 0.38)
    reduced = np.clip(rgb * (1.0 - mix) + replacement * mix, 0.0, 1.0)
    write_log(
        "Applied mild natural RGB star-core reduction without StarNet recombination: "
        f"mask_mean={float(np.mean(core_mask)):.5f}; "
        f"detected_components={int(np.count_nonzero(valid))}"
    )
    return np.clip(reduced * 65535.0, 0.0, 65535.0).round().astype(np.uint16)


def _apply_duoband_nebula_finish(
    image: np.ndarray,
    write_log: LogCallback,
    palette: str = "warm",
) -> np.ndarray:
    rgb = _to_float01(image)
    lum = _rgb_luminance(rgb).astype(np.float32)

    black_percentile = 6.0 if palette == "lifted" else 10.0
    black = float(np.percentile(lum, black_percentile))
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
        "lifted": (
            np.array([1.32, 0.80, 0.50], dtype=np.float32),
            np.array([0.82, 0.96, 1.08], dtype=np.float32),
            0.035,
        ),
    }
    warm_color, cool_color, sky_floor = palettes.get(palette, palettes["warm"])

    sky_mask = np.clip(1.0 - nebula * 1.35 - star_core * 0.75, 0.0, 1.0)
    sky_mask = cv2.GaussianBlur(sky_mask.astype(np.float32), (0, 0), 2.0)
    sky_tint = np.array([0.50, 0.51, 0.50], dtype=np.float32)
    if palette == "lifted":
        sky_tint = np.array([0.62, 0.60, 0.54], dtype=np.float32)
    neutral_sky = np.clip(detail[..., None] * sky_tint, 0.0, 1.0)
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
    sky_darken = 0.12 if palette == "lifted" else 0.36
    finished = np.clip(finished * (1.0 - sky_mask[..., None] * sky_darken), 0.0, 1.0)

    star_neutral = np.clip(base_lum[..., None] * np.array([1.10, 1.02, 0.94], dtype=np.float32), 0.0, 1.0)
    finished = np.clip(finished * (1.0 - star_core[..., None] * 0.78) + star_neutral * (star_core[..., None] * 0.78), 0.0, 1.0)

    final_lum = _rgb_luminance(finished)
    saturation = 0.78 + nebula[..., None] * (0.92 if palette != "gold" else 1.05) + star_core[..., None] * 0.15
    finished = np.clip(final_lum[..., None] + (finished - final_lum[..., None]) * saturation, 0.0, 1.0)
    sky_desaturate = 0.10 if palette == "lifted" else 0.18
    finished = np.clip(finished * (1.0 - sky_mask[..., None] * sky_desaturate) + final_lum[..., None] * (sky_mask[..., None] * sky_desaturate), 0.0, 1.0)

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


def _is_reflection_style_nebula(analysis: object | None, image: np.ndarray, write_log: Callable[[str], None]) -> bool:
    metrics = getattr(analysis, "metrics", {}) if analysis is not None else {}
    raw_p999 = float(metrics.get("raw_p999", 0.0))
    emission_score = red_emission_dominance(image)
    reflection_score = reflection_nebula_bias(image)
    reflection_style = reflection_score >= 0.08 and emission_score < 3.0
    write_log(
        "Nebula style probe: "
        f"red_emission_dominance={emission_score:.3f}; "
        f"reflection_nebula_bias={reflection_score:.3f}; "
        f"raw_p999={raw_p999:.5f}; "
        f"reflection_style={reflection_style}"
    )
    return reflection_style


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


def _orientation_score(image: np.ndarray, reference: np.ndarray) -> float:
    img = np.asarray(image)
    ref = np.asarray(reference)
    if img.ndim < 2 or ref.ndim < 2 or img.shape[:2] != ref.shape[:2]:
        return -1.0

    def _small_lum(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            data = arr[..., :3].astype(np.float32)
            if np.issubdtype(arr.dtype, np.integer):
                data /= max(float(np.iinfo(arr.dtype).max), 1.0)
            elif data.size and float(np.nanmax(data)) > 1.0:
                data /= 65535.0
            lum = data[..., 0] * 0.2126 + data[..., 1] * 0.7152 + data[..., 2] * 0.0722
        else:
            lum = np.squeeze(arr).astype(np.float32)
            if np.issubdtype(arr.dtype, np.integer):
                lum /= max(float(np.iinfo(arr.dtype).max), 1.0)
            elif lum.size and float(np.nanmax(lum)) > 1.0:
                lum /= 65535.0
        lum = np.nan_to_num(lum.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        height, width = lum.shape[:2]
        scale = max(height / 180.0, width / 180.0, 1.0)
        size = (max(8, int(width / scale)), max(8, int(height / scale)))
        small = cv2.resize(lum, size, interpolation=cv2.INTER_AREA)
        low, high = np.percentile(small, (2.0, 99.5))
        small = np.clip((small - low) / max(1e-6, high - low), 0.0, 1.0)
        small = cv2.GaussianBlur(small.astype(np.float32), (0, 0), 1.0)
        small -= float(np.mean(small))
        return small / max(float(np.std(small)), 1e-6)

    a = _small_lum(img)
    b = _small_lum(ref)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.mean(a * b))


def _orient_like_reference(image: np.ndarray, reference: np.ndarray, write_log: LogCallback, label: str) -> np.ndarray:
    arr = np.asarray(image)
    ref = np.asarray(reference)
    if arr.shape[:2] != ref.shape[:2]:
        write_log(f"{label} orientation check skipped: shape mismatch {arr.shape[:2]} vs {ref.shape[:2]}.")
        return arr

    candidates: list[tuple[str, np.ndarray]] = [
        ("original", arr),
        ("vertical flip", np.flipud(arr)),
        ("horizontal flip", np.fliplr(arr)),
        ("vertical+horizontal flip", np.flipud(np.fliplr(arr))),
    ]
    scores = [(name, candidate, _orientation_score(candidate, ref)) for name, candidate in candidates]
    best_name, best_image, best_score = max(scores, key=lambda item: item[2])
    original_score = scores[0][2]
    write_log(
        f"{label} orientation scores: "
        + ", ".join(f"{name}={score:.5f}" for name, _, score in scores)
    )
    write_log(f"{label} orientation preserved; automatic flip correction is disabled for pipeline safety.")
    return arr


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
    nebula_fallback_stretch_strength: str | None = None,
) -> Path:
    mode = settings.color_calibration_mode
    object_type = _normalized_object_type(settings)
    use_deconvolution_layer = (
        bool(getattr(settings, "siril_deconvolution_enabled", False))
        and mode in {"Basic", "Siril Photometric"}
        and object_type == "galaxy"
    )
    if mode == "Off":
        write_log("Color calibration is off; applying local stretch only.")
        return _run_local_stretch_calibration(working, stretched, calibrated, write_log)

    siril_exe = find_siril_executable(Path(settings.siril_folder))
    if not siril_exe:
        if object_type == "nebula":
            write_log("Siril executable not found; preserving measured RGB for nebula natural finish.")
            return _run_nebula_measured_rgb_fallback(
                working,
                stretched,
                calibrated,
                write_log,
                stretch_strength=nebula_fallback_stretch_strength,
            )
        write_log("Siril executable not found; using Python fallback color calibration.")
        return _run_python_fallback_calibration(working, stretched, calibrated, settings, write_log)

    local_spcc_catalog = find_local_spcc_catalog()
    pcc_command = (
        build_siril_pcc_command(
            original,
            optional_object_name=settings.siril_object_name.strip() or None,
            optional_ra_dec=settings.siril_ra_dec.strip() or None,
            optional_focal_length=settings.siril_focal_length.strip() or None,
            optional_pixel_size=settings.siril_pixel_size.strip() or None,
        )
        if mode == "Siril Photometric"
        else None
    )
    pcc_available = mode == "Siril Photometric" and bool(pcc_command)
    if object_type == "nebula" and not pcc_available:
        if mode == "Siril Photometric":
            write_log("Nebula SPCC/PCC metadata unavailable; using Siril background extraction plus non-photometric measured-RGB fallback.")
            if local_spcc_catalog:
                astap_exe = find_astap_executable()
                write_log(f"Local Gaia SPCC catalog found: {local_spcc_catalog}")
                if astap_exe:
                    astap_db = find_astap_database(astap_exe)
                    write_log(f"ASTAP found for plate-solving fallback: {astap_exe}")
                    if astap_db:
                        write_log(f"ASTAP star database found: {astap_db}")
                    else:
                        write_log("ASTAP star database not found; SPCC cannot plate-solve unsolved FITS files yet.")
                else:
                    write_log("ASTAP executable not found; SPCC still requires solved WCS metadata or ASTAP plate solving.")
            else:
                write_log("Local Gaia SPCC catalog not found; SPCC/PCC cannot use local Gaia calibration.")
        else:
            write_log("Nebula non-photometric color mode; using Siril background extraction before measured-RGB fallback finish.")
        write_log("Nebula non-photometric fallback selected before Siril local color scripts.")
        return _run_nebula_measured_rgb_fallback(
            working,
            stretched,
            calibrated,
            write_log,
            stretch_strength=nebula_fallback_stretch_strength,
        )
    if pcc_available:
        siril_input = job_folder / original.name
        if siril_input.resolve() != original.resolve():
            shutil.copy2(original, siril_input)
        if siril_catalog_calibration_path(pcc_command) == "local_spcc":
            write_log(f"Siril local SPCC Gaia catalog path: {local_spcc_catalog}")
            write_log(f"Siril local SPCC command: {pcc_command}")
        else:
            write_log(f"Siril online PCC metadata command: {pcc_command}")
    elif mode == "Siril Photometric":
        pcc_succeeded = False
        siril_input = job_folder / "siril_input.tif"
        shutil.copy2(working, siril_input)
        write_log("Siril PCC metadata unavailable; using Siril local background/color calibration instead of Python fallback.")
    else:
        siril_input = job_folder / "siril_input.tif"
        shutil.copy2(working, siril_input)
    siril_output_fit = job_folder / "siril_output.fit"

    write_log(f"Siril executable: {siril_exe}")
    write_log(
        "Siril calibration settings: "
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

    catalog_calibration_path = siril_catalog_calibration_path(pcc_command)
    if pcc_available:
        write_log("Siril background extraction: enabled before color calibration.")
        write_log("Siril-based calibration step: enabled.")
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
            pcc_command=pcc_command,
        )
    elif object_type == "nebula":
        script_path = create_nebula_local_color_script(
            siril_input,
            siril_output_fit,
            job_folder,
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

    pcc_succeeded = False
    write_log(f"Siril script: {script_path}")
    try:
        run_siril_script(
            siril_exe,
            script_path,
            job_folder,
            write_log,
            spcc_catalog_dir=local_spcc_catalog if catalog_calibration_path == "local_spcc" else None,
        )
        pcc_succeeded = pcc_available
    except Exception as exc:
        if pcc_available and catalog_calibration_path == "local_spcc":
            write_log(f"Local SPCC Gaia catalog failed; trying online PCC next. Error: {exc}")
            online_pcc_command = "pcc -catalog=apass"
            online_script_path = create_photometric_color_script(
                siril_input,
                siril_output_fit,
                job_folder,
                optional_object_name=settings.siril_object_name.strip() or None,
                optional_ra_dec=settings.siril_ra_dec.strip() or None,
                optional_focal_length=settings.siril_focal_length.strip() or None,
                optional_pixel_size=settings.siril_pixel_size.strip() or None,
                apply_scnr=settings.siril_apply_scnr,
                color_saturation=settings.siril_color_saturation,
                pcc_command=online_pcc_command,
            )
            write_log(f"Siril online PCC fallback script: {online_script_path}")
            write_log(f"Siril online PCC fallback command: {online_pcc_command}")
            try:
                run_siril_script(siril_exe, online_script_path, job_folder, write_log)
            except Exception as online_exc:
                catalog_calibration_path = None
                exc = online_exc
                write_log(f"Online PCC failed after local SPCC failure. Error: {online_exc}")
            else:
                catalog_calibration_path = "online_pcc"
                pcc_succeeded = True
                write_log("Online PCC succeeded.")

        if pcc_succeeded:
            pass
        elif mode == "Siril Photometric":
            if getattr(settings, "pcc_failure_policy", "continue") == "pause":
                write_log(f"Siril PCC failed and requires user decision. Error: {exc}")
                raise PccCalibrationFailed(
                    "Siril PCC failed because the star catalog could not be reached or solved. "
                    "Continue without PCC, or abort this run."
                ) from exc
            if object_type == "nebula":
                write_log(
                    "SPCC/PCC failed, using non-photometric color fallback. "
                    "Bypassing Siril/Python color scripts to preserve measured RGB and original orientation. "
                    f"Error: {exc}"
                )
                return _run_nebula_measured_rgb_fallback(
                    working,
                    stretched,
                    calibrated,
                    write_log,
                    stretch_strength=nebula_fallback_stretch_strength,
                )
            write_log(
                "Siril PCC failed; trying Siril local background/color calibration before Python fallback. "
                f"Error: {exc}"
            )
            pcc_succeeded = False
            if object_type == "nebula":
                fallback_script_path = create_nebula_local_color_script(
                    siril_input,
                    siril_output_fit,
                    job_folder,
                    apply_scnr=settings.siril_apply_scnr,
                    color_saturation=settings.siril_color_saturation,
                )
            else:
                fallback_script_path = create_basic_color_script(
                    siril_input,
                    siril_output_fit,
                    job_folder,
                    apply_scnr=settings.siril_apply_scnr,
                    color_saturation=settings.siril_color_saturation,
                )
            write_log(f"Siril local fallback script: {fallback_script_path}")
            try:
                run_siril_script(siril_exe, fallback_script_path, job_folder, write_log)
            except Exception as fallback_exc:
                write_log(
                    "Siril local fallback failed; using Python fallback color calibration. "
                    f"Error: {fallback_exc}"
                )
                return _run_python_fallback_calibration(working, stretched, calibrated, settings, write_log)
            catalog_calibration_path = "fallback"
            write_log("Siril local fallback color calibration succeeded.")
        elif mode == "Basic":
            if object_type == "nebula":
                write_log(
                    f"Siril {mode} failed; preserving measured RGB for nebula natural finish. "
                    f"Error: {exc}"
                )
                return _run_nebula_measured_rgb_fallback(
                    working,
                    stretched,
                    calibrated,
                    write_log,
                    stretch_strength=nebula_fallback_stretch_strength,
                )
            write_log(f"Siril {mode} failed; using Python fallback color calibration. Error: {exc}")
            return _run_python_fallback_calibration(working, stretched, calibrated, settings, write_log)
        else:
            raise
    if not siril_output_fit.exists():
        raise RuntimeError(f"Siril completed but did not create {siril_output_fit}")
    if catalog_calibration_path == "local_spcc":
        write_log("Local SPCC Gaia catalog succeeded.")
    elif catalog_calibration_path == "online_pcc":
        write_log("Online PCC succeeded.")
    else:
        write_log("Fallback/background calibration used.")
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
    write_log(f"Preserving Siril FITS orientation for {object_type} output.")
    deconvolution_image = None
    if use_deconvolution_layer and deconvolved_output_fit.exists():
        deconvolution_image = load_image(deconvolved_output_fit, write_log)
        write_log("Preserving Siril deconvolution layer orientation for galaxy output.")
    raw_siril = job_folder / "siril_calibrated.tif"
    save_tiff(raw_siril, siril_image, write_log)
    _log_existing_image(raw_siril, write_log, "siril_calibrated.tif")
    catalog_success_marker = job_folder / "siril_catalog_color_succeeded.txt"

    if mode == "Basic" or not pcc_succeeded:
        if mode == "Siril Photometric" and not pcc_succeeded:
            write_log("Siril PCC did not complete; treating calibrated output as local Siril color calibration.")
        write_log("color_calibration_used: natural_color_fallback")
        write_log("color_calibration_success: false")
        if mode == "Siril Photometric" and not pcc_succeeded:
            write_log("fallback_reason: SPCC/PCC command did not produce a catalog-calibrated result.")
        elif mode == "Basic":
            write_log("fallback_reason: Basic Siril/local color mode selected.")
        chroma_95 = chroma_percentile(siril_image, 95.0)
        emission_score = red_emission_dominance(siril_image)
        reflection_score = reflection_nebula_bias(siril_image)
        write_log(
            f"Siril Basic object type: {object_type}; "
            f"chroma p95={chroma_95:.5f}; red_emission_dominance={emission_score:.3f}; "
            f"reflection_nebula_bias={reflection_score:.3f}"
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
        elif object_type == "nebula":
            write_log("Object type is Nebula; preserving Siril-calibrated RGB as the nebula processing master.")
            finished_image = siril_image
        else:
            write_log("Object type is Nebula; using emission nebula color finish.")
            finished_image = apply_goal_look(siril_image, write_log, stretch=False)
    elif object_type == "nebula":
        write_log("Siril catalog color calibration succeeded for nebula; preserving calibrated RGB for DeepSky nebula enhancement.")
        catalog_success_marker.write_text(str(catalog_calibration_path or "catalog"), encoding="utf-8")
        write_log(f"color_calibration_used: {catalog_calibration_path or 'local_spcc'}")
        write_log("color_calibration_success: true")
        finished_image = siril_image
    elif object_type == "galaxy":
        write_log("Siril catalog color calibration succeeded for galaxy; preserving catalog color through linked galaxy stretch.")
        catalog_success_marker.write_text(str(catalog_calibration_path or "catalog"), encoding="utf-8")
        write_log(f"color_calibration_used: {catalog_calibration_path or 'local_spcc'}")
        write_log("color_calibration_success: true")
        finished_image = apply_pcc_galaxy_look(siril_image, write_log)
        if deconvolution_image is not None:
            write_log("Applying Siril deconvolution as protected luminance detail after PCC galaxy stretch.")
            finished_image = blend_galaxy_deconvolution_detail(finished_image, deconvolution_image, write_log)
    else:
        write_log("Siril PCC succeeded; preserving Siril photometric color without manual color shaping.")
        catalog_success_marker.write_text(str(catalog_calibration_path or "catalog"), encoding="utf-8")
        write_log(f"color_calibration_used: {catalog_calibration_path or 'online_pcc'}")
        write_log("color_calibration_success: true")
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
        write_log("Python fallback nebula calibration: preserving calibrated RGB for the unified natural nebula finish.")
        calibrated_image = python_color
    else:
        calibrated_image = apply_broadband_look(python_color, write_log)
    save_tiff(calibrated, calibrated_image, write_log)
    _log_existing_image(calibrated, write_log, "calibrated.tif")
    shutil.copy2(calibrated, stretched)
    _log_existing_image(stretched, write_log, "stretched.tif")
    return calibrated


def _run_nebula_measured_rgb_fallback(
    working: Path,
    stretched: Path,
    calibrated: Path,
    write_log: LogCallback,
    *,
    stretch_strength: str | None = None,
) -> Path:
    source = load_image(working, write_log)
    if stretch_strength:
        write_log(f"Nebula non-photometric fallback: applying protected {stretch_strength} auto stretch.")
        source = astrophotography_stretch(source, strength=stretch_strength)
    write_log(
        "Nebula non-photometric fallback: preserving measured RGB pixels; "
        "no Siril local color script, no Python color rebalance, no synthetic hue injection."
    )
    save_tiff(calibrated, source, write_log)
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
    plate_result = solve_image(original, write_log, allow_plate_solve=True)
    target_identification = identify_target(plate_result, getattr(settings, "siril_object_name", "") or None)
    write_plate_solve_debug(job_folder / "plate_solve_debug.json", plate_result)
    (job_folder / "target_identification.json").write_text(json.dumps(target_identification.to_dict(), indent=2), encoding="utf-8")
    write_log(f"metadata_has_wcs: {str(plate_result.metadata_has_wcs).lower()}")
    write_log(f"solve_source: {plate_result.source}")
    write_log(f"plate_solver_used: {plate_result.solver or 'none'}")
    write_log(f"solved_ra_deg: {plate_result.ra_deg if plate_result.ra_deg is not None else 'none'}")
    write_log(f"solved_dec_deg: {plate_result.dec_deg if plate_result.dec_deg is not None else 'none'}")
    write_log(f"target_identified: {str(target_identification.identified).lower()}")
    write_log(f"target_name: {target_identification.target_name or 'none'}")
    if target_identification.identified:
        write_log(
            "Target-aware guardrails enabled: "
            f"{target_identification.target_name}; expected_color_family={', '.join(target_identification.expected_color_family)}; "
            "guidance only, no pixel recoloring."
        )
    catalog_calibration_input = original
    if plate_result.source == "plate_solve" and original.suffix.lower() in {".fit", ".fits", ".fts"}:
        wcs_enriched = job_folder / f"{original.stem}_wcs{original.suffix}"
        if write_wcs_enriched_fits(original, wcs_enriched, write_log):
            catalog_calibration_input = wcs_enriched
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
    if object_type == "star_cluster" and star_handling_mode != "standard":
        write_log("Star Cluster mode preserves the star field; forcing star settings to standard.")
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
    green_duoband_raw = object_type == "nebula" and _looks_like_green_duoband_raw(
        working_image_for_routing,
        analysis,
        original.name,
    )
    weak_snr_nebula_raw = object_type == "nebula" and _looks_like_weak_snr_nebula_raw(analysis)
    lifted_duoband_gradient_raw = False
    reflection_style_nebula_hint = False
    use_natural_nebula_pipeline = mode == PipelineMode.FULL and object_type == "nebula"
    if object_type == "nebula":
        write_log("Nebula audit: using one calibrated nebula pipeline; no object-specific color branches.")
        if weak_snr_nebula_raw:
            write_log("Weak-SNR nebula frame detected; using conservative stretch and heavier sky chroma cleanup.")

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

    nebula_auto_pcc_command = None
    galaxy_auto_pcc_command = None
    skip_catalog_pcc = getattr(settings, "pcc_failure_policy", "continue") == "continue_without_pcc"
    if mode == PipelineMode.FULL and object_type == "nebula" and settings.color_calibration_mode != "Off" and not skip_catalog_pcc:
        nebula_auto_pcc_command = build_siril_pcc_command(
            catalog_calibration_input,
            optional_object_name=settings.siril_object_name.strip() or None,
            optional_ra_dec=settings.siril_ra_dec.strip() or None,
            optional_focal_length=settings.siril_focal_length.strip() or None,
            optional_pixel_size=settings.siril_pixel_size.strip() or None,
        )
        if nebula_auto_pcc_command:
            write_log(f"Nebula Siril catalog color calibration available and will be used: {nebula_auto_pcc_command}")
        else:
            write_log("Nebula Siril catalog color calibration unavailable; FITS is missing usable WCS/plate-solve metadata.")
    elif mode == PipelineMode.FULL and object_type == "nebula" and settings.color_calibration_mode != "Off":
        write_log("Siril catalog color calibration skipped by user choice; using non-catalog Siril color calibration.")
    if mode == PipelineMode.FULL and object_type == "galaxy" and settings.color_calibration_mode != "Off" and not skip_catalog_pcc:
        galaxy_auto_pcc_command = build_siril_pcc_command(
            catalog_calibration_input,
            optional_object_name=settings.siril_object_name.strip() or None,
            optional_ra_dec=settings.siril_ra_dec.strip() or None,
            optional_focal_length=settings.siril_focal_length.strip() or None,
            optional_pixel_size=settings.siril_pixel_size.strip() or None,
        )
        if galaxy_auto_pcc_command:
            write_log(f"Galaxy Siril catalog color calibration available and will be used automatically: {galaxy_auto_pcc_command}")
        else:
            write_log("Galaxy Siril catalog color calibration unavailable; FITS is missing usable WCS/plate-solve metadata.")
    elif mode == PipelineMode.FULL and object_type == "galaxy" and settings.color_calibration_mode != "Off":
        write_log("Siril catalog color calibration skipped by user choice; using non-catalog Siril color calibration.")

    should_use_siril_calibration = settings.color_calibration_mode != "Off" and (
        mode == PipelineMode.SIRIL
        or (mode == PipelineMode.FULL and use_prestretched)
        or gradient_galaxy_siril
        or (mode == PipelineMode.FULL and siril_deconvolution_requested)
        or bool(nebula_auto_pcc_command)
        or bool(galaxy_auto_pcc_command)
    )

    if should_use_siril_calibration:
        if siril_deconvolution_requested and not gradient_galaxy_siril and not use_prestretched and mode == PipelineMode.FULL:
            write_log("Siril deconvolution requested; routing galaxy run through Siril calibration path.")
        original_color_mode = settings.color_calibration_mode
        if nebula_auto_pcc_command and mode == PipelineMode.FULL and object_type == "nebula":
            settings.color_calibration_mode = "Siril Photometric"
            write_log("Nebula mode: forcing Siril catalog color calibration before DeepSky strong nebula color separation.")
        elif galaxy_auto_pcc_command and mode == PipelineMode.FULL and object_type == "galaxy":
            settings.color_calibration_mode = "Siril Photometric"
            write_log("Galaxy mode: forcing Siril catalog color calibration before the automatic measured-color galaxy finish.")
        write_log("Siril calibration path enabled for this run; applying it to the working TIFF.")
        try:
            _run_siril_calibration(
                catalog_calibration_input,
                working,
                stretched,
                calibrated,
                job_folder,
                settings,
                write_log,
                darkroom_small_galaxy=gradient_galaxy_siril,
                nebula_fallback_stretch_strength=(
                    _adjust_stretch_strength("seestar_weak_nebula", stretch_level)
                    if weak_snr_nebula_raw and not use_prestretched
                    else None
                ),
            )
        finally:
            settings.color_calibration_mode = original_color_mode
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
        green_duoband_raw = object_type == "nebula" and _looks_like_green_duoband_raw(
            working_image,
            analysis,
            original.name,
        )
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
            normalized_name = original.name.lower().replace("_", " ").replace("-", " ")
            lifted_duoband_gradient_raw = (
                (("duo" in normalized_name and "band" in normalized_name) or "dual band" in normalized_name or "dualband" in normalized_name)
                and _working_background_spread(working_image) > 0.55
            )
            stretch_source_image = working_image if lifted_duoband_gradient_raw else load_image(original, write_log)
            stretch_source_image = _flatten_lifted_duoband_gradient(stretch_source_image, write_log)
        elif weak_snr_nebula_raw:
            base_strength = "seestar_weak_nebula"
            baseline_reason = f"weak-SNR nebula using capped linked-luminance micro-stretch ({baseline_reason})"
        elif reflection_style_nebula_hint and base_strength in {"seestar_aggressive", "seestar_extra_aggressive"}:
            base_strength = "seestar"
            baseline_reason = f"weak/reflection nebula color path; limiting sky noise lift ({baseline_reason})"
        stretch_strength = _adjust_stretch_strength(base_strength, stretch_level)
        write_log(f"Auto stretch baseline: {base_strength} ({baseline_reason}).")
        write_log(f"Applying {stretch_strength} stretch after user adjustment: {stretch_level}.")
        stretched_image = astrophotography_stretch(stretch_source_image, strength=stretch_strength)
        save_tiff(stretched, stretched_image, write_log)
        _log_existing_image(stretched, write_log, "stretched.tif")
        if green_duoband_raw:
            palette = str(getattr(settings, "duoband_palette", "warm") or "warm").strip().lower()
            if lifted_duoband_gradient_raw and palette == "warm":
                palette = "lifted"
            write_log(f"Green-dominant duo-band raw finish: applying {palette} color lift.")
            calibrated_image = _apply_duoband_nebula_finish(stretched_image, write_log, palette)
            save_tiff(calibrated, calibrated_image, write_log)
        elif object_type in {"galaxy", "star cluster"}:
            write_log(f"Applying protected raw broadband finish for: {object_type}.")
            calibrated_image = apply_prestretched_broadband_look(stretched_image, write_log)
            save_tiff(calibrated, calibrated_image, write_log)
        elif weak_snr_nebula_raw:
            # The capped micro-stretch is the complete global tone operation for
            # weak nebula frames.  apply_goal_look adds another black point and
            # contrast remap, turning a tiny lift into exaggerated mottling.
            write_log("Weak-SNR nebula: preserving micro-stretched measured RGB without a second global finish.")
            calibrated_image = stretched_image
            save_tiff(calibrated, calibrated_image, write_log)
        elif reflection_style_nebula_hint:
            write_log("Applying protected raw reflection-nebula finish to preserve red-brown RGB color.")
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
    preserve_siril_galaxy_finish = gradient_galaxy_siril
    skip_siril_galaxy_star_reduction = (
        preserve_siril_galaxy_finish
        and object_type == "galaxy"
        and not starless_only_requested
        and _is_compact_siril_galaxy(analysis)
    )
    reflection_style_nebula = False

    skip_generic_deepsnr_for_nebula = use_natural_nebula_pipeline
    if mode in {PipelineMode.FULL, PipelineMode.DEEPSNR} and not preserve_siril_galaxy_finish and not skip_generic_deepsnr_for_nebula:
        deepsnr_exe = find_executable(Path(settings.deepsnr_folder))
        if not deepsnr_exe:
            raise FileNotFoundError("DeepSNR executable not found. Update the DeepSNR path in settings.")
        write_log(f"DeepSNR executable: {deepsnr_exe}")
        run_deepsnr(current, denoised, deepsnr_exe, write_log)
        _log_existing_image(denoised, write_log, "denoised.tif")
        current = denoised
    elif skip_generic_deepsnr_for_nebula:
        write_log("Skipping generic DeepSNR stage for Nebula natural RGB pipeline; using internal masked sky denoise to avoid tile artifacts.")
    elif preserve_siril_galaxy_finish and mode == PipelineMode.FULL:
        write_log("Skipping generic DeepSNR stage; Siril finish already applied.")

    if use_natural_nebula_pipeline and not preserve_siril_galaxy_finish:
        nebula_catalog_color = (job_folder / "siril_catalog_color_succeeded.txt").exists()
        if nebula_catalog_color:
            write_log("Nebula natural RGB pipeline: catalog color available; using stronger measured chroma preservation.")
        elif skip_catalog_pcc:
            write_log("Nebula natural RGB pipeline: Siril/PCC intentionally bypassed; using the single measured-RGB route.")
        else:
            write_log("SPCC/PCC failed, using non-photometric color fallback.")
            write_log("Nebula natural RGB pipeline: SPCC/PCC unavailable; using conservative measured-RGB fallback to avoid painted color.")
        write_log(
            "Nebula natural RGB pipeline: measured RGB only; no synthetic color painting, "
            "no warm_bias/red_bias/cyan_bias, no HOO/showcase color mapping."
        )
        write_log("Nebula hybrid pipeline: June-detail luminance with measured low-frequency RGB color.")
        working_reference = load_image(working, write_log)
        natural_nebula = apply_natural_nebula_rgb_look(
            load_image(current, write_log),
            write_log,
            color_reference=working_reference,
            catalog_color=nebula_catalog_color,
        )
        natural_nebula = _orient_like_reference(natural_nebula, working_reference, write_log, "Natural nebula final")

        # Restore the June processing order: denoise before separation, enhance
        # the starless luminance, then add measured color without replacing that
        # luminance.  This keeps real filament/dust structure out of the color
        # and denoise layers.
        deepsnr_exe = find_executable(Path(settings.deepsnr_folder))
        if not deepsnr_exe:
            raise FileNotFoundError("DeepSNR executable not found. Update the DeepSNR path in settings.")
        write_log(f"Nebula hybrid DeepSNR executable: {deepsnr_exe}")
        run_deepsnr(current, denoised, deepsnr_exe, write_log)
        _log_existing_image(denoised, write_log, "hybrid denoised.tif")

        starnet_exe = find_executable(Path(settings.starnet_folder))
        if not starnet_exe:
            raise FileNotFoundError("StarNet executable not found. Update the StarNet path in settings.")
        write_log("Nebula hybrid pipeline: separating stars before detail and color recomposition.")
        write_log(f"StarNet executable: {starnet_exe}")
        run_starnet(denoised, starless_test, starnet_exe, write_log)
        _log_existing_image(starless_test, write_log, "hybrid starless.tif")
        subtract_images(denoised, starless_test, starless_test_stars)
        _log_existing_image(starless_test_stars, write_log, "hybrid stars.tif")

        if green_duoband_raw:
            write_log("Green duo-band hybrid: skipping broadband June lift to keep the background dark.")
            june_detail = load_image(starless_test, write_log)
            measured_color_reference = load_image(calibrated, write_log)
        else:
            june_detail = apply_starless_nebula_detail(
                load_image(starless_test, write_log),
                write_log,
                natural_hybrid=True,
            )
            measured_color_reference = working_reference
        colored_detail = apply_measured_color_to_nebula_detail(
            june_detail,
            measured_color_reference,
            write_log,
            star_layer=load_image(starless_test_stars, write_log),
        )
        save_tiff(starless_test, colored_detail, write_log)
        _log_existing_image(starless_test, write_log, "colored June-detail starless.tif")
        if starless_only_requested:
            write_log("Starless enabled; keeping colored June-detail starless image.")
            shutil.copy2(starless_test, final)
        else:
            keep_fraction = 0.07 if weak_snr_nebula_raw else 0.08
            write_log(f"Slight Star Reduction enabled; recombining brightest {keep_fraction:.0%} of StarNet star layer.")
            threshold = add_bright_star_fraction(starless_test, starless_test_stars, final, keep_fraction=keep_fraction)
            write_log(f"Star reduction kept bright stars with layer threshold {threshold:.1f}.")
        corrected_final = _orient_like_reference(load_image(final, write_log), working_reference, write_log, "Natural nebula StarNet final")
        save_tiff(final, corrected_final, write_log)
        _log_existing_image(final, write_log, "star-reduced final.tif")

        save_png(final_png, load_image(final, write_log), write_log)
        after_preview = job_folder / "after_preview.png"
        calibrated_preview = job_folder / "calibrated_preview.png"
        make_preview(final, after_preview, log=write_log, stretch_for_display=False)
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

    skip_reflection_nebula_starnet = (
        mode == PipelineMode.FULL
        and object_type == "nebula"
        and reflection_style_nebula
        and not starless_test_requested
    )

    if mode in {PipelineMode.FULL, PipelineMode.STARNET} and object_type != "nebula" and not preserve_siril_galaxy_finish and not skip_reflection_nebula_starnet:
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
        if object_type == "nebula":
            color_separation = str(getattr(settings, "nebula_color_separation", "Balanced") or "Balanced")
            nebula_color_reference = job_folder / "siril_calibrated.tif"
            nebula_texture_reference = nebula_color_reference
            if nebula_color_reference.exists():
                write_log("Using Siril-calibrated image as nebula chroma reference.")
            else:
                nebula_color_reference = calibrated
                nebula_texture_reference = working
                write_log("Using calibrated image for nebula color strength and original working image for hue texture.")
            write_log("Nebula Color: Enhanced")
            write_log(f"DeepSky color separation: {color_separation}")
            starless_image = load_image(starless, write_log)
            stars_image = load_image(stars, write_log)
            color_reference_image = load_image(nebula_color_reference, write_log)
            texture_reference_image = load_image(nebula_texture_reference, write_log)
            star_reference_image = load_image(stretched, write_log)
            starless_image, nebula_star_density = _repair_nebula_starless_base(
                starless_image,
                stars_image,
                color_reference_image,
                write_log,
            )
            if starless_only_requested:
                write_log("Starless enabled; composing controlled starless nebula without star recombination.")
                composed_nebula = compose_pixinsight_nebula_layers(
                    starless_image,
                    np.zeros_like(stars_image),
                    write_log,
                    star_strength=0.0,
                    color_separation=color_separation,
                    color_reference_image=color_reference_image,
                    color_texture_reference_image=texture_reference_image,
                    star_color_reference_image=star_reference_image,
                    weak_snr_nebula=weak_snr_nebula_raw,
                )
            else:
                if starless_test_requested:
                    star_strength = 0.11 if weak_snr_nebula_raw and nebula_star_density > 0.055 else 0.14 if nebula_star_density > 0.055 else 0.22
                elif object_type == "nebula":
                    star_strength = 0.14 if weak_snr_nebula_raw and nebula_star_density > 0.055 else 0.20 if nebula_star_density > 0.055 else 0.34
                else:
                    star_strength = 1.0
                write_log(f"Composing controlled nebula starless/stars layers with star strength {star_strength:.2f}.")
                composed_nebula = compose_pixinsight_nebula_layers(
                    starless_image,
                    stars_image,
                    write_log,
                    star_strength=star_strength,
                    color_separation=color_separation,
                    color_reference_image=color_reference_image,
                    color_texture_reference_image=texture_reference_image,
                    star_color_reference_image=star_reference_image,
                    weak_snr_nebula=weak_snr_nebula_raw,
                )
            save_tiff(final, composed_nebula, write_log)
            _log_existing_image(final, write_log, "final.tif")
            current = final
            if final.exists():
                write_log("Applying nebula edge artifact crop/cleanup.")
                edge_cropped = _crop_edge_artifacts(
                    load_image(final, write_log),
                    fraction=0.025,
                    max_side_fraction=0.22 if lifted_duoband_gradient_raw else 0.12,
                    max_total_fraction=0.34 if lifted_duoband_gradient_raw else 0.18,
                )
                if lifted_duoband_gradient_raw:
                    edge_cropped = _crop_lifted_duoband_artifacts(edge_cropped)
                edge_cropped = _clean_lifted_nebula_color_borders(edge_cropped, write_log)
                save_tiff(final, edge_cropped, write_log)
                _log_existing_image(final, write_log, "edge-cropped final.tif")
            save_png(final_png, load_image(final, write_log), write_log)
            before_preview = job_folder / "before_preview.png"
            after_preview = job_folder / "after_preview.png"
            calibrated_preview = job_folder / "calibrated_preview.png"
            make_preview(working, before_preview, log=write_log, stretch_for_display=True)
            make_preview(final, after_preview, log=write_log, stretch_for_display=False)
            preview_source = calibrated if mode == PipelineMode.SIRIL else final
            make_preview(preview_source, calibrated_preview, log=write_log, stretch_for_display=False)
            write_log(f"Final image: {final}")
            write_log("Done.")
            return {
                "job_folder": job_folder,
                "final": final,
                "png": final_png,
                "before_preview": before_preview,
                "after_preview": after_preview,
                "calibrated_preview": calibrated_preview,
                "log": log_file,
            }
        gentle_nebula_star_reduction = reflection_style_nebula and starless_test_requested
        if starless_test_requested and object_type == "nebula" and not green_duoband_raw:
            gentle_nebula_star_reduction = gentle_nebula_star_reduction or _needs_gentle_nebula_star_reduction(
                analysis,
                load_image(current, write_log),
            )
        if starless_test_requested and object_type == "nebula" and not green_duoband_raw and not gentle_nebula_star_reduction:
            write_log("Enhancing starless nebula dust/detail before star recombination.")
            enhanced_starless = apply_starless_nebula_detail(load_image(starless, write_log), write_log)
            save_tiff(starless, enhanced_starless, write_log)
            _log_existing_image(starless, write_log, "enhanced starless.tif")
        elif gentle_nebula_star_reduction:
            write_log("Reflection-style nebula detected; using gentle star reduction without starless dust enhancer.")
            cleaned_starless = _clean_reflection_nebula_starless_layer(load_image(starless, write_log), write_log)
            save_tiff(starless, cleaned_starless, write_log)
            _log_existing_image(starless, write_log, "cleaned reflection starless.tif")
        elif green_duoband_raw:
            write_log("Skipping starless nebula dust/detail enhancer for green-dominant duo-band raw frame.")
        if starless_test_requested:
            if starless_only_requested:
                write_log("Starless enabled; keeping StarNet starless image without recombining the star layer.")
                shutil.copy2(starless, final)
            else:
                keep_fraction = 0.18 if gentle_nebula_star_reduction else (0.10 if object_type == "nebula" else 0.60)
                write_log(f"Slight Star Reduction enabled; recombining starless image with brightest {keep_fraction:.0%} of stars.")
                threshold = add_bright_star_fraction(starless, stars, final, keep_fraction=keep_fraction)
                write_log(f"Star reduction kept bright stars with layer threshold {threshold:.1f}.")
                if object_type == "nebula" and reflection_style_nebula and not green_duoband_raw:
                    write_log("Applying reflection-style red-brown nebula finish after gentle star recombination.")
                    reflection_finished = _apply_reflection_nebula_color_preserve(load_image(final, write_log), write_log)
                    save_tiff(final, reflection_finished, write_log)
                elif object_type == "nebula" and not gentle_nebula_star_reduction and not green_duoband_raw:
                    write_log("Applying PixInsight-style nebula finish.")
                    pixinsight_nebula = apply_pixinsight_style_nebula_finish(load_image(final, write_log), write_log)
                    save_tiff(final, pixinsight_nebula, write_log)
        else:
            if object_type == "nebula" and not green_duoband_raw and not reflection_style_nebula:
                write_log("Nebula star reduction disabled; recombining the full star layer.")
                add_images(starless, stars, final)
                write_log("Applying PixInsight-style nebula finish.")
                pixinsight_nebula = apply_pixinsight_style_nebula_finish(load_image(final, write_log), write_log)
                save_tiff(final, pixinsight_nebula, write_log)
            elif reflection_style_nebula:
                write_log("Reflection-style nebula finish: preserving the full star field without star reduction.")
                add_images(starless, stars, final)
                reflection_finished = _apply_reflection_nebula_color_preserve(load_image(final, write_log), write_log)
                save_tiff(final, reflection_finished, write_log)
            else:
                add_images(starless, stars, final)
        _log_existing_image(final, write_log, "final.tif")
        current = final
    elif skip_reflection_nebula_starnet:
        write_log("Legacy reflection-nebula branch disabled; preserving the current calibrated RGB image for the unified nebula route.")
        shutil.copy2(current, final)
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
        and object_type != "nebula"
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
        edge_cropped = _crop_edge_artifacts(
            load_image(final, write_log),
            fraction=0.025,
            max_side_fraction=0.22 if lifted_duoband_gradient_raw else 0.12,
            max_total_fraction=0.34 if lifted_duoband_gradient_raw else 0.18,
        )
        if lifted_duoband_gradient_raw:
            edge_cropped = _crop_lifted_duoband_artifacts(edge_cropped)
        edge_cropped = _clean_lifted_nebula_color_borders(edge_cropped, write_log)
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
