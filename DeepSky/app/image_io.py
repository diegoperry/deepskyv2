from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import cv2
import tifffile
from astropy.io import fits
from PIL import Image

SUPPORTED_INPUTS = {".fits", ".fit", ".fts", ".tif", ".tiff"}
LogCallback = Callable[[str], None]


def is_supported_input(path: Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_INPUTS


def _normalize_image_shape(image: np.ndarray) -> tuple[np.ndarray, str]:
    arr = np.asarray(image)
    note = "grayscale" if arr.ndim == 2 else "unknown"

    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
        note = "planar RGB converted to RGB"
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        note = "RGB"
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
        note = "RGBA alpha dropped"
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
        note = "single-page grayscale"

    return arr, note


def describe_array(path: Path, image: np.ndarray, note: str = "") -> str:
    arr = np.asarray(image)
    if arr.size:
        min_value = np.nanmin(arr)
        max_value = np.nanmax(arr)
    else:
        min_value = "empty"
        max_value = "empty"
    suffix = f", channel_order={note}" if note else ""
    return (
        f"{Path(path)} shape={arr.shape}, ndim={arr.ndim}, "
        f"dtype={arr.dtype}, min={min_value}, max={max_value}{suffix}"
    )


def load_fits(path: Path, log: LogCallback | None = None) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        data = None
        hdu_index = -1
        for index, hdu in enumerate(hdul):
            if hdu.data is not None:
                data = hdu.data
                hdu_index = index
                break
    if data is None:
        raise ValueError("No image data found in FITS file.")

    arr = np.asarray(data, dtype=np.float32)
    arr = np.squeeze(arr)
    arr, note = _normalize_image_shape(arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if log:
        log(f"Loaded FITS HDU {hdu_index}: {describe_array(path, arr, note)}")
    return arr


def load_tiff_image(path: Path, log: LogCallback | None = None) -> np.ndarray:
    chosen = None
    chosen_note = ""
    chosen_label = ""

    try:
        with tifffile.TiffFile(path) as tif:
            for index, series in enumerate(tif.series):
                arr = series.asarray()
                arr = np.asarray(arr)
                if arr.ndim == 4 and arr.shape[0] > 0:
                    arr = arr[0]
                    page_note = f"series {index}, first page from stack"
                else:
                    page_note = f"series {index}"

                normalized, shape_note = _normalize_image_shape(arr)
                if normalized.ndim in (2, 3):
                    chosen = normalized
                    chosen_note = f"{page_note}; {shape_note}"
                    chosen_label = f"selected {page_note}"
                    break

            if chosen is None:
                for index, page in enumerate(tif.pages):
                    arr = page.asarray()
                    normalized, shape_note = _normalize_image_shape(arr)
                    if normalized.ndim in (2, 3):
                        chosen = normalized
                        chosen_note = f"page {index}; {shape_note}"
                        chosen_label = f"selected page {index}"
                        break
    except Exception as exc:
        if log:
            log(f"tifffile could not read {path}; trying OpenCV fallback. Error: {exc}")
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if arr is not None:
            if arr.ndim == 3 and arr.shape[-1] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                fallback_note = "OpenCV BGR converted to RGB"
            elif arr.ndim == 3 and arr.shape[-1] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)[..., :3]
                fallback_note = "OpenCV BGRA converted to RGB"
            else:
                fallback_note = "OpenCV grayscale"
            chosen, shape_note = _normalize_image_shape(arr)
            chosen_note = f"{fallback_note}; {shape_note}"
            chosen_label = "OpenCV fallback"

    if chosen is None:
        try:
            pil = Image.open(path)
            arr = np.asarray(pil)
            chosen, shape_note = _normalize_image_shape(arr)
            chosen_note = f"Pillow fallback; {shape_note}"
            chosen_label = "Pillow fallback"
        except Exception as exc:
            raise ValueError(f"No valid image page found in TIFF: {path}") from exc

    if log:
        log(f"Loaded TIFF {chosen_label}: {describe_array(path, chosen, chosen_note)}")
    return chosen


def load_tiff(path: Path) -> np.ndarray:
    return load_tiff_image(path)


def load_image(path: Path, log: LogCallback | None = None) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix in {".fits", ".fit", ".fts"}:
        return load_fits(path, log)
    if suffix in {".tif", ".tiff"}:
        return load_tiff_image(path, log)
    raise ValueError(f"Unsupported input format: {suffix}")


def save_tiff(path: Path, image: np.ndarray, log: LogCallback | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr, note = _normalize_image_shape(np.asarray(image))
    if arr.dtype != np.uint16:
        if np.issubdtype(arr.dtype, np.floating):
            data = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            if data.size and float(np.nanmax(data)) <= 1.0:
                data = data * 65535.0
            arr = np.clip(data, 0.0, 65535.0).round().astype(np.uint16)
        elif np.issubdtype(arr.dtype, np.integer):
            info = np.iinfo(arr.dtype)
            if info.max > 0:
                arr = (arr.astype(np.float32) / float(info.max) * 65535.0).clip(0, 65535).round().astype(np.uint16)
            else:
                arr = np.zeros(arr.shape, dtype=np.uint16)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        tifffile.imwrite(path, arr, photometric="rgb", planarconfig="contig", compression=None)
    else:
        tifffile.imwrite(path, np.squeeze(arr), photometric="minisblack", compression=None)
    if log:
        log(f"Wrote TIFF: {describe_array(path, arr, note)}")


def _to_uint16_working(image: np.ndarray) -> np.ndarray:
    arr, _ = _normalize_image_shape(np.asarray(image))
    if arr.dtype == np.uint16:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        if info.max <= 0:
            return np.zeros(arr.shape, dtype=np.uint16)
        return (arr.astype(np.float32) / float(info.max) * 65535.0).clip(0, 65535).astype(np.uint16)

    data = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    background = np.percentile(data, 0.5)
    data = np.clip(data - background, 0.0, None)
    high = np.percentile(data, 99.9)
    if high <= 0:
        high = float(np.max(data)) if np.max(data) > 0 else 1.0
    return (np.clip(data / high, 0.0, 1.0) * 65535.0).astype(np.uint16)


def convert_to_working_tiff(input_path: Path, output_path: Path, log: LogCallback | None = None) -> np.ndarray:
    image = load_image(input_path, log)
    working = _to_uint16_working(image)
    save_tiff(output_path, working, log)
    return working


def _stretch_rgb_for_preview(image: np.ndarray) -> np.ndarray:
    channels = []
    for index in range(3):
        channel = image[..., index]
        low = np.percentile(channel, 0.5)
        high = np.percentile(channel, 99.5)
        if high <= low:
            high = low + 1.0
        channels.append(np.clip((channel - low) / (high - low), 0.0, 1.0))

    display = np.stack(channels, axis=-1).astype(np.float32)
    luminance = display[..., 0] * 0.2126 + display[..., 1] * 0.7152 + display[..., 2] * 0.0722
    background_mask = luminance < np.percentile(luminance, 45)

    if np.count_nonzero(background_mask) > 128:
        background = np.median(display[background_mask], axis=0)
        neutral = float(np.mean(background))
        gains = neutral / np.maximum(background, 1e-4)
        gains = np.clip(gains, 0.55, 1.85)
        display = np.clip(display * gains.reshape(1, 1, 3), 0.0, 1.0)

    return display


def make_preview(input_path: Path, output_path: Path, max_size: tuple[int, int] = (900, 700), log: LogCallback | None = None) -> None:
    image = load_image(input_path, log)
    arr, note = _normalize_image_shape(np.asarray(image))
    if arr.ndim == 3 and arr.shape[-1] == 3:
        display = arr[..., :3].astype(np.float32)
        display = np.nan_to_num(display, nan=0.0, posinf=0.0, neginf=0.0)
        display = _stretch_rgb_for_preview(display)
    else:
        display = np.squeeze(arr).astype(np.float32)
        display = np.nan_to_num(display, nan=0.0, posinf=0.0, neginf=0.0)
        low = np.percentile(display, 0.5)
        high = np.percentile(display, 99.5)
        if high <= low:
            high = low + 1.0
        display = np.clip((display - low) / (high - low), 0.0, 1.0)

    arr8 = (display * 255).astype(np.uint8)
    pil = Image.fromarray(arr8, mode="RGB" if arr8.ndim == 3 else "L")
    pil.thumbnail(max_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(output_path)
    if log:
        log(f"Wrote preview: {describe_array(output_path, arr8, f'preview from {note}')}")
