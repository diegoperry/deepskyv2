from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u


LogCallback = Callable[[str], None]
FITS_SUFFIXES = {".fit", ".fits", ".fts"}
ASTAP_ENV = "ASTAP_EXE"


@dataclass
class PlateSolveResult:
    solved: bool
    source: str
    ra_deg: float | None
    dec_deg: float | None
    pixel_scale_arcsec: float | None
    rotation_deg: float | None
    fov_width_deg: float | None
    fov_height_deg: float | None
    confidence: float | None
    solver: str | None
    error: str | None
    metadata_has_wcs: bool = False
    object_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _empty(value: object | None) -> bool:
    text = "" if value is None else str(value).strip()
    return not text or text.lower() in {"unknown", "none", "null", "nan", "-"}


def _parse_angle(value: object, *, is_ra: bool, numeric_ra_is_degrees: bool = False) -> float | None:
    if _empty(value):
        return None
    text = str(value).strip()
    try:
        number = float(text)
        if is_ra and not numeric_ra_is_degrees and 0.0 <= number <= 24.0:
            return number * 15.0
        return number
    except ValueError:
        pass
    try:
        unit = u.hourangle if is_ra else u.deg
        coord = SkyCoord(text, "0d" if is_ra else "0h", unit=(unit, u.deg)) if is_ra else SkyCoord("0h", text, unit=(u.hourangle, u.deg))
        return float(coord.ra.deg if is_ra else coord.dec.deg)
    except Exception:
        return None


def _read_primary_header(path: Path) -> tuple[fits.Header | None, int | None, int | None]:
    if path.suffix.lower() not in FITS_SUFFIXES:
        return None, None, None
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            shape = getattr(hdu.data, "shape", None)
            if not shape or len(shape) < 2:
                continue
            return hdu.header, int(shape[-1]), int(shape[-2])
    return None, None, None


def _header_result(path: Path) -> PlateSolveResult:
    try:
        header, width, height = _read_primary_header(path)
    except Exception as exc:
        return PlateSolveResult(False, "unknown", None, None, None, None, None, None, None, None, str(exc))
    if header is None:
        return PlateSolveResult(False, "unknown", None, None, None, None, None, None, None, None, "No FITS header found")

    object_name = str(header.get("OBJECT") or header.get("OBJNAME") or "").strip() or None
    ra = _parse_angle(header.get("CRVAL1"), is_ra=True, numeric_ra_is_degrees=True)
    if ra is None:
        ra = _parse_angle(header.get("RA") or header.get("OBJCTRA"), is_ra=True, numeric_ra_is_degrees=True)
    dec = _parse_angle(header.get("CRVAL2") or header.get("DEC") or header.get("OBJCTDEC"), is_ra=False)
    has_wcs_keys = not any(_empty(header.get(key)) for key in ("CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2"))

    pixel_scale = None
    rotation = None
    if has_wcs_keys:
        try:
            wcs = WCS(header)
            scales = wcs.proj_plane_pixel_scales() * 3600.0
            pixel_scale = float(sum(abs(float(scale)) for scale in scales[:2]) / 2.0)
            matrix = getattr(wcs.wcs, "pc", None)
            if matrix is not None:
                rotation = float(__import__("math").degrees(__import__("math").atan2(matrix[1][0], matrix[0][0])))
        except Exception:
            pass

    focal_length = header.get("FOCALLEN") or header.get("FOCLEN") or header.get("FOCUS")
    pixel_size = header.get("XPIXSZ") or header.get("PIXSIZE1") or header.get("YPIXSZ") or header.get("PIXSIZE2")
    if pixel_scale is None:
        try:
            if not _empty(focal_length) and not _empty(pixel_size):
                pixel_scale = 206.265 * float(pixel_size) / float(focal_length)
        except Exception:
            pixel_scale = None

    fov_width = pixel_scale * width / 3600.0 if pixel_scale and width else None
    fov_height = pixel_scale * height / 3600.0 if pixel_scale and height else None
    if has_wcs_keys and ra is not None and dec is not None:
        return PlateSolveResult(True, "metadata", ra, dec, pixel_scale, rotation, fov_width, fov_height, 1.0, "metadata", None, True, object_name)
    return PlateSolveResult(False, "unknown", ra, dec, pixel_scale, rotation, fov_width, fov_height, None, None, "Missing reliable WCS/RA/Dec metadata", has_wcs_keys, object_name)


def find_astap_executable() -> Path | None:
    env = os.environ.get(ASTAP_ENV)
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.extend(
        [
            Path(r"C:\Program Files\astap\astap.exe"),
            Path(r"C:\Program Files (x86)\astap\astap.exe"),
            Path(r"C:\Apps\ASTAP\astap.exe"),
        ]
    )
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _read_astap_wcs(input_path: Path) -> Path | None:
    for candidate in (
        input_path.with_suffix(".wcs"),
        input_path.with_name(input_path.stem + ".wcs"),
        input_path.with_name(input_path.name + ".wcs"),
    ):
        if candidate.exists():
            return candidate
    return None


def _parse_astap_output(text: str) -> tuple[float | None, float | None]:
    ra = dec = None
    ra_match = re.search(r"\bRA\b[^0-9+\-.]*([0-9.+:-]+)", text, re.IGNORECASE)
    dec_match = re.search(r"\bDEC\b[^0-9+\-.]*([+\-]?[0-9.+:-]+)", text, re.IGNORECASE)
    if ra_match:
        ra = _parse_angle(ra_match.group(1), is_ra=True)
    if dec_match:
        dec = _parse_angle(dec_match.group(1), is_ra=False)
    return ra, dec


def run_astap_plate_solve(input_path: Path, log: LogCallback | None = None) -> PlateSolveResult:
    exe = find_astap_executable()
    if exe is None:
        return PlateSolveResult(False, "unknown", None, None, None, None, None, None, None, "astap", "ASTAP executable not found")

    command = [str(exe), "-f", str(input_path), "-r", "180", "-z", "0"]
    if log:
        log(f"Plate solving with ASTAP: {' '.join(command)}")
    try:
        completed = subprocess.run(command, cwd=str(input_path.parent), capture_output=True, text=True, timeout=180, check=False)
    except Exception as exc:
        return PlateSolveResult(False, "unknown", None, None, None, None, None, None, None, "astap", str(exc))

    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    wcs_path = _read_astap_wcs(input_path)
    if wcs_path:
        try:
            with fits.open(wcs_path, memmap=False) as hdul:
                header = hdul[0].header
            temp = input_path.with_suffix(".fits")
            result = _header_result(temp) if temp.exists() else PlateSolveResult(False, "unknown", None, None, None, None, None, None, None, "astap", None)
            ra = _parse_angle(header.get("CRVAL1"), is_ra=True)
            dec = _parse_angle(header.get("CRVAL2"), is_ra=False)
            if ra is not None and dec is not None:
                result.solved = True
                result.source = "plate_solve"
                result.ra_deg = ra
                result.dec_deg = dec
                result.confidence = 0.82
                result.solver = "astap"
                result.error = None
                return result
        except Exception as exc:
            return PlateSolveResult(False, "unknown", None, None, None, None, None, None, None, "astap", f"ASTAP WCS parse failed: {exc}")

    ra, dec = _parse_astap_output(output)
    if completed.returncode == 0 and ra is not None and dec is not None:
        return PlateSolveResult(True, "plate_solve", ra, dec, None, None, None, None, 0.65, "astap", None)
    return PlateSolveResult(False, "unknown", None, None, None, None, None, None, None, "astap", f"ASTAP failed or returned no WCS: {output.strip()[:500]}")


def solve_image(input_path: Path, log: LogCallback | None = None, *, allow_plate_solve: bool = True) -> PlateSolveResult:
    metadata = _header_result(Path(input_path))
    if metadata.solved:
        return metadata
    if allow_plate_solve:
        solved = run_astap_plate_solve(Path(input_path), log)
        if solved.solved:
            solved.object_name = metadata.object_name
            solved.metadata_has_wcs = metadata.metadata_has_wcs
            return solved
        if log:
            log(f"Plate solve failed: {solved.error}")
    metadata.source = "unknown"
    metadata.solver = None
    return metadata


def write_plate_solve_debug(path: Path, result: PlateSolveResult) -> None:
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def write_wcs_enriched_fits(input_path: Path, output_path: Path, log: LogCallback | None = None) -> bool:
    """Write a FITS copy with ASTAP WCS headers when a sidecar solution exists."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.suffix.lower() not in FITS_SUFFIXES:
        return False
    wcs_path = _read_astap_wcs(input_path)
    if wcs_path is None:
        return False
    try:
        with fits.open(wcs_path, memmap=False) as solved_hdul:
            solved_header = solved_hdul[0].header
        with fits.open(input_path, memmap=False) as hdul:
            for hdu in hdul:
                if hdu.data is None:
                    continue
                header = hdu.header.copy()
                for key, value in solved_header.items():
                    if key in {
                        "WCSAXES",
                        "CTYPE1",
                        "CTYPE2",
                        "CRVAL1",
                        "CRVAL2",
                        "CRPIX1",
                        "CRPIX2",
                        "CDELT1",
                        "CDELT2",
                        "CUNIT1",
                        "CUNIT2",
                        "CROTA1",
                        "CROTA2",
                        "CD1_1",
                        "CD1_2",
                        "CD2_1",
                        "CD2_2",
                        "PC1_1",
                        "PC1_2",
                        "PC2_1",
                        "PC2_2",
                        "EQUINOX",
                        "RADESYS",
                    }:
                        header[key] = value
                fits.PrimaryHDU(data=hdu.data, header=header).writeto(output_path, overwrite=True)
                if log:
                    log(f"Created WCS-enriched FITS for Siril catalog calibration: {output_path.name}")
                return True
    except Exception as exc:
        if log:
            log(f"Could not create WCS-enriched FITS: {exc}")
    return False
