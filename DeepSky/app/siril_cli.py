from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable
import os

from astropy.io import fits


LogCallback = Callable[[str], None]

SIRIL_SCRIPT_COMMAND = ["{exe}", "-s", "{script}", "-d", "{work_dir}"]
SCNR_COMMAND = "rmgreen"
SATURATION_COMMAND = "satu {amount:.2f}"
SAVE_TIFF_COMMAND = 'savetif "{output_stem}"'
SAVE_FITS_COMMAND = 'save "{output_stem}"'
SIRIL_REQUIRES_COMMAND = "requires 1.4.0"
SIRIL_BASIC_LINEAR_COMMANDS = [
    "subsky 2",
    "denoise -vst",
    "denoise -mod=0.5",
    "wavelet 4 1",
    "wrecons 0.1 0.4 1.0 1.0",
    "autostretch -linked -2.8 0.10",
    "ght -D=0.85 -B=0.04 -SP=0.18",
    "linstretch -BP=0.05",
]
SIRIL_PCC_SUFFIXES = {".fit", ".fits", ".fts"}
LOCAL_SPCC_CATALOG_ENV = "SIRIL_SPCC_CATALOG_DIR"
DEFAULT_LOCAL_SPCC_CATALOG_DIR = Path(r"C:\Apps\SirilCatalogs\GaiaDR3_SPCC")


def find_siril_executable(siril_folder: Path) -> Path | None:
    folder = Path(siril_folder)
    if folder.is_file() and folder.suffix.lower() == ".exe":
        return folder
    if not folder.exists():
        return None

    cli_matches = sorted(folder.rglob("siril-cli.exe"))
    if cli_matches:
        return cli_matches[0]

    gui_matches = sorted(folder.rglob("siril.exe"))
    if gui_matches:
        return gui_matches[0]
    return None


def get_siril_help(executable_path: Path) -> str:
    exe = Path(executable_path)
    if not exe.exists():
        return "Executable not found."

    for command in ([str(exe), "--help"], [str(exe), "-h"], [str(exe)]):
        try:
            completed = subprocess.run(
                command,
                cwd=str(exe.parent),
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
        except Exception as exc:
            return f"{' '.join(command)} failed: {exc}"

        output = (completed.stdout or "") + (completed.stderr or "")
        if output.strip():
            return output.strip()
    return "Siril produced no help output."


def _format_script_command(executable_path: Path, script_path: Path, working_directory: Path) -> list[str]:
    values = {
        "exe": str(executable_path),
        "script": str(script_path),
        "work_dir": str(working_directory),
    }
    return [part.format(**values) for part in SIRIL_SCRIPT_COMMAND]


def _escape_siril_config_path(path: Path) -> str:
    return str(path).replace("\\", "\\\\")


def find_local_spcc_catalog(catalog_dir: Path | str | None = None) -> Path | None:
    candidates: list[Path] = []
    if catalog_dir:
        candidates.append(Path(catalog_dir))
    env_value = os.environ.get(LOCAL_SPCC_CATALOG_ENV)
    if env_value:
        candidates.append(Path(env_value))
    candidates.append(DEFAULT_LOCAL_SPCC_CATALOG_DIR)

    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            resolved = candidate.expanduser()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            if resolved.is_dir() and any(resolved.glob("*.dat")):
                return resolved
        except OSError:
            continue
    return None


def _write_isolated_siril_config(siril_local: Path, spcc_catalog_dir: Path | None) -> None:
    if not spcc_catalog_dir:
        return

    config_dir = siril_local / "siril"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.1.4.ini"
    try:
        resolved_catalog_dir = spcc_catalog_dir.resolve()
    except Exception:
        resolved_catalog_dir = spcc_catalog_dir
    escaped_catalog_dir = _escape_siril_config_path(resolved_catalog_dir)
    config_path.write_text(
        "\n".join(
            [
                "[core]",
                f"catalogue_gaia_photo={escaped_catalog_dir}",
                "",
                "[gui]",
                "use_spcc_repository=false",
                "auto_update_spcc=false",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_siril_script(
    executable_path: Path,
    script_path: Path,
    working_directory: Path,
    log: LogCallback | None = None,
    spcc_catalog_dir: Path | None = None,
) -> None:
    executable_path = Path(executable_path).resolve()
    script_path = Path(script_path).resolve()
    working_directory = Path(working_directory).resolve()
    command = _format_script_command(executable_path, script_path, working_directory)
    if log:
        log(f"Running Siril: {' '.join(command)}")

    siril_profile = Path(working_directory) / ".siril_profile"
    siril_local = siril_profile / "Local"
    siril_roaming = siril_profile / "Roaming"
    siril_cache = siril_profile / "cache"
    for folder in (siril_local, siril_roaming, siril_cache):
        folder.mkdir(parents=True, exist_ok=True)
    _write_isolated_siril_config(siril_local, spcc_catalog_dir)
    _write_isolated_siril_config(siril_roaming, spcc_catalog_dir)

    env = os.environ.copy()
    env["LOCALAPPDATA"] = str(siril_local)
    env["APPDATA"] = str(siril_roaming)
    env["XDG_CACHE_HOME"] = str(siril_cache)
    env["XDG_CONFIG_HOME"] = str(siril_roaming)
    if spcc_catalog_dir:
        env[LOCAL_SPCC_CATALOG_ENV] = str(spcc_catalog_dir)

    output_log = Path(working_directory) / "siril_output.log"
    with output_log.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(working_directory),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            message = line.rstrip()
            log_file.write(message + "\n")
            if log:
                try:
                    log(message)
                except UnicodeEncodeError:
                    log(message.encode("ascii", errors="replace").decode("ascii"))
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"Siril failed with exit code {return_code}. See {output_log}")


def _relative_or_name(path: Path, work_dir: Path) -> str:
    try:
        return path.relative_to(work_dir).as_posix()
    except ValueError:
        return path.name


def _saturation_amount(color_saturation: int) -> float:
    return max(0.0, min(1.0, float(color_saturation) / 100.0))


def _read_fits_header_values(input_path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    with fits.open(input_path, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.header:
                for key in (
                    "OBJECT",
                    "OBJNAME",
                    "CRVAL1",
                    "RA",
                    "OBJCTRA",
                    "CRVAL2",
                    "CTYPE1",
                    "CTYPE2",
                    "DEC",
                    "OBJCTDEC",
                    "FOCALLEN",
                    "FOCLEN",
                    "FOCUS",
                    "XPIXSZ",
                    "PIXSIZE1",
                    "YPIXSZ",
                    "PIXSIZE2",
                ):
                    if key in hdu.header and key not in values:
                        values[key] = hdu.header[key]
            if hdu.data is not None:
                break
    return values


def _format_header_value(value: object) -> str:
    return str(value).strip().replace(",", ".")


def _looks_like_empty_header_value(value: object | None) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return not text or text.lower() in {"unknown", "none", "null", "nan", "-"}


def build_siril_pcc_command(
    input_path: Path,
    *,
    optional_object_name: str | None = None,
    optional_ra_dec: str | None = None,
    optional_focal_length: str | None = None,
    optional_pixel_size: str | None = None,
    prefer_local_spcc: bool = True,
) -> str | None:
    path = Path(input_path)
    if path.suffix.lower() not in SIRIL_PCC_SUFFIXES:
        return None

    try:
        metadata = _read_fits_header_values(path)
    except Exception:
        return None

    has_wcs = (
        not _looks_like_empty_header_value(metadata.get("CRVAL1"))
        and not _looks_like_empty_header_value(metadata.get("CRVAL2"))
        and not _looks_like_empty_header_value(metadata.get("CTYPE1"))
        and not _looks_like_empty_header_value(metadata.get("CTYPE2"))
    )
    if not has_wcs:
        return None

    if prefer_local_spcc and find_local_spcc_catalog() is not None:
        return "spcc -catalog=localgaia"

    return "pcc -catalog=apass"


def siril_catalog_calibration_path(pcc_command: str | None) -> str | None:
    if not pcc_command:
        return None
    if pcc_command.strip().lower().startswith("spcc"):
        return "local_spcc"
    return "online_pcc"


def create_basic_color_script(
    input_path: Path,
    output_path: Path,
    work_dir: Path,
    apply_scnr: bool = False,
    color_saturation: int = 35,
    enable_deconvolution: bool = False,
    deconvolution_iterations: int = 14,
    deconvolution_alpha: int = 1800,
) -> Path:
    script_path = Path(work_dir) / "siril_basic_color.ssf"
    input_name = _relative_or_name(Path(input_path), Path(work_dir))
    output_stem = Path(output_path).with_suffix("").name
    safe_iterations = max(1, min(30, int(deconvolution_iterations)))
    safe_alpha = max(500, min(10000, int(deconvolution_alpha)))
    lines = [
        SIRIL_REQUIRES_COMMAND,
        "# DeepSky Siril linear processing path",
        f'load "{input_name}"',
    ]
    if enable_deconvolution:
        lines.extend(
            [
                "subsky 2",
                "denoise -vst",
                "denoise -mod=0.5",
                "makepsf stars -sym -ks=27 -savepsf=deepsky_deconvolution_psf.fit",
                f"rl -loadpsf=deepsky_deconvolution_psf.fit -iters={safe_iterations} -fh -alpha={safe_alpha}",
                "wavelet 4 1",
                "wrecons 0.1 0.4 1.0 1.0",
                "autostretch -linked -2.8 0.10",
                "ght -D=0.85 -B=0.04 -SP=0.18",
                "linstretch -BP=0.05",
            ]
        )
    else:
        lines.extend(SIRIL_BASIC_LINEAR_COMMANDS)
    if apply_scnr:
        lines.append(SCNR_COMMAND)
    lines.extend([SAVE_FITS_COMMAND.format(output_stem=output_stem), "close"])
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path


def create_deconvolution_script(
    input_path: Path,
    output_path: Path,
    work_dir: Path,
    iterations: int = 8,
    alpha: int = 3000,
) -> Path:
    script_path = Path(work_dir) / "siril_deconvolution.ssf"
    input_name = _relative_or_name(Path(input_path), Path(work_dir))
    output_stem = Path(output_path).with_suffix("").name
    psf_name = "deepsky_deconvolution_psf.fit"
    safe_iterations = max(1, min(20, int(iterations)))
    safe_alpha = max(500, min(10000, int(alpha)))
    lines = [
        SIRIL_REQUIRES_COMMAND,
        "# Optional DeepSky Richardson-Lucy deconvolution test",
        f'load "{input_name}"',
        f"makepsf stars -sym -ks=17 -savepsf={psf_name}",
        f"rl -loadpsf={psf_name} -iters={safe_iterations} -fh -alpha={safe_alpha}",
        SAVE_FITS_COMMAND.format(output_stem=output_stem),
        "close",
    ]
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path


def create_nebula_local_color_script(
    input_path: Path,
    output_path: Path,
    work_dir: Path,
    apply_scnr: bool = False,
    color_saturation: int = 35,
) -> Path:
    """Gentle Siril fallback for nebula frames when PCC is unavailable.

    Keep this path linear. DeepSky does the nebula stretch, DeepSNR cleanup,
    and natural color finish later; pre-stretching here makes faint sky noise
    and stacking edges look like real nebulosity.
    """
    script_path = Path(work_dir) / "siril_nebula_local_color.ssf"
    input_name = _relative_or_name(Path(input_path), Path(work_dir))
    output_stem = Path(output_path).with_suffix("").name
    lines = [
        SIRIL_REQUIRES_COMMAND,
        "# DeepSky controlled nebula local calibration path",
        f'load "{input_name}"',
        "subsky 2",
    ]
    if apply_scnr:
        lines.append(SCNR_COMMAND)
    lines.extend([SAVE_FITS_COMMAND.format(output_stem=output_stem), "close"])
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path


def create_background_extraction_script(
    input_path: Path,
    output_path: Path,
    work_dir: Path,
) -> Path:
    """Create a Siril background-only script with no color calibration.

    This is intentionally independent from PCC/SPCC so linear narrowband and
    duo-band frames can use Siril's spatial background model without applying a
    broadband catalog color solution.
    """
    script_path = Path(work_dir) / "siril_background_only.ssf"
    input_name = _relative_or_name(Path(input_path), Path(work_dir))
    output_stem = Path(output_path).with_suffix("").name
    lines = [
        SIRIL_REQUIRES_COMMAND,
        "# DeepSky canonical nebula background extraction only",
        f'load "{input_name}"',
        "subsky 2",
        SAVE_FITS_COMMAND.format(output_stem=output_stem),
        "close",
    ]
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path


def create_stacked_rgb_narrowband_script(
    input_path: Path,
    output_path: Path,
    work_dir: Path,
) -> Path:
    """Build an HOO-style Siril composition from an already-stacked RGB image.

    Siril's extract_HaOIII command is for undebayered CFA data. DeepSky accepts
    already-stacked RGB FITS/TIFF files, so this route follows Siril's documented
    split + PixelMath + rgbcomp workflow instead.
    """
    script_path = Path(work_dir) / "siril_stacked_rgb_narrowband.ssf"
    input_name = _relative_or_name(Path(input_path), Path(work_dir))
    output_stem = Path(output_path).with_suffix("").name
    lines = [
        SIRIL_REQUIRES_COMMAND,
        "# DeepSky stacked-RGB HOO-style narrowband composition",
        "set32bits",
        f'load "{input_name}"',
        "split deepsky_nb_ha deepsky_nb_green deepsky_nb_blue",
        'pm "$deepsky_nb_green$ * 0.65 + $deepsky_nb_blue$ * 0.35" -nosum',
        "save deepsky_nb_oiii",
        'pm "$deepsky_nb_ha$ * 0.18 + $deepsky_nb_oiii$ * 0.82" -nosum',
        "save deepsky_nb_hoo_green",
        f"rgbcomp deepsky_nb_ha deepsky_nb_hoo_green deepsky_nb_oiii -out={output_stem} -nosum",
        "close",
    ]
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path

def create_photometric_color_script(
    input_path: Path,
    output_path: Path,
    work_dir: Path,
    optional_object_name: str | None = None,
    optional_ra_dec: str | None = None,
    optional_focal_length: str | None = None,
    optional_pixel_size: str | None = None,
    apply_scnr: bool = False,
    color_saturation: int = 35,
    pcc_command: str | None = None,
) -> Path:
    script_path = Path(work_dir) / "siril_photometric_color.ssf"
    input_name = _relative_or_name(Path(input_path), Path(work_dir))
    output_stem = Path(output_path).with_suffix("").name
    pcc_command = pcc_command or build_siril_pcc_command(
        Path(input_path),
        optional_object_name=optional_object_name,
        optional_ra_dec=optional_ra_dec,
        optional_focal_length=optional_focal_length,
        optional_pixel_size=optional_pixel_size,
    )
    if pcc_command is None:
        pcc_command = "pcc -catalog=apass"

    lines = [
        SIRIL_REQUIRES_COMMAND,
        "# DeepSky Siril catalog color calibration",
        f'load "{input_name}"',
    ]
    if Path(input_path).stem != "siril_background_only":
        lines.extend(["# Siril background extraction before color calibration", "subsky 2"])
    else:
        lines.append("# Background extraction already completed by canonical nebula stage 2")
    lines.append(pcc_command)
    if apply_scnr:
        lines.append(SCNR_COMMAND)
    amount = _saturation_amount(color_saturation)
    if amount > 0:
        lines.append(SATURATION_COMMAND.format(amount=amount))
    lines.extend([SAVE_FITS_COMMAND.format(output_stem=output_stem), "close"])
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path
