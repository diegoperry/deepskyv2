from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable

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


def run_siril_script(
    executable_path: Path,
    script_path: Path,
    working_directory: Path,
    log: LogCallback | None = None,
) -> None:
    command = _format_script_command(executable_path, script_path, working_directory)
    if log:
        log(f"Running Siril: {' '.join(command)}")

    output_log = Path(working_directory) / "siril_output.log"
    with output_log.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(working_directory),
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
                log(message)
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
                for key in ("CRVAL1", "RA", "CRVAL2", "DEC", "FOCALLEN", "FOCLEN", "XPIXSZ", "PIXSIZE1", "YPIXSZ"):
                    if key in hdu.header and key not in values:
                        values[key] = hdu.header[key]
            if hdu.data is not None:
                break
    return values


def _format_header_value(value: object) -> str:
    return str(value).strip().replace(",", ".")


def build_siril_pcc_command(input_path: Path) -> str | None:
    path = Path(input_path)
    if path.suffix.lower() not in SIRIL_PCC_SUFFIXES:
        return None

    try:
        metadata = _read_fits_header_values(path)
    except Exception:
        return None

    ra = metadata.get("CRVAL1") or metadata.get("RA")
    dec = metadata.get("CRVAL2") or metadata.get("DEC")
    focal = metadata.get("FOCALLEN") or metadata.get("FOCLEN")
    pixel_size = metadata.get("XPIXSZ") or metadata.get("PIXSIZE1") or metadata.get("YPIXSZ")

    if None in {ra, dec, focal, pixel_size}:
        return None

    return (
        f"pcc {_format_header_value(ra)},{_format_header_value(dec)} -noflip -platesolve "
        f"-focal={_format_header_value(focal)} -pixelsize={_format_header_value(pixel_size)} "
        "-downscale -catalog=apass"
    )


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
) -> Path:
    script_path = Path(work_dir) / "siril_photometric_color.ssf"
    input_name = _relative_or_name(Path(input_path), Path(work_dir))
    output_stem = Path(output_path).with_suffix("").name
    pcc_command = build_siril_pcc_command(Path(input_path))
    if pcc_command is None:
        pcc_parts = ["pcc"]
        if optional_object_name:
            pcc_parts.append(f'-object="{optional_object_name}"')
        if optional_ra_dec:
            pcc_parts.append(f'-coordinates="{optional_ra_dec}"')
        if optional_focal_length:
            pcc_parts.append(f"-focal={optional_focal_length}")
        if optional_pixel_size:
            pcc_parts.append(f"-pixelsize={optional_pixel_size}")
        pcc_command = " ".join(pcc_parts)

    lines = [
        SIRIL_REQUIRES_COMMAND,
        "# DeepSky Siril Photometric Color Calibration",
        f'load "{input_name}"',
        pcc_command,
    ]
    if apply_scnr:
        lines.append(SCNR_COMMAND)
    amount = _saturation_amount(color_saturation)
    if amount > 0:
        lines.append(SATURATION_COMMAND.format(amount=amount))
    lines.extend([SAVE_FITS_COMMAND.format(output_stem=output_stem), "close"])
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path
