from __future__ import annotations

import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np

from .cli_tools import find_executable, run_deepsnr, run_starnet
from .goal_look import (
    apply_broadband_look,
    apply_goal_look,
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
    save_tiff,
)
from .input_analysis import analyze_input_stretch
from .image_math import add_images, subtract_images
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


def _run_siril_calibration(
    original: Path,
    working: Path,
    stretched: Path,
    calibrated: Path,
    job_folder: Path,
    settings: AppSettings,
    write_log: LogCallback,
) -> Path:
    mode = settings.color_calibration_mode
    if mode == "Off":
        write_log("Color calibration is off; applying local stretch only.")
        stretched_image = astrophotography_stretch(load_image(working, write_log))
        save_tiff(stretched, stretched_image, write_log)
        _log_existing_image(stretched, write_log, "stretched.tif")
        shutil.copy2(stretched, calibrated)
        _log_existing_image(calibrated, write_log, "calibrated.tif")
        return calibrated

    siril_exe = find_siril_executable(Path(settings.siril_folder))
    if not siril_exe:
        write_log("Siril executable not found; using Python fallback color calibration.")
        return _run_python_fallback_calibration(working, stretched, calibrated, write_log)

    pcc_command = build_siril_pcc_command(original) if mode == "Siril Photometric" else None
    if mode == "Siril Photometric" and pcc_command:
        siril_input = job_folder / original.name
        if siril_input.resolve() != original.resolve():
            shutil.copy2(original, siril_input)
        write_log(f"Siril PCC metadata command: {pcc_command}")
    elif mode == "Siril Photometric":
        write_log("Siril PCC metadata unavailable; using Python fallback color calibration.")
        return _run_python_fallback_calibration(working, stretched, calibrated, write_log)
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
            return _run_python_fallback_calibration(working, stretched, calibrated, write_log)
        raise
    if not siril_output_fit.exists():
        raise RuntimeError(f"Siril completed but did not create {siril_output_fit}")
    write_log("Siril color calibration succeeded.")
    _log_existing_image(siril_output_fit, write_log, "siril_output.fit")

    siril_image = load_image(siril_output_fit, write_log)
    siril_image = np.flipud(siril_image)
    write_log("Corrected Siril FITS orientation with vertical flip.")
    raw_siril = job_folder / "siril_calibrated.tif"
    save_tiff(raw_siril, siril_image, write_log)
    _log_existing_image(raw_siril, write_log, "siril_calibrated.tif")

    if mode == "Basic":
        chroma_95 = chroma_percentile(siril_image, 95.0)
        emission_score = red_emission_dominance(siril_image)
        write_log(f"Siril Basic chroma check: p95={chroma_95:.5f}; red_emission_dominance={emission_score:.3f}")
        if chroma_95 < 0.18 and emission_score >= 3.0:
            write_log("Siril Basic output is low-chroma with strong red emission; using Python star-photometry color as nebula source.")
            source = load_image(working, write_log)
            python_color = python_fallback_color_calibration(source, write_log)
            python_reference = job_folder / "python_color_reference.tif"
            save_tiff(python_reference, python_color, write_log)
            _log_existing_image(python_reference, write_log, "python_color_reference.tif")
            finished_image = apply_goal_look(python_color, write_log)
        elif emission_score < 3.0:
            write_log("Broadband/galaxy-like color detected; using neutral broadband finish.")
            finished_image = apply_broadband_look(siril_image, write_log)
            deepsnr_exe = find_executable(Path(settings.deepsnr_folder))
            if deepsnr_exe:
                pre_denoise = job_folder / "broadband_pre_deepsnr.tif"
                deepsnr_background = job_folder / "broadband_deepsnr.tif"
                save_tiff(pre_denoise, finished_image, write_log)
                write_log(f"DeepSNR background cleanup executable: {deepsnr_exe}")
                try:
                    run_deepsnr(pre_denoise, deepsnr_background, deepsnr_exe, write_log)
                    _log_existing_image(deepsnr_background, write_log, "broadband_deepsnr.tif")
                    finished_image = blend_broadband_background_denoise(
                        finished_image,
                        load_image(deepsnr_background, write_log),
                        settings.galaxy_background_smoothness,
                        settings.galaxy_background_darkness,
                        settings.galaxy_chroma_noise_reduction,
                        settings.galaxy_protect_detail,
                        write_log,
                    )
                except Exception as exc:
                    write_log(f"DeepSNR background cleanup failed; keeping broadband finish. Error: {exc}")
            else:
                write_log("DeepSNR background cleanup skipped; executable not found.")
        else:
            finished_image = apply_goal_look(siril_image, write_log)
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
    write_log: LogCallback,
) -> Path:
    source = load_image(working, write_log)
    python_color = python_fallback_color_calibration(source, write_log)
    emission_score = red_emission_dominance(python_color)
    write_log(f"Python fallback scene classification: red_emission_dominance={emission_score:.3f}")
    if emission_score >= 3.0:
        calibrated_image = apply_goal_look(python_color, write_log)
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
    try:
        analysis = analyze_input_stretch(original)
        metrics = ", ".join(f"{key}={value:.4f}" for key, value in analysis.metrics.items())
        if analysis.likely_stretched:
            write_log(f"WARNING: Pre-stretched input suspected ({analysis.confidence} confidence). {analysis.message}")
            write_log(f"Input stretch analysis: {metrics}")
        else:
            write_log(f"Input stretch analysis: {analysis.message} ({metrics})")
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
    final = job_folder / "final.tif"

    write_log("Creating 16-bit working TIFF.")
    convert_to_working_tiff(original, working, write_log)
    _log_existing_image(working, write_log, "working.tif")

    if settings.prestretched_input:
        write_log("Pre-stretched input mode enabled; skipping DeepSky/Siril stretch and color-stretch stage.")
        write_log("Using the uploaded image as calibrated image for denoise/star processing.")
        shutil.copy2(working, stretched)
        _log_existing_image(stretched, write_log, "stretched.tif")
        shutil.copy2(working, calibrated)
        _log_existing_image(calibrated, write_log, "calibrated.tif")
    elif mode == PipelineMode.STRETCH:
        write_log("Applying local astrophotography stretch.")
        stretched_image = astrophotography_stretch(load_image(working, write_log))
        save_tiff(stretched, stretched_image, write_log)
        _log_existing_image(stretched, write_log, "stretched.tif")
        shutil.copy2(stretched, calibrated)
        _log_existing_image(calibrated, write_log, "calibrated.tif")
    else:
        _run_siril_calibration(original, working, stretched, calibrated, job_folder, settings, write_log)

    current = calibrated

    if mode in {PipelineMode.FULL, PipelineMode.DEEPSNR}:
        deepsnr_exe = find_executable(Path(settings.deepsnr_folder))
        if not deepsnr_exe:
            raise FileNotFoundError("DeepSNR executable not found. Update the DeepSNR path in settings.")
        write_log(f"DeepSNR executable: {deepsnr_exe}")
        run_deepsnr(current, denoised, deepsnr_exe, write_log)
        _log_existing_image(denoised, write_log, "denoised.tif")
        current = denoised

    if mode in {PipelineMode.FULL, PipelineMode.STARNET}:
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
        add_images(starless, stars, final)
        _log_existing_image(final, write_log, "final.tif")
        current = final

    if mode in {PipelineMode.STRETCH, PipelineMode.DEEPSNR, PipelineMode.SIRIL}:
        shutil.copy2(current, final)
        _log_existing_image(final, write_log, "final.tif")

    after_preview = job_folder / "after_preview.png"
    preview_source = calibrated if mode == PipelineMode.SIRIL else final
    make_preview(preview_source, after_preview, log=write_log)
    calibrated_preview = job_folder / "calibrated_preview.png"
    make_preview(calibrated, calibrated_preview, log=write_log)
    write_log(f"Final image: {final}")
    write_log("Done.")

    return {
        "job_folder": job_folder,
        "before_preview": before_preview,
        "after_preview": after_preview,
        "final": final,
        "log": log_file,
    }
