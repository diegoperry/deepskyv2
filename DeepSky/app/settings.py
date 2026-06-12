from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


if getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(sys.executable).resolve().parent
    APP_ROOT = PROJECT_ROOT
else:
    APP_ROOT = Path(__file__).resolve().parents[1]
    PROJECT_ROOT = APP_ROOT.parent
SETTINGS_PATH = PROJECT_ROOT / "settings.json"


@dataclass
class AppSettings:
    starnet_folder: str
    deepsnr_folder: str
    siril_folder: str
    output_folder: str
    color_calibration_mode: str
    siril_object_name: str
    siril_ra_dec: str
    siril_focal_length: str
    siril_pixel_size: str
    siril_apply_scnr: bool
    siril_color_saturation: int
    siril_deconvolution_enabled: bool
    starless_test_enabled: bool
    star_handling_mode: str
    siril_debug_mode: bool
    galaxy_background_smoothness: int
    galaxy_background_darkness: int
    galaxy_chroma_noise_reduction: int
    galaxy_protect_detail: bool
    input_processing_mode: str
    stretch_level: str
    telescope_profile: str
    prestretched_input: bool
    object_type: str


def _default_tool_folder(name: str, extracted_prefix: str) -> Path:
    tools_path = PROJECT_ROOT / "tools" / name
    if tools_path.exists() and any(tools_path.rglob("*.exe")):
        return tools_path

    matches = sorted(PROJECT_ROOT.glob(f"{extracted_prefix}*"))
    for match in matches:
        if match.is_dir() and any(match.rglob("*.exe")):
            return match
    return tools_path


def default_settings() -> AppSettings:
    return AppSettings(
        starnet_folder=str(_default_tool_folder("starnet", "starnet")),
        deepsnr_folder=str(_default_tool_folder("deepsnr", "deepsnr")),
        siril_folder=str(_default_tool_folder("siril", "siril")),
        output_folder=str(PROJECT_ROOT / "outputs"),
        color_calibration_mode="Basic",
        siril_object_name="",
        siril_ra_dec="",
        siril_focal_length="",
        siril_pixel_size="",
        siril_apply_scnr=False,
        siril_color_saturation=15,
        siril_deconvolution_enabled=True,
        starless_test_enabled=True,
        star_handling_mode="Slight Star Reduction",
        siril_debug_mode=False,
        galaxy_background_smoothness=90,
        galaxy_background_darkness=90,
        galaxy_chroma_noise_reduction=96,
        galaxy_protect_detail=True,
        input_processing_mode="Auto",
        stretch_level="Standard",
        telescope_profile="Auto",
        prestretched_input=False,
        object_type="Nebula",
    )


def load_settings() -> AppSettings:
    defaults = default_settings()
    if not SETTINGS_PATH.exists():
        save_settings(defaults)
        return defaults

    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults

    merged = asdict(defaults)
    merged.update({k: v for k, v in data.items() if k in merged and v is not None})
    return AppSettings(**merged)


def save_settings(settings: AppSettings) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        json.dumps(asdict(settings), indent=2),
        encoding="utf-8",
    )
