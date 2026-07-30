from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.astrosharp_native import (
    MODEL_BUNDLE,
    apply_astrosharp_native_dual,
    run_astrosharp_native_dual,
)
from app.astrosharp_v2 import AstroSharpPsfEstimate, estimate_astrosharp_psf


def _synthetic_star_field(size: int = 112) -> np.ndarray:
    y, x = np.mgrid[:size, :size].astype(np.float32)
    image = np.full((size, size, 3), 0.025, dtype=np.float32)
    for row in range(18, size - 12, 18):
        for col in range(16, size - 12, 19):
            radius2 = (y - row) ** 2 + (x - col) ** 2
            star = np.exp(-radius2 / (2.0 * 1.8**2)) * 0.72
            image[..., 0] += star
            image[..., 1] += star * 0.90
            image[..., 2] += star * 0.78
    ridge = np.exp(-((y - (38.0 + x * 0.25)) ** 2) / (2.0 * 8.0**2))
    image[..., 0] += ridge * 0.22
    image[..., 1] += ridge * 0.08
    return np.clip(image, 0.0, 1.0)


def test_native_bundle_contains_every_official_quarter_step_psf() -> None:
    with np.load(MODEL_BUNDLE) as bundle:
        for quarter in range(4, 33):
            key = f"psf_{quarter / 4:g}".replace(".", "p")
            for layer in range(1, 6):
                assert f"{key}_w{layer}" in bundle.files


def test_psf_estimator_uses_measured_stars() -> None:
    estimate = estimate_astrosharp_psf(_synthetic_star_field())
    assert estimate.accepted_stars >= 5
    assert estimate.confidence in {"limited", "measured"}
    assert 1.0 <= estimate.dso_psf <= 3.0
    assert estimate.star_psf <= estimate.dso_psf


def test_native_dualpsf_executes_distinct_models_and_writes_proof(
    tmp_path: Path,
) -> None:
    image = _synthetic_star_field(96)
    psf = AstroSharpPsfEstimate(4.7, 2.0, 1.75, 25, "measured")
    manifest_path = tmp_path / "astrosharp_manifest.json"
    first = run_astrosharp_native_dual(
        image,
        psf=psf,
        chunk_size=64,
        manifest_path=manifest_path,
    )
    second = run_astrosharp_native_dual(image, psf=psf, chunk_size=64)

    assert first.dso_donor.shape == image.shape
    assert first.star_donor.shape == image.shape
    assert float(np.mean(np.abs(first.dso_donor - image))) > 1e-4
    assert float(np.mean(np.abs(first.star_donor - image))) > 1e-4
    assert float(np.mean(np.abs(first.dso_donor - first.star_donor))) > 1e-5
    assert np.array_equal(first.dso_donor, second.dso_donor)
    assert np.array_equal(first.star_donor, second.star_donor)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["actually_executed"] is True
    assert manifest["engine"] == "AstroSharp DualPSF native"
    assert manifest["dso_donor_sha256"] != manifest["input_array_sha256"]
    assert manifest["star_donor_sha256"] != manifest["input_array_sha256"]
    assert manifest["dso_donor_sha256"] != manifest["star_donor_sha256"]

def test_native_dualpsf_blend_records_accepted_output(tmp_path: Path) -> None:
    image = _synthetic_star_field(96)
    psf = AstroSharpPsfEstimate(4.7, 2.0, 1.75, 25, "measured")
    manifest_path = tmp_path / "accepted_manifest.json"
    result, inference = apply_astrosharp_native_dual(
        image,
        psf=psf,
        maximum_mix=0.90,
        chunk_size=64,
        manifest_path=manifest_path,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert result.shape == image.shape
    assert manifest["accepted_delta_mean"] > 0.0
    assert manifest["accepted_output_sha256"]
    assert manifest["maximum_mix"] == 0.90
    assert inference.manifest == manifest
