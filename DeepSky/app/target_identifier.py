from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from astropy.coordinates import SkyCoord
import astropy.units as u

from .plate_solver import PlateSolveResult


@dataclass
class TargetIdentification:
    identified: bool
    target_name: str | None
    catalog_ids: list[str]
    object_type: str | None
    angular_distance_from_center_deg: float | None
    confidence: float
    expected_color_family: list[str]
    processing_notes: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CatalogTarget:
    name: str
    ids: tuple[str, ...]
    ra_deg: float
    dec_deg: float
    object_type: str
    size_deg: float
    expected_color_family: tuple[str, ...]
    notes: tuple[str, ...] = ()


CATALOG: tuple[CatalogTarget, ...] = (
    CatalogTarget("Andromeda Galaxy", ("M31", "NGC 224"), 10.6847, 41.2690, "galaxy", 3.1, ("warm core", "blue arms", "dust lanes")),
    CatalogTarget("Whirlpool Galaxy", ("M51", "NGC 5194"), 202.4696, 47.1952, "galaxy", 0.19, ("warm core", "blue arms")),
    CatalogTarget("Bode's Galaxy", ("M81", "NGC 3031"), 148.8882, 69.0653, "galaxy", 0.45, ("gold core", "blue arms")),
    CatalogTarget("M66", ("M66", "NGC 3627"), 170.0625, 12.9915, "galaxy", 0.15, ("warm core", "subtle blue arms")),
    CatalogTarget("Horsehead Nebula", ("IC 434", "Barnard 33"), 85.2440, -2.4580, "emission/dark nebula", 1.0, ("red-brown", "pink", "warm dust"), ("protect red-brown signal",)),
    CatalogTarget("IC 63", ("IC 63", "Ghost of Cassiopeia"), 14.7750, 60.9130, "emission/reflection nebula", 0.18, ("red-brown", "pink", "warm dust"), ("avoid green/gray fallback",)),
    CatalogTarget("Western Veil Nebula", ("NGC 6960", "Caldwell 34"), 312.7830, 30.7080, "supernova remnant", 1.2, ("red", "cyan", "blue")),
    CatalogTarget("Eastern Veil Nebula", ("NGC 6992", "NGC 6995"), 313.7330, 31.7420, "supernova remnant", 1.2, ("red", "cyan", "blue")),
    CatalogTarget("Pacman Nebula", ("NGC 281", "IC 11"), 13.1880, 56.6240, "emission nebula", 0.58, ("red", "pink", "warm dust", "pale cyan")),
    CatalogTarget("Bubble Nebula", ("NGC 7635",), 350.2010, 61.2010, "emission nebula", 0.25, ("red", "pink", "blue/cyan shell")),
    CatalogTarget("Eagle Nebula", ("M16", "NGC 6611"), 274.7000, -13.8067, "emission nebula", 0.58, ("red", "gold", "blue/cyan core")),
    CatalogTarget("Orion Nebula", ("M42", "NGC 1976"), 83.8221, -5.3911, "emission/reflection nebula", 1.1, ("blue/cyan core", "red-brown", "warm dust")),
)


def _empty_identification() -> TargetIdentification:
    return TargetIdentification(False, None, [], None, None, 0.0, [], {})


def identify_target(result: PlateSolveResult, object_hint: str | None = None) -> TargetIdentification:
    if not result.solved or result.ra_deg is None or result.dec_deg is None:
        return _empty_identification()

    center = SkyCoord(result.ra_deg * u.deg, result.dec_deg * u.deg)
    fov = max(
        value for value in (result.fov_width_deg, result.fov_height_deg, 0.75) if value is not None
    )
    search_radius = max(0.35, min(4.0, fov * 0.85))
    hint = (object_hint or result.object_name or "").strip().lower()

    best: tuple[float, CatalogTarget, float] | None = None
    for target in CATALOG:
        target_coord = SkyCoord(target.ra_deg * u.deg, target.dec_deg * u.deg)
        distance = float(center.separation(target_coord).deg)
        if distance > search_radius:
            continue
        hint_bonus = 0.65 if hint and (target.name.lower() in hint or any(item.lower() in hint for item in target.ids)) else 0.0
        size_fit = min(1.0, target.size_deg / max(0.08, fov)) if fov else 0.5
        score = max(0.0, 1.0 - distance / search_radius) + hint_bonus + size_fit * 0.25
        if best is None or score > best[0]:
            best = (score, target, distance)

    if best is None:
        return _empty_identification()

    score, target, distance = best
    confidence = float(max(0.15, min(0.98, score / 1.9)))
    notes = {
        "soft_guardrails_only": True,
        "target_color_family_is_not_a_recoloring_rule": True,
        "notes": list(target.notes),
        "fov_deg": None if fov is None or math.isnan(fov) else fov,
    }
    return TargetIdentification(
        True,
        target.name,
        list(target.ids),
        target.object_type,
        distance,
        confidence,
        list(target.expected_color_family),
        notes,
    )
