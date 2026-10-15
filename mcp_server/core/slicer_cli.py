"""Core facade for external and fallback slicing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..slicer_cli import (
    FilamentDatabase,
    FilamentProfile,
    generate_gcode,
    get_filament_profile,
    load_filament_database,
)


class SlicerService:
    """Application service exposing slicing without transport concerns."""

    def generate(
        self,
        stl_path: str | Path,
        material: str,
        layer_height_mm: float,
        infill_percent: int,
        output_path: str | Path,
        *,
        slicer_binary: str | None = None,
        use_external: bool = True,
    ) -> dict[str, Any]:
        return generate_gcode(
            stl_path,
            material,
            layer_height_mm,
            infill_percent,
            output_path,
            slicer_binary=slicer_binary,
            use_external=use_external,
        )

    def profile(self, material: str) -> FilamentProfile:
        return get_filament_profile(material)


__all__ = [
    "FilamentDatabase",
    "FilamentProfile",
    "SlicerService",
    "generate_gcode",
    "get_filament_profile",
    "load_filament_database",
]
