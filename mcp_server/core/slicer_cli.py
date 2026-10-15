"""Core facade for external and fallback slicing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..slicer_cli import SlicerCLI, SlicingJobResult


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
        return SlicerCLI().slice_mesh(stl_path, output_path, material, layer_height_mm, infill_percent).to_dict()

    def profile(self, material: str) -> dict[str, Any]:
        return SlicerCLI().filaments_db.get(material.upper(), SlicerCLI().filaments_db["PLA"])


__all__ = [
    "SlicingJobResult",
    "SlicerService",
]
