"""Core facade for validated STL geometry analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..mesh_analyzer import analyze_mesh, load_mesh


class MeshAnalysisService:
    """Application service exposing geometry analysis to adapters."""

    def analyze(self, stl_path: str | Path, layer_height_mm: float = 0.2) -> dict[str, Any]:
        return analyze_mesh(stl_path, layer_height_mm)

    def load(self, stl_path: str | Path):
        return load_mesh(stl_path)


__all__ = ["MeshAnalysisService", "analyze_mesh", "load_mesh"]
