"""Core facade for validated STL geometry analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..mesh_analyzer import MeshAnalyzer


class MeshAnalysisService:
    """Application service exposing geometry analysis to adapters."""

    def analyze(self, stl_path: str | Path, layer_height_mm: float = 0.2) -> dict[str, Any]:
        return MeshAnalyzer().analyze(stl_path).to_dict()

    def load(self, stl_path: str | Path):
        return MeshAnalyzer().load_mesh(stl_path)


__all__ = ["MeshAnalysisService", "MeshAnalyzer"]
