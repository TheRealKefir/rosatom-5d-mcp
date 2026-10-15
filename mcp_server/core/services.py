"""Transport-independent application services used by adapters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .fleet_client import FleetService
from .five_axis_postproc import FiveAxisPostProcessor
from .gcode_parser import GCodeSecurityService
from .mesh_analyzer import MeshAnalysisService
from .slicer_cli import SlicerService


class PrintingCore:
    """Compose all domain services without depending on MCP or another transport."""

    def __init__(self) -> None:
        self.mesh = MeshAnalysisService()
        self.slicer = SlicerService()
        self.security = GCodeSecurityService()
        self.five_axis = FiveAxisPostProcessor()
        self.fleet = FleetService()

    def analyze_stl(self, stl_path: str | Path) -> dict[str, Any]:
        return self.mesh.analyze(stl_path)

    def recommend_print_parameters(
        self,
        stl_path: str | Path,
        material: str,
        strength_priority: bool = False,
    ) -> dict[str, Any]:
        profile = self.slicer.profile(material)
        analysis = self.mesh.analyze(stl_path)
        strength = bool(strength_priority)
        return {
            "material": material,
            "nozzle_temp": profile.nozzle_temperature_c.recommended,
            "bed_temp": profile.bed_temperature_c.recommended,
            "print_speed": round(max(profile.print_speed_mm_s.min, profile.print_speed_mm_s.recommended * (0.65 if strength else 1.0)), 2),
            "infill_percent": 80 if strength else 35,
            "layer_height": profile.layer_height_mm.min if strength else profile.layer_height_mm.recommended,
            "cooling_percent": profile.cooling_percent.recommended,
            "supports": analysis["printability"]["supports_recommended"],
            "geometry": analysis,
        }

    def generate_safe_gcode(
        self,
        stl_path: str | Path,
        material: str,
        layer_height: float,
        infill: int,
        output_path: str | Path,
        *,
        mode_5d: bool = False,
    ) -> dict[str, Any]:
        result = self.slicer.generate(stl_path, material, layer_height, infill, output_path)
        if mode_5d:
            result["five_axis"] = self.five_axis.process(output_path, output_path)
            result["security"] = self.security.verify(output_path)
            if not result["security"]["safe"]:
                raise RuntimeError(f"postprocessed G-code failed safety audit: {result['security']['errors']}")
        return result

    def verify_gcode_security(self, gcode_path: str | Path) -> dict[str, Any]:
        return self.security.verify(gcode_path)

    def get_fleet_status(self) -> dict[str, Any]:
        return self.fleet.status()

    def send_to_printer(self, printer_id: str, gcode_path: str | Path) -> dict[str, Any]:
        return self.fleet.dispatch(printer_id, gcode_path)


__all__ = ["PrintingCore"]
