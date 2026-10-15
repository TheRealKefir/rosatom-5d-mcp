"""In-process printer fleet simulation service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .gcode_parser import GCodeSecurityService


DEFAULT_FLEET: dict[str, dict[str, Any]] = {
    "printer_3d_1": {
        "name": "Rosatom FDM-3D-01",
        "kinematics": "cartesian_3d",
        "status": "IDLE",
        "nozzle_temperature_limit_c": 300,
        "bed_temperature_limit_c": 120,
        "bed_size_mm": {"x": 300, "y": 300, "z": 300},
    },
    "printer_3d_2": {
        "name": "Rosatom FDM-3D-02",
        "kinematics": "cartesian_3d",
        "status": "PRINTING",
        "nozzle_temperature_limit_c": 300,
        "bed_temperature_limit_c": 120,
        "bed_size_mm": {"x": 300, "y": 300, "z": 300},
    },
    "printer_5d_1": {
        "name": "Rosatom Stereotech-5D-01",
        "kinematics": "tilting_rotary_table_a_c",
        "status": "IDLE",
        "nozzle_temperature_limit_c": 420,
        "bed_temperature_limit_c": 150,
        "bed_size_mm": {"x": 500, "y": 500, "z": 500},
    },
}


class FleetService:
    """Safe simulation boundary; no network or machine command is sent."""

    def __init__(self, fleet: dict[str, dict[str, Any]] | None = None) -> None:
        self._fleet = {key: dict(value) for key, value in (fleet or DEFAULT_FLEET).items()}

    def status(self) -> dict[str, Any]:
        return {"simulation": True, "printers": {key: dict(value) for key, value in self._fleet.items()}}

    def dispatch(self, printer_id: str, gcode_path: str | Path) -> dict[str, Any]:
        if printer_id not in self._fleet:
            raise ValueError(f"unknown printer_id {printer_id!r}")
        printer = self._fleet[printer_id]
        bed = printer["bed_size_mm"]
        report = GCodeSecurityService().verify(gcode_path, bed_size_mm=(float(bed["x"]), float(bed["y"])))
        if not report["is_safe"]:
            return {
                "status": "REJECTED_SECURITY",
                "printer_id": printer_id,
                "sha256": report["sha256"],
                "security": report,
                "simulation": True,
            }
        return {
            "status": "DISPATCHED",
            "printer_id": printer_id,
            "printer": printer["name"],
            "sha256": report["sha256"],
            "started": False,
            "simulation": True,
            "message": "Validated job queued in simulation; no network or machine command was sent.",
        }


__all__ = ["DEFAULT_FLEET", "FleetService"]
