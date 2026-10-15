"""Core G-code security service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..security_guard import PhysicalMachineLimits, SlicingParametersInput, SecurityGuard
import hashlib


class GCodeSecurityService:
    """Application service for non-executing G-code validation."""

    def validate_parameters(self, **values: float) -> dict[str, Any]:
        return SecurityGuard().validate_slicing_parameters(values).to_dict()

    def verify(
        self,
        gcode_path: str | Path,
        *,
        expected_sha256: str | None = None,
        bed_size_mm: tuple[float, float] = (300.0, 300.0),
    ) -> dict[str, Any]:
        report = SecurityGuard(PhysicalMachineLimits(build_volume_x_mm=bed_size_mm[0], build_volume_y_mm=bed_size_mm[1])).analyze_gcode_file(gcode_path)
        result = report.to_dict()
        if expected_sha256 and report.sha256 != expected_sha256:
            result["is_safe"] = False
            result["violations"].append("SHA-256 checksum mismatch")
        return result

    def digest(self, gcode_path: str | Path) -> str:
        digest = hashlib.sha256()
        with open(gcode_path, "rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()


__all__ = [
    "GCodeSecurityService",
    "PhysicalMachineLimits",
    "SlicingParametersInput",
]
