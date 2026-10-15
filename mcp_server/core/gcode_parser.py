"""Core G-code security service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..security_guard import (
    MachineLimits,
    SlicingParameters,
    SlicingParameters as SlicingParameterModel,
    sha256_file,
    validate_slicing_parameters,
    verify_gcode,
)


class GCodeSecurityService:
    """Application service for non-executing G-code validation."""

    def validate_parameters(self, **values: float) -> dict[str, Any]:
        return validate_slicing_parameters(**values)

    def verify(
        self,
        gcode_path: str | Path,
        *,
        expected_sha256: str | None = None,
        bed_size_mm: tuple[float, float] = (300.0, 300.0),
    ) -> dict[str, Any]:
        return verify_gcode(gcode_path, expected_sha256=expected_sha256, bed_size_mm=bed_size_mm)

    def digest(self, gcode_path: str | Path) -> str:
        return sha256_file(gcode_path)


__all__ = [
    "GCodeSecurityService",
    "MachineLimits",
    "SlicingParameters",
    "SlicingParameterModel",
    "sha256_file",
    "validate_slicing_parameters",
    "verify_gcode",
]
