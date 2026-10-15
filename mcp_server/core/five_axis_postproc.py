"""Core facade for the 5D A/C post-processor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..five_axis_postproc import FiveAxisLimits, postprocess_gcode, postprocess_lines


class FiveAxisPostProcessor:
    """Application service for conservative, validated A/C annotation."""

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        limits: FiveAxisLimits | None = None,
    ) -> dict[str, Any]:
        return postprocess_gcode(input_path, output_path, limits=limits)

    def process_lines(
        self,
        lines: Iterable[str],
        *,
        limits: FiveAxisLimits | None = None,
    ) -> tuple[list[str], dict[str, int]]:
        return postprocess_lines(lines, limits)


__all__ = ["FiveAxisLimits", "FiveAxisPostProcessor", "postprocess_gcode", "postprocess_lines"]
