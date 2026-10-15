"""Core facade for the 5D A/C post-processor."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..five_axis_postproc import FiveAxisPostProcessor as _LegacyFiveAxisPostProcessor
from ..five_axis_postproc import KinematicTableConfig, FiveAxisPostprocStats


class FiveAxisPostProcessor:
    """Application service for conservative, validated A/C annotation."""

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        limits: KinematicTableConfig | None = None,
    ) -> FiveAxisPostprocStats:
        return _LegacyFiveAxisPostProcessor(limits).process_gcode(input_path, output_path)

    def process_lines(
        self,
        lines: Iterable[str],
        *,
        limits: KinematicTableConfig | None = None,
    ) -> tuple[list[str], dict[str, int]]:
        raise NotImplementedError("The legacy post-processor works with files; use process().")


__all__ = ["KinematicTableConfig", "FiveAxisPostprocStats", "FiveAxisPostProcessor"]
