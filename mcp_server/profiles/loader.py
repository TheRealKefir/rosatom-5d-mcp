"""Shared material/profile loading for the MCP server.

The JSON database is the source of truth for material values.  INI files are
machine/slicer presets and can be selected by name by adapters or deployments.
"""

from __future__ import annotations

import configparser
import json
from pathlib import Path
from typing import Any

PROFILES_DIR = Path(__file__).resolve().parent
FILAMENTS_PATH = PROFILES_DIR / "filaments.json"


def load_filaments(path: Path = FILAMENTS_PATH) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("filaments.json must contain an object")
    return {str(key).upper(): value for key, value in data.items()}


def get_filament(material: str, path: Path = FILAMENTS_PATH) -> dict[str, Any]:
    materials = load_filaments(path)
    key = material.strip().upper()
    if key not in materials:
        raise ValueError(f"Unknown material {material!r}; available: {sorted(materials)}")
    return dict(materials[key])


def load_preset(name: str, profiles_dir: Path = PROFILES_DIR) -> dict[str, Any]:
    """Load an INI preset by filename or stem and return plain dictionaries."""
    filename = name if name.endswith(".ini") else f"{name}.ini"
    path = profiles_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Profile not found: {path}")
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    result: dict[str, Any] = {"name": path.stem, "path": str(path)}
    for section in parser.sections():
        result[section] = dict(parser.items(section))
    return result


def list_presets(profiles_dir: Path = PROFILES_DIR) -> list[str]:
    return sorted(path.stem for path in profiles_dir.glob("*.ini"))


__all__ = ["FILAMENTS_PATH", "get_filament", "list_presets", "load_filaments", "load_preset"]
