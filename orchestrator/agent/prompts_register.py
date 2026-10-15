"""
MCP tool descriptions used by the LLM as function-calling schema.
Loaded from prompts/tools/*.md and converted to OpenAI-style tool dicts.

These tool definitions match the ACTUAL MCP server tools in mcp_server/server.py:
 - analyze_stl
 - recommend_print_parameters
 - generate_safe_gcode
 - verify_gcode_security
 - diagnose_print_defect_or_spool
 - get_fleet_status
 - send_to_printer
"""

from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).parent / "prompts" / "tools"


def _load_tool_description(filename: str) -> str:
    """Load tool description from markdown file, stripping front matter."""
    path = TOOLS_DIR / filename
    if not path.exists():
        return f"Tool description from {filename}"
    
    text = path.read_text(encoding="utf-8")
    
    # Strip YAML front matter
    if text.startswith("---"):
        lines = text.split("\n")
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                text = "\n".join(lines[i + 1:])
                break
    
    return text.strip()


# OpenAI function-calling compatible definitions
# These match the actual @mcp.tool() functions in mcp_server/server.py
MCP_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "analyze_stl",
            "description": _load_tool_description("analyze_ctl_geometry.md"),
            "parameters": {
                "type": "object",
                "properties": {
                    "stl_path": {
                        "type": "string",
                        "description": "Путь к STL-файлу 3D-модели",
                    }
                },
                "required": ["stl_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_print_parameters",
            "description": _load_tool_description("calculate_print_parameters.md"),
            "parameters": {
                "type": "object",
                "properties": {
                    "stl_path": {
                        "type": "string",
                        "description": "Путь к STL-файлу",
                    },
                    "material": {
                        "type": "string",
                        "description": "Целевой материал (PLA, ABS, PETG, PA12-CF, PEEK)",
                        "enum": ["PLA", "ABS", "PETG", "PA12-CF", "PEEK", "ASA", "PC", "NYLON"],
                    },
                    "strength_priority": {
                        "type": "string",
                        "description": "Приоритет прочности: 'max_strength', 'balanced', 'fast_prototype'",
                        "enum": ["max_strength", "balanced", "fast_prototype"],
                    },
                },
                "required": ["stl_path", "material"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_safe_gcode",
            "description": _load_tool_description("slice_model.md"),
            "parameters": {
                "type": "object",
                "properties": {
                    "stl_path": {
                        "type": "string",
                        "description": "Путь к STL-файлу",
                    },
                    "material": {
                        "type": "string",
                        "description": "Материал печати (PLA, ABS, PETG, PA12-CF, PEEK)",
                    },
                    "layer_height": {
                        "type": "number",
                        "description": "Высота слоя в мм (0.1-0.4)",
                    },
                    "infill": {
                        "type": "integer",
                        "description": "Процент заполнения (0-100)",
                    },
                    "mode_5d": {
                        "type": "boolean",
                        "description": "Включить 5D-режим Stereotech (многоосевая печать без поддержек)",
                    },
                    "output_gcode_path": {
                        "type": "string",
                        "description": "Путь для сохранения G-кода (опционально)",
                    },
                },
                "required": ["stl_path", "material", "layer_height", "infill", "mode_5d"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_gcode_security",
            "description": (
                "Статический аудит безопасности G-кода перед отправкой на принтер. "
                "Проверяет: наличие G28 (homing), температурные лимиты, "
                "габаритный коридор, вычисляет SHA-256 хэш."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "gcode_path": {
                        "type": "string",
                        "description": "Путь к .gcode файлу для проверки",
                    }
                },
                "required": ["gcode_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diagnose_print_defect_or_spool",
            "description": _load_tool_description("analyze_visual_data.md"),
            "parameters": {
                "type": "object",
                "properties": {
                    "defect_type": {
                        "type": "string",
                        "description": "Тип дефекта: warping, stringing, delamination, under_extrusion, layer_shift, wet_filament",
                    },
                    "observed_symptoms": {
                        "type": "string",
                        "description": "Описание наблюдаемых симптомов",
                    },
                    "current_material": {
                        "type": "string",
                        "description": "Текущий используемый материал",
                    },
                    "current_nozzle_temp_c": {
                        "type": "number",
                        "description": "Текущая температура сопла (опционально)",
                    },
                    "current_bed_temp_c": {
                        "type": "number",
                        "description": "Текущая температура стола (опционально)",
                    },
                },
                "required": ["defect_type", "observed_symptoms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fleet_status",
            "description": _load_tool_description("manage_printer_fleet.md"),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_to_printer",
            "description": (
                "Отправляет проверенный G-код на конкретный принтер из парка. "
                "Перед отправкой автоматически выполняет проверку безопасности "
                "и соответствия габаритов области печати принтера."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "printer_id": {
                        "type": "string",
                        "description": "ID принтера (STEREO-5D-01, STEREO-5D-02, PRUSA-3D-01)",
                    },
                    "gcode_path": {
                        "type": "string",
                        "description": "Путь к .gcode файлу для отправки",
                    },
                },
                "required": ["printer_id", "gcode_path"],
            },
        },
    },
]