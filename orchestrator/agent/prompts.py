"""
Central prompt registry — loads .md / .j2 templates from the prompts/ folder.
Supports both:
 - Simple file loading (by filename)
 - Jinja2 template rendering with variables
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader
from loguru import logger

PROMPTS_DIR = Path(__file__).parent / "prompts"

_env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)), autoescape=False)


def load_prompt(name: str, **kwargs) -> str:
    """
    Load a prompt template by filename (with extension).
    Jinja2 variables are filled from kwargs.
    """
    tpl = _env.get_template(name)
    return tpl.render(**kwargs)


def _safe_read(filename: str) -> str:
    """Safely read a prompt file, stripping YAML front matter if present."""
    path = PROMPTS_DIR / filename
    if not path.exists():
        logger.warning("Prompt file not found: {}", path)
        return ""
    
    text = path.read_text(encoding="utf-8")
    
    # Strip YAML front matter (---\n...\n---\n)
    if text.startswith("---"):
        lines = text.split("\n")
        end_idx = -1
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_idx = i
                break
        if end_idx > 0:
            text = "\n".join(lines[end_idx + 1:])
    
    return text.strip()


# Pre-load commonly used system prompts
MAIN_SYSTEM = _safe_read("main_agents.system.md")
VISION_SYSTEM = _safe_read("vision_analyzer.system.md")
PARAMS_SYSTEM = _safe_read("params_calculator.system.md")
REPORT_SYSTEM = _safe_read("report_formatter.system.md")
SAFETY_SYSTEM = _safe_read("safety_refusal.system.md")