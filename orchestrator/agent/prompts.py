from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import json

Role = Literal["system", "user", "assistant"]

@dataclass(frozen=True, slots=True)
class PromptMeta:
    name: str
    version: str = "1.0"
    role: Role = "system"
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    vars: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    path: Path | None = None

@dataclass(frozen=True, slots=True)
class Prompt:
    meta: PromptMeta
    body: str
    _template: Any = field(repr=False, default=None, compare=False)

    def render(self, **values: Any) -> str:
        if not self.meta.vars:
            return self.body.strip()
        missing = set(self.meta.vars) - values.keys()
        if missing:
            raise KeyError(f"[{self.meta.name}] no such variables: {sorted(missing)}")
        return self._template.render(**values).strip()

    def as_message(self, **values: Any) -> dict:
        return {"role": self.meta.role, "content": self.render(**values)}

    def __str__(self) -> str:
        return self.body.strip()