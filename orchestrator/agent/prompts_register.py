from __future__ import annotations
import re, logging
from pathlib import Path
from typing import Iterator
import yaml
from jinja2 import Environment, BaseLoader, meta

from .types import Prompt, PromptMeta

log = logging.getLogger("prompts")

_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

class PromptError(RuntimeError):
    """Error with prompts."""


class PromptRegistry:
    def __init__(self, root: str | Path, strict: bool = True):
        self.root = Path(root)
        self.strict = strict
        self._cache: dict[str, Prompt] = {}
        self._env = Environment(loader=BaseLoader(), autoescape=False,
                                trim_blocks=True, lstrip_blocks=True)
        self.load_all()

    def load_all(self) -> None:
        self._cache.clear()
        files = sorted(self.root.glob("*.md")) + sorted(self.root.glob("*.j2"))
        for path in files:
            if "_drafts" in path.parts:
                continue
            p = self._parse(path)
            self._validate(p)
            self._cache[p.meta.name] = p
        log.info("загружено промптов: %d → %s",
                 len(self._cache), ", ".join(f"{k}@{v.meta.version}" for k, v in self._cache.items()))

    def _parse(self, path: Path) -> Prompt:
        raw = path.read_text(encoding="utf-8")
        m = _FRONT.match(raw)
        if not m:
            raise PromptError(f"{path}: no YAML front matter")
        meta_d = yaml.safe_load(m.group(1)) or {}
        body = raw[m.end():]
        meta = PromptMeta(
            name=meta_d.get("name", path.stem.split(".")[0]),
            version=str(meta_d.get("version", "1.0")),
            role=meta_d.get("role", "system"),
            model=meta_d.get("model"),
            temperature=meta_d.get("temperature"),
            max_tokens=meta_d.get("max_tokens"),
            vars=tuple(meta_d.get("vars", []) or []),
            tags=tuple(meta_d.get("tags", []) or []),
            path=path,
        )
        tpl = self._env.from_string(body) if meta.vars else None
        return Prompt(meta=meta, body=body, _template=tpl)

    def _validate(self, p: Prompt) -> None:
        declared = set(p.meta.vars)
        if p._template is not None:
            found = meta.find_undeclared_variables(p._template.environment.parse(p.body))
            undeclared = found - declared
            unused = declared - found
            if undeclared and self.strict:
                raise PromptError(
                    f"[{p.meta.name}] в шаблоне есть {{{{ {sorted(undeclared)} }}}}, "
                    f"но они не объявлены в front matter 'vars'")
            if unused and self.strict:
                raise PromptError(
                    f"[{p.meta.name}] объявлены, но не используются: {sorted(unused)}")
        if p.meta.name in self._cache:
            raise PromptError(f"дубликат имени промпта: {p.meta.name}")

    def get(self, name: str) -> Prompt:
        try:
            return self._cache[name]
        except KeyError:
            raise PromptError(f"промпт '{name}' не найден. Есть: {sorted(self._cache)}") from None

    def __getattr__(self, name: str) -> Prompt:
        if name.startswith("_"):
            raise AttributeError(name)
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._cache

    def __iter__(self) -> Iterator[Prompt]:
        return iter(self._cache.values())

    def reload(self) -> None:
        log.warning("reload prompts...")
        self.load_all()