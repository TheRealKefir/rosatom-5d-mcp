"""
VLM Inspector — sends images (photos, STL renders) to a Vision-Language Model
and returns structured analysis (defect classification, context extraction).

Supports multiple VLM backends:
 - OpenAI GPT-4o
 - Qwen2.5-VL (via OpenAI-compatible API)
 - Claude 3.5 Sonnet (via Anthropic API)
 - Local Ollama models with vision capability
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Optional

from openai import AsyncOpenAI
from loguru import logger


VLM_MODEL = os.getenv("VLM_MODEL", "qwen2.5-vl-72b-instruct")
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "https://api.openai.com/v1")
VLM_API_KEY = os.getenv("VLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))

# System prompts for different analysis modes
_DEFECT_SYSTEM = (
    "Ты — эксперт-технолог по FDM 3D/5D-печати с 15-летним опытом. "
    "Анализируй изображения напечатанных деталей и возвращай ТОЛЬКО валидный JSON с полями:\n"
    "- detected_defects: list[str] — список обнаруженных дефектов "
    "(warping, stringing, delamination, under-extrusion, layer_shift, elephant_foot, "
    "over-extrusion, blobbing, z-wobble)\n"
    "- severity: str — 'low' | 'medium' | 'high' | 'critical'\n"
    "- material_suggestion: str — предположение о материале по визуальным признакам\n"
    "- infill_suggestion: int — рекомендуемый процент заполнения (0-100)\n"
    "- temperature_adjustments: dict — {nozzle_delta: int, bed_delta: int, fan_speed: int}\n"
    "- notes: str — пояснение на русском языке, что видно на фото и какие действия нужны\n"
    "Не добавляй пояснений вне JSON."
)

_CONTEXT_SYSTEM = (
    "Ты — инженер-технолог аддитивного производства. Анализируй фото деталей, "
    "чертежей и мест установки. Определи:\n"
    "- Тип детали (корпус, кронштейн, шестерня, крыльчатка и т.д.)\n"
    "- Условия эксплуатации (температура, нагрузки, среда)\n"
    "- Подходящие материалы для FDM-печати\n"
    "- Рекомендации по режиму печати (3D или 5D)\n\n"
    "Возвращай ТОЛЬКО валидный JSON с полями:\n"
    "- part_type: str\n"
    "- operating_conditions: dict {temperature_c: int, load_type: str, environment: str}\n"
    "- material_suggestion: str\n"
    "- infill_suggestion: int\n"
    "- recommended_mode: str — '3D' или '5D'\n"
    "- notes: str"
)

_STL_RENDER_SYSTEM = (
    "Ты — эксперт по геометрическому анализу 3D-моделей для аддитивного производства. "
    "Перед тобой рендеры STL-модели с четырёх ракурсов. Красным подсвечены зоны с углом "
    "нависания более 45°. Оцени:\n"
    "- Долю нависающих элементов (визуально)\n"
    "- Наличие мостов (bridges) — плоских горизонтальных участков без опоры снизу\n"
    "- Тонкие стенки или мелкие элементы\n"
    "- Рекомендацию 3D или 5D режима печати\n\n"
    "Возвращай ТОЛЬКО валидный JSON с полями:\n"
    "- visual_overhang_estimate: str — 'none' | 'minimal' | 'moderate' | 'severe'\n"
    "- bridges_detected: bool\n"
    "- thin_walls_detected: bool\n"
    "- recommended_mode: str — '3D' или '5D'\n"
    "- notes: str"
)


def _get_mime_type(path: str) -> str:
    """Determine MIME type from file extension."""
    ext = Path(path).suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return mime_map.get(ext, "image/png")


class VLMInspector:
    """Analyse images via a multimodal LLM (GPT-4o / Qwen-VL / Claude)."""

    def __init__(
        self,
        model: str = VLM_MODEL,
        base_url: str = VLM_BASE_URL,
        api_key: str = VLM_API_KEY,
    ):
        self._model = model
        self._client = AsyncOpenAI(
            api_key=api_key or "sk-stub",
            base_url=base_url,
        )

    @staticmethod
    def _encode_image(path: str) -> tuple[str, str]:
        """Encode image as base64 and return (b64_data, mime_type)."""
        mime = _get_mime_type(path)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return b64, mime

    def _build_image_content(self, path: str) -> dict:
        """Build OpenAI-compatible image_url content block."""
        b64, mime = self._encode_image(path)
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        }

    async def _call_vlm(
        self,
        system_prompt: str,
        user_content: list[dict],
        temperature: float = 0.2,
    ) -> dict:
        """Send request to VLM and parse JSON response."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
            return json.loads(raw)

        except json.JSONDecodeError as e:
            logger.warning("VLM returned non-JSON response: {}", raw[:200])
            # Try to extract JSON from response
            try:
                import re
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except Exception:
                pass
            return {"error": f"Invalid JSON from VLM: {e}", "raw_response": raw[:500]}

        except Exception as e:
            logger.error("VLM call failed: {}", e)
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze_photo(
        self,
        photo_path: str,
        context_prompt: str = "",
        render_paths: Optional[list[str]] = None,
    ) -> dict:
        """
        Universal photo analysis: send user photo + optional STL renders to VLM.
        
        Automatically detects the analysis mode:
        - If photo_path looks like a defect photo → defect analysis
        - If render_paths provided without photo → STL geometry analysis
        - Otherwise → context analysis (part type, requirements)
        
        Returns structured JSON with analysis results.
        """
        user_content: list[dict] = []

        # Add text context
        if context_prompt:
            user_content.append({"type": "text", "text": context_prompt})

        # Determine system prompt based on content
        has_photo = photo_path and Path(photo_path).exists()
        has_renders = render_paths and any(Path(p).exists() for p in render_paths)

        if has_photo and not has_renders:
            # Photo-only mode → defect or context analysis
            user_content.append(self._build_image_content(photo_path))
            # Check if context suggests defect analysis
            defect_keywords = ["дефект", "брак", "деламинация", "отрыв", "паутин",
                             "warping", "stringing", "layer", "смещ", "волос",
                             "неудач", "плох", "испорч"]
            is_defect = any(kw in context_prompt.lower() for kw in defect_keywords)
            system = _DEFECT_SYSTEM if is_defect else _CONTEXT_SYSTEM
        elif has_renders and not has_photo:
            # Renders-only → STL geometry analysis
            for rp in render_paths:
                if Path(rp).exists():
                    user_content.append(self._build_image_content(rp))
            system = _STL_RENDER_SYSTEM
        elif has_photo and has_renders:
            # Both photo and renders → full context analysis
            user_content.append(self._build_image_content(photo_path))
            for rp in render_paths:
                if Path(rp).exists():
                    user_content.append(self._build_image_content(rp))
            system = _CONTEXT_SYSTEM
        else:
            # No valid images
            logger.warning("VLM: no valid images provided")
            return {
                "error": "No valid images provided",
                "detected_defects": [],
                "notes": "Не удалось загрузить изображения для анализа",
            }

        if not user_content:
            user_content.append({"type": "text", "text": "Проанализируй изображения"})

        return await self._call_vlm(system, user_content)

    async def analyze_defect(self, photo_path: str, material: str = "") -> dict:
        """
        Specialized defect analysis for closed-loop correction.
        Always uses the defect detection system prompt.
        """
        user_content: list[dict] = []

        if material:
            user_content.append({
                "type": "text",
                "text": f"Материал: {material}. Определи дефект печати на этом фото.",
            })

        if Path(photo_path).exists():
            user_content.append(self._build_image_content(photo_path))
        else:
            return {"error": f"Photo not found: {photo_path}"}

        return await self._call_vlm(_DEFECT_SYSTEM, user_content)

    async def analyze_stl_renders(self, render_paths: list[str]) -> dict:
        """
        Analyze STL model renders to assess printability.
        """
        user_content: list[dict] = [
            {"type": "text", "text": "Оцени технологичность 3D-модели по рендерам:"},
        ]

        valid_paths = [p for p in render_paths if Path(p).exists()]
        if not valid_paths:
            return {"error": "No valid render images found"}

        for rp in valid_paths:
            user_content.append(self._build_image_content(rp))

        return await self._call_vlm(_STL_RENDER_SYSTEM, user_content)
