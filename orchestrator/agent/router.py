# orchestrator/agent/router.py
from __future__ import annotations
import json
import logging
import os
from typing import AsyncIterator, Any
from pathlib import Path

from openai import AsyncOpenAI

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

DEFAULT_OPENAI_API_URL = os.getenv("AI_API_URL", "https://openrouter.ai/api/v1")
DEFAULT_OPENAI_MODEL = os.getenv("AI_MODEL", "giga-chat-pro")
from pydantic import BaseModel

from .mcp_client import MCPClient
from .prompts_register import PromptRegistry
from .vlm_inspector import VLMInspector

log = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system" | "tool"
    content: Any  # str | list[dict] для мультимодальности
    name: str | None = None
    tool_call_id: str | None = None


class Router:
    """
    Оркестратор диалога: принимает запросы пользователя, управляет LLM,
    вызывает MCP-инструменты, возвращает ответы.
    """
    
    def __init__(
        self,
        llm_client: AsyncOpenAI | None,
        mcp_client: MCPClient,
        prompts: PromptRegistry,
        vlm_inspector: VLMInspector | None = None,
        model: str = DEFAULT_OPENAI_MODEL,
        max_tool_rounds: int = 5,
    ):
        self.llm = llm_client or AsyncOpenAI(
            api_key=os.getenv("AI_API_KEY"),
            base_url=DEFAULT_OPENAI_API_URL,
        )
        self.mcp = mcp_client
        self.prompts = prompts
        self.vlm = vlm_inspector
        self.model = model or DEFAULT_OPENAI_MODEL
        self.max_rounds = max_tool_rounds
        self.api_base_url = DEFAULT_OPENAI_API_URL
        
        self.history: list[ChatMessage] = []
    
    async def chat(
        self,
        user_message: str,
        stl_path: Path | None = None,
        image_paths: list[Path] | None = None,
    ) -> AsyncIterator[str]:
        """
        Основной метод: принимает запрос пользователя, возвращает стрим ответа.
        
        Args:
            user_message: Текстовый запрос пользователя
            stl_path: Путь к загруженному STL (опционально)
            image_paths: Пути к загруженным фото (опционально)
        
        Yields:
            str: Части ответа LLM (для стриминга в WebSocket)
        """
        messages = await self._build_messages(user_message, stl_path, image_paths)
        
        for round_num in range(self.max_rounds):
            log.info(f"Round {round_num + 1}/{self.max_rounds}, messages: {len(messages)}")
            
            response = await self.llm.chat.completions.create(
                model=self.model,
                messages=[m.model_dump(exclude_none=True) for m in messages],
                temperature=self.prompts.main_agent.meta.temperature or 0.2,
                max_tokens=self.prompts.main_agent.meta.max_tokens or 2048,
                stream=True,
            )
            
            assistant_content = ""
            tool_calls = []
            
            async for chunk in response:
                delta = chunk.choices[0].delta
                
                if delta.content:
                    assistant_content += delta.content
                    yield delta.content
                
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if len(tool_calls) <= idx:
                            tool_calls.append({
                                "id": tc_delta.id,
                                "type": "function",
                                "function": {"name": "", "arguments": ""}
                            })
                        if tc_delta.id:
                            tool_calls[idx]["id"] = tc_delta.id
                        if tc_delta.function.name:
                            tool_calls[idx]["function"]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments
            
            if not tool_calls:
                messages.append(ChatMessage(role="assistant", content=assistant_content))
                self.history.extend(messages[2:])
                return
            
            messages.append(ChatMessage(
                role="assistant",
                content=assistant_content or None,
                tool_calls=tool_calls
            ))
            
            for tc in tool_calls:
                tool_name = tc["function"]["name"]
                tool_args = json.loads(tc["function"]["arguments"])
                
                log.info(f"Calling tool: {tool_name}({tool_args})")
                
                try:
                    tool_result = await self.mcp.call_tool(tool_name, tool_args)
                    tool_result_str = json.dumps(tool_result, ensure_ascii=False)
                except Exception as e:
                    log.error(f"Tool {tool_name} failed: {e}")
                    tool_result_str = json.dumps({"error": str(e)})
                
                messages.append(ChatMessage(
                    role="tool",
                    content=tool_result_str,
                    name=tool_name,
                    tool_call_id=tc["id"]
                ))
        
        yield "\n\n⚠️ Превышено максимальное количество итераций. Попробуйте упростить запрос."
    
    async def _build_messages(
        self,
        user_message: str,
        stl_path: Path | None,
        image_paths: list[Path] | None,
    ) -> list[ChatMessage]:
        """
        Формирует messages array для LLM:
        - system prompt (main_agent)
        - история диалога
        - текущий запрос пользователя (с файлами)
        """
        messages = []
        
        system_prompt = self.prompts.main_agent.body
        messages.append(ChatMessage(role="system", content=system_prompt))
        
        messages.extend(self.history)
        
        user_content = []
        
        if image_paths:
            user_content.append({"type": "text", "text": user_message})
            for img_path in image_paths:
                if self.vlm:
                    vision_result = await self.vlm.analyze(img_path)
                    user_content.append({
                        "type": "text",
                        "text": f"[Vision-анализ {img_path.name}]: {json.dumps(vision_result, ensure_ascii=False)}"
                    })
                else:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"file://{img_path.absolute()}"}
                    })
        else:
            user_content.append({"type": "text", "text": user_message})
        
        if stl_path:
            from ..core.mesh_analyzer import MeshAnalyzer
            analyzer = MeshAnalyzer()
            geo_metrics = analyzer.analyze(stl_path)
            
            user_content.append({
                "type": "text",
                "text": f"[STL-анализ {stl_path.name}]: {json.dumps(geo_metrics, ensure_ascii=False)}"
            })
        
        if len(user_content) == 1 and user_content[0]["type"] == "text":
            messages.append(ChatMessage(role="user", content=user_message))
        else:
            messages.append(ChatMessage(role="user", content=user_content))
        
        return messages
    
    def clear_history(self):
        """Очистить историю диалога."""
        self.history.clear()
        log.info("History cleared")