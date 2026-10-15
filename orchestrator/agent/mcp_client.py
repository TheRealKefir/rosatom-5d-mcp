# orchestrator/agent/mcp_client.py
from __future__ import annotations
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

log = logging.getLogger(__name__)


@dataclass
class ToolCallResult:
    """Результат вызова инструмента."""
    tool_name: str
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None


class MCPClient:
    """
    Клиент для MCP-сервера Росатома.
    
    Поднимает MCP-сервер как subprocess через stdio и предоставляет
    асинхронный доступ к инструментам.
    
    Использование:
        async with MCPClient("mcp_server/server.py") as client:
            result = await client.call_tool("analyze_stl", {"stl_path": "/path/to/model.stl"})
            print(result.data)
    """
    
    def __init__(
        self,
        server_script: str | Path,
        python_executable: str = "python",
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
    ):
        self.server_script = Path(server_script)
        self.python_executable = python_executable
        self.env = env or {}
        self.cwd = cwd or self.server_script.parent
        
        # Внутреннее состояние
        self._session: ClientSession | None = None
        self._cm = None  # context manager
        self._tools_cache: list[dict] | None = None
    
    async def __aenter__(self) -> MCPClient:
        """Поднимает MCP-сервер и устанавливает соединение."""
        server_params = StdioServerParameters(
            command=self.python_executable,
            args=[str(self.server_script)],
            env=self.env,
            cwd=str(self.cwd),
        )
        
        self._cm = stdio_client(server_params)
        read, write = await self._cm.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        
        log.info("MCP-сервер запущен: %s", self.server_script)
        
        tools_resp = await self._session.list_tools()
        self._tools_cache = [
            {"name": t.name, "description": t.description, "inputSchema": t.inputSchema}
            for t in tools_resp.tools
        ]
        log.info("Доступно инструментов: %d", len(self._tools_cache))
        
        return self
    
    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Останавливает MCP-сервер."""
        if self._session:
            await self._session.__aexit__(exc_type, exc, tb)
        if self._cm:
            await self._cm.__aexit__(exc_type, exc, tb)
        log.info("MCP-сервер остановлен")
    
    async def list_tools(self) -> list[dict]:
        """Возвращает список доступных инструментов (из кэша)."""
        if self._tools_cache is None:
            raise RuntimeError("MCPClient не инициализирован. Используй async with.")
        return self._tools_cache
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolCallResult:
        """
        Вызывает инструмент MCP-сервера.
        
        Args:
            tool_name: Имя инструмента (например, "analyze_stl")
            arguments: Аргументы для инструмента
        
        Returns:
            ToolCallResult с данными или ошибкой
        """
        if self._session is None:
            raise RuntimeError("MCPClient не инициализирован. Используй async with.")
        
        log.info("Вызов инструмента: %s(%s)", tool_name, json.dumps(arguments, ensure_ascii=False)[:200])
        
        try:
            result = await self._session.call_tool(tool_name, arguments)
            
            if not result.content:
                return ToolCallResult(tool_name=tool_name, success=False, error="Пустой ответ от инструмента")
            
            first_block = result.content[0]
            if hasattr(first_block, "text"):
                raw_text = first_block.text
            else:
                raw_text = str(first_block)
            
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError:
                data = {"raw_response": raw_text}
            
            if isinstance(data, dict) and data.get("error"):
                return ToolCallResult(
                    tool_name=tool_name,
                    success=False,
                    data=data,
                    error=data.get("error"),
                )
            
            return ToolCallResult(tool_name=tool_name, success=True, data=data)
        
        except Exception as e:
            log.exception("Ошибка вызова инструмента %s", tool_name)
            return ToolCallResult(tool_name=tool_name, success=False, error=str(e))
    
    async def analyze_stl(self, stl_path: str | Path) -> ToolCallResult:
        """Геометрический анализ STL-модели."""
        return await self.call_tool("analyze_stl", {"stl_path": str(stl_path)})
    
    async def recommend_print_parameters(
        self,
        stl_path: str | Path,
        material: str,
        strength_priority: bool = False,
    ) -> ToolCallResult:
        """Подбор параметров печати."""
        return await self.call_tool("recommend_print_parameters", {
            "stl_path": str(stl_path),
            "material": material,
            "strength_priority": strength_priority,
        })
    
    async def generate_safe_gcode(
        self,
        stl_path: str | Path,
        material: str,
        layer_height: float,
        infill: int,
        mode_5d: bool,
        output_gcode_path: str | Path,
    ) -> ToolCallResult:
        """Генерация G-кода с аудитом безопасности."""
        return await self.call_tool("generate_safe_gcode", {
            "stl_path": str(stl_path),
            "material": material,
            "layer_height": layer_height,
            "infill": infill,
            "mode_5d": mode_5d,
            "output_gcode_path": str(output_gcode_path),
        })
    
    async def verify_gcode_security(self, gcode_path: str | Path) -> ToolCallResult:
        """Аудит безопасности G-кода (SHA-256 + проверка опасных команд)."""
        return await self.call_tool("verify_gcode_security", {"gcode_path": str(gcode_path)})
    
    async def diagnose_print_defect_or_spool(
        self,
        defect_type: str,
        observed_symptoms: str,
        current_material: str | None = None,
    ) -> ToolCallResult:
        """Диагностика дефекта печати или проблемы с катушкой."""
        args = {
            "defect_type": defect_type,
            "observed_symptoms": observed_symptoms,
        }
        if current_material:
            args["current_material"] = current_material
        return await self.call_tool("diagnose_print_defect_or_spool", args)
    
    async def get_fleet_status(self) -> ToolCallResult:
        """Статус парка принтеров."""
        return await self.call_tool("get_fleet_status", {})
    
    async def send_to_printer(
        self,
        printer_id: str,
        gcode_path: str | Path,
    ) -> ToolCallResult:
        """Отправка G-кода на принтер."""
        return await self.call_tool("send_to_printer", {
            "printer_id": printer_id,
            "gcode_path": str(gcode_path),
        })