# orchestrator/agent/mcp_client.py
"""
Rosatom 5D Additive Manufacturing MCP Client.

Provides an async interface for orchestrating MCP tools via:
1. Official Model Context Protocol (MCP) Python SDK over Server-Sent Events (SSE).
2. Robust REST fallback over HTTP (POST /call_tool) for high availability during hackathon operations.

Features:
- Dual-transport architecture (Official MCP SDK with automatic REST fallback)
- Dynamic timeouts: 120s for slicer/gcode workloads, 30s for analytical calls
- Intelligent tool name and argument aliasing for cross-module compatibility
- Automatic reconnection and retry logic on connection dropouts
- Structured execution timing and operational logging with loguru
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from loguru import logger

# Official MCP SDK imports
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    import mcp.types as mcp_types
    HAS_MCP_SDK = True
except ImportError:
    HAS_MCP_SDK = False
    ClientSession = None  # type: ignore[assignment, misc]
    sse_client = None  # type: ignore[assignment, misc]
    mcp_types = None  # type: ignore[assignment, misc]


# Configuration & Defaults
DEFAULT_MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://mcp-server:8000")
DEFAULT_TIMEOUT: float = 30.0
SLICER_TIMEOUT: float = 120.0
SLICER_TOOL_KEYWORDS: tuple[str, ...] = ("slice", "slicing", "gcode", "generate_safe_gcode")

# Tool name aliases for backward compatibility across orchestrator modules
TOOL_ALIASES: dict[str, str] = {
    "analyze_stl_geometry": "analyze_stl",
    "get_printer_fleet_status": "get_fleet_status",
    "dispatch_print_job": "send_to_printer",
    "slice_model_to_gcode": "generate_safe_gcode",
}


@dataclass
class ToolCallResult:
    """Результат вызова инструмента (сохранен для обратной совместимости)."""
    tool_name: str
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None


class MCPClient:
    """
    Async client for Rosatom Additive Manufacturing MCP Server.

    Connects to the MCP server using SSE transport (via official MCP SDK)
    and falls back to HTTP REST endpoints (POST /call_tool) when SSE is
    unavailable.
    """

    def __init__(self, base_url: str | None = None) -> None:
        raw_url = base_url or os.getenv("MCP_SERVER_URL", DEFAULT_MCP_SERVER_URL)
        self.base_url: str = raw_url.rstrip("/")
        self._base_url: str = self.base_url

        # Connection & Transport State
        self._transport_mode: Literal["mcp_sdk", "rest"] | None = None
        self._session: ClientSession | None = None
        self._runner_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None
        self._connected_event: asyncio.Event | None = None
        self._init_error: Exception | None = None

        # REST Client
        self._http_client: httpx.AsyncClient | None = None

        # Concurrency & Cache
        self._lock: asyncio.Lock = asyncio.Lock()
        self._cached_tools: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def transport_mode(self) -> str | None:
        """Currently active transport mode ('mcp_sdk', 'rest', or None)."""
        return self._transport_mode

    @property
    def is_connected(self) -> bool:
        """Check whether the client is currently connected or ready for REST calls."""
        if self._transport_mode == "mcp_sdk":
            return (
                self._session is not None
                and self._runner_task is not None
                and not self._runner_task.done()
            )
        elif self._transport_mode == "rest":
            return self._http_client is not None and not self._http_client.is_closed
        return False

    # ------------------------------------------------------------------
    # Async Context Manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Lifecycle: connect, close, reconnect
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Establish connection to the MCP server.

        Attempts to connect via official MCP SDK client session over SSE transport.
        If that fails, gracefully falls back to a REST-based approach using httpx.
        """
        async with self._lock:
            await self._establish_connection()

    async def _establish_connection(self) -> None:
        """Internal connection initialization logic."""
        # Cleanup any existing connection state
        await self._cleanup()

        self._transport_mode = None
        self._init_error = None
        self._connected_event = asyncio.Event()
        self._stop_event = asyncio.Event()

        # 1. Attempt official MCP Python SDK with SSE transport
        if HAS_MCP_SDK:
            sse_url = self.base_url if self.base_url.endswith("/sse") else f"{self.base_url}/sse"
            logger.info("[MCP] Attempting connection via MCP SDK (SSE) at: {}", sse_url)

            self._runner_task = asyncio.create_task(
                self._run_mcp_sse_session(sse_url),
                name="mcp_sdk_sse_runner",
            )

            try:
                # Wait up to 5 seconds for SSE handshake and initialization
                await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "[MCP] SSE handshake timed out (5s) for {}. Switching to REST fallback.",
                    sse_url,
                )
                if self._stop_event:
                    self._stop_event.set()

            if self._session is not None and self._transport_mode == "mcp_sdk":
                logger.success(
                    "[MCP] Connected to MCP server via official MCP SDK (SSE) at {}", sse_url
                )
                # Keep backup HTTP client initialized for hybrid requests
                self._http_client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=httpx.Timeout(SLICER_TIMEOUT),
                )
                return

            # If SDK connection failed, log rationale
            err_msg = str(self._init_error) if self._init_error else "handshake timeout"
            logger.warning(
                "[MCP] MCP SDK SSE connection failed ({}). Activating REST fallback transport.",
                err_msg,
            )

            # Ensure runner task is stopped cleanly
            if self._stop_event:
                self._stop_event.set()
            if self._runner_task and not self._runner_task.done():
                try:
                    await asyncio.wait_for(self._runner_task, timeout=1.0)
                except Exception:
                    self._runner_task.cancel()
            self._runner_task = None
            self._session = None
        else:
            logger.warning("[MCP] Official 'mcp' SDK is not installed. Defaulting to REST transport.")

        # 2. Fall back to REST transport (httpx)
        self._transport_mode = "rest"
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(SLICER_TIMEOUT),
        )
        logger.info("[MCP] REST transport initialized at: {}", self.base_url)

    async def _run_mcp_sse_session(self, sse_url: str) -> None:
        """Background coroutine managing the MCP SSE client and session lifecycle."""
        try:
            async with sse_client(
                sse_url,
                timeout=5.0,
                sse_read_timeout=300.0,
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    self._session = session
                    self._transport_mode = "mcp_sdk"
                    if self._connected_event:
                        self._connected_event.set()

                    logger.debug("[MCP SSE Runner] MCP ClientSession active, awaiting stop signal.")
                    if self._stop_event:
                        await self._stop_event.wait()
                    logger.debug("[MCP SSE Runner] Stop signal received, shutting down session.")
        except Exception as exc:
            self._init_error = exc
            if self._connected_event and not self._connected_event.is_set():
                self._connected_event.set()
            logger.debug("[MCP SSE Runner] Session ended with message: {}", exc)
        finally:
            self._session = None
            if self._transport_mode == "mcp_sdk":
                self._transport_mode = None

    async def close(self) -> None:
        """Cleanup all network connections, background tasks, and active sessions."""
        async with self._lock:
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Internal cleanup implementation."""
        # Stop MCP SDK background runner
        if self._stop_event and not self._stop_event.is_set():
            self._stop_event.set()

        if self._runner_task and not self._runner_task.done():
            try:
                await asyncio.wait_for(self._runner_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._runner_task.cancel()
            except Exception as e:
                logger.debug("[MCP] Error awaiting runner shutdown: {}", e)

        self._runner_task = None
        self._session = None

        # Close httpx REST client
        if self._http_client and not self._http_client.is_closed:
            try:
                await self._http_client.aclose()
            except Exception as e:
                logger.debug("[MCP] Error closing httpx client: {}", e)

        self._http_client = None
        self._transport_mode = None
        logger.info("[MCP] MCPClient closed cleanly")

    async def reconnect(self) -> None:
        """Explicitly re-establish connection to MCP server."""
        logger.info("[MCP] Initiating client reconnection...")
        async with self._lock:
            await self._establish_connection()

    # ------------------------------------------------------------------
    # Tool Discovery
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[dict[str, Any]]:
        """
        List available tools from the MCP server.

        Returns a list of tool specification dicts with 'name', 'description',
        and 'inputSchema'.
        """
        if not self.is_connected:
            await self.connect()

        # 1. Try MCP SDK session
        if self._transport_mode == "mcp_sdk" and self._session is not None:
            try:
                tools_resp = await asyncio.wait_for(self._session.list_tools(), timeout=10.0)
                tools: list[dict[str, Any]] = [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "inputSchema": t.inputSchema if hasattr(t, "inputSchema") else {},
                    }
                    for t in tools_resp.tools
                ]
                self._cached_tools = tools
                logger.info("[MCP] Discovered {} tools via MCP SDK", len(tools))
                return tools
            except Exception as exc:
                logger.warning("[MCP] Failed to list tools via MCP SDK ({}). Trying REST.", exc)

        # 2. Try REST discovery endpoints
        if self._http_client is not None and not self._http_client.is_closed:
            for endpoint in ("/tools", "/api/tools", "/list_tools"):
                try:
                    resp = await self._http_client.get(endpoint, timeout=5.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        tools_list = data if isinstance(data, list) else data.get("tools", [])
                        if isinstance(tools_list, list) and tools_list:
                            self._cached_tools = tools_list
                            logger.info("[MCP] Discovered {} tools via REST ({})", len(tools_list), endpoint)
                            return tools_list
                except Exception:
                    continue

        # 3. Return cached or predefined fallback tools
        if self._cached_tools:
            return self._cached_tools

        logger.info("[MCP] Using built-in tool definitions for Rosatom 5D Additive Cell")
        return self._get_default_tools_metadata()

    # ------------------------------------------------------------------
    # Tool Invocation
    # ------------------------------------------------------------------

    def get_default_timeout(self, tool_name: str) -> float:
        """Determine timeout based on tool workload (120s for slicer, 30s for others)."""
        name_lower = tool_name.lower()
        if any(keyword in name_lower for keyword in SLICER_TOOL_KEYWORDS):
            return SLICER_TIMEOUT
        return DEFAULT_TIMEOUT

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        Invoke an MCP tool and return result dictionary.

        Args:
            tool_name: Tool identifier (e.g. 'analyze_stl', 'generate_safe_gcode')
            arguments: Dictionary of arguments for the tool
            timeout: Optional custom timeout in seconds (defaults to 120s for slicer, 30s otherwise)

        Returns:
            Dictionary containing the tool output or error data.
        """
        if arguments is None:
            arguments = {}

        target_name, target_args = self._normalize_tool_call(tool_name, arguments)
        actual_timeout = timeout if timeout is not None else self.get_default_timeout(target_name)

        # Ensure active connection
        if not self.is_connected:
            logger.info("[MCP] Connection inactive. Reconnecting before calling '{}'...", target_name)
            try:
                await self.connect()
            except Exception as e:
                logger.error("[MCP] Auto-reconnect failed: {}", e)

        start_time = time.monotonic()
        logger.info(
            "[MCP] >>> Invoking '{}' (target='{}', transport={}, timeout={:.0f}s)",
            tool_name,
            target_name,
            self._transport_mode or "rest",
            actual_timeout,
        )

        try:
            result = await self._execute_call_with_fallback(
                target_name, target_args, actual_timeout
            )
            elapsed = time.monotonic() - start_time
            logger.info(
                "[MCP] <<< Tool '{}' completed successfully in {:.3f}s (transport={})",
                tool_name,
                elapsed,
                self._transport_mode or "rest",
            )
            return result

        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            ConnectionError,
            asyncio.TimeoutError,
            TimeoutError,
        ) as net_err:
            elapsed = time.monotonic() - start_time
            logger.warning(
                "[MCP] Network error on tool '{}' after {:.3f}s: {}. Reconnecting...",
                tool_name,
                elapsed,
                net_err,
            )

            # Reconnection logic: reconnect once and retry the call
            await self.reconnect()
            retry_start = time.monotonic()
            try:
                result = await self._execute_call_with_fallback(
                    target_name, target_args, actual_timeout
                )
                retry_elapsed = time.monotonic() - retry_start
                logger.info(
                    "[MCP] <<< Tool '{}' succeeded on retry in {:.3f}s",
                    tool_name,
                    retry_elapsed,
                )
                return result
            except Exception as retry_err:
                total_elapsed = time.monotonic() - start_time
                logger.error(
                    "[MCP] <<< Tool '{}' retry failed after {:.3f}s total: {}",
                    tool_name,
                    total_elapsed,
                    retry_err,
                )
                raise

        except Exception as exc:
            elapsed = time.monotonic() - start_time
            logger.error(
                "[MCP] <<< Tool '{}' execution failed after {:.3f}s: {}",
                tool_name,
                elapsed,
                exc,
            )
            raise

    async def call_tool_safe(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        Wraps call_tool with comprehensive error handling.
        Never raises exceptions; returns {"error": ...} on failure.
        """
        if arguments is None:
            arguments = {}

        try:
            result = await self.call_tool(tool_name, arguments, timeout=timeout)
            if isinstance(result, dict) and "error" in result and "success" not in result:
                result["success"] = False
            return result
        except Exception as exc:
            logger.warning("[MCP] call_tool_safe caught error for '{}': {}", tool_name, exc)
            err_dict: dict[str, Any] = {
                "error": str(exc),
                "success": False,
                "tool_name": tool_name,
            }
            # Include safety defaults for specific tools
            if tool_name == "render_stl_model":
                err_dict["render_paths"] = []
            return err_dict

    # ------------------------------------------------------------------
    # Internal Execution: MCP SDK vs REST Fallback
    # ------------------------------------------------------------------

    async def _execute_call_with_fallback(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        """Executes call via MCP SDK if available, falling back to REST on error."""
        # 1. Try MCP SDK over SSE if transport is active
        if self._transport_mode == "mcp_sdk" and self._session is not None:
            try:
                call_coro = self._session.call_tool(tool_name, arguments)
                raw_result = await asyncio.wait_for(call_coro, timeout=timeout)
                return self._parse_mcp_result(raw_result)
            except Exception as exc:
                logger.warning(
                    "[MCP SDK] Call tool '{}' failed ({}). Degrading to REST fallback.",
                    tool_name,
                    exc,
                )
                # If runner terminated, permanently shift transport mode to rest
                if self._runner_task and self._runner_task.done():
                    self._transport_mode = "rest"
                # Fall back to REST for this execution
                return await self._call_tool_rest(tool_name, arguments, timeout=timeout)

        # 2. REST transport (httpx)
        return await self._call_tool_rest(tool_name, arguments, timeout=timeout)

    async def _call_tool_rest(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        """
        Invoke MCP tool via REST POST /call_tool endpoint.

        Payload: {"tool_name": ..., "arguments": ...}
        """
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(timeout),
            )

        endpoint = f"{self.base_url}/call_tool"
        payload = {
            "tool_name": tool_name,
            "arguments": arguments,
        }

        try:
            response = await self._http_client.post(
                endpoint,
                json=payload,
                timeout=timeout,
            )
        except httpx.TimeoutException as te:
            raise TimeoutError(f"REST call to '{tool_name}' timed out after {timeout:.1f}s") from te
        except httpx.RequestError as re:
            raise ConnectionError(f"REST connection to '{tool_name}' failed: {re}") from re

        if response.status_code >= 400:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            logger.error("[REST] Server error on '{}': {}", tool_name, error_msg[:250])
            return {
                "error": error_msg,
                "status_code": response.status_code,
                "success": False,
                "tool_name": tool_name,
            }

        # Parse JSON body
        try:
            data = response.json()
        except Exception:
            return {"raw_response": response.text, "success": True}

        # Handle string responses containing JSON
        if isinstance(data, str):
            try:
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    return parsed
                return {"result": parsed, "success": True}
            except Exception:
                return {"result": data, "success": True}

        if isinstance(data, dict):
            return data

        return {"result": data, "success": True}

    def _parse_mcp_result(self, result: Any) -> dict[str, Any]:
        """Convert MCP CallToolResult from Python SDK into a standard dictionary."""
        if result is None:
            return {"success": False, "error": "Empty response from MCP tool"}

        is_error = getattr(result, "isError", False)
        content_blocks = getattr(result, "content", [])

        if not content_blocks:
            if is_error:
                return {"success": False, "error": "MCP tool returned error with empty content"}
            return {"success": True}

        text_parts: list[str] = []
        other_blocks: list[dict[str, Any]] = []

        for block in content_blocks:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif hasattr(block, "type") and block.type == "text" and hasattr(block, "text"):
                text_parts.append(block.text)
            elif isinstance(block, str):
                text_parts.append(block)
            elif hasattr(block, "model_dump"):
                other_blocks.append(block.model_dump())
            elif isinstance(block, dict):
                other_blocks.append(block)
            else:
                text_parts.append(str(block))

        raw_text = "\n".join(text_parts).strip()
        parsed_json: Any = None

        if raw_text:
            try:
                parsed_json = json.loads(raw_text)
            except json.JSONDecodeError:
                parsed_json = None

        if isinstance(parsed_json, dict):
            if is_error and "error" not in parsed_json:
                parsed_json["error"] = raw_text
                parsed_json["success"] = False
            return parsed_json

        if is_error:
            return {
                "success": False,
                "error": raw_text or "MCP tool execution reported error",
                "content": other_blocks or raw_text,
            }

        if parsed_json is not None:
            return {"success": True, "result": parsed_json}

        if raw_text:
            return {"success": True, "text": raw_text}

        if other_blocks:
            return {"success": True, "content": other_blocks}

        return {"success": True}

    # ------------------------------------------------------------------
    # Aliasing & Argument Normalization
    # ------------------------------------------------------------------

    def _normalize_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """Normalize tool names and translate arguments for compatibility."""
        target_name = TOOL_ALIASES.get(tool_name, tool_name)
        target_args = dict(arguments)

        if tool_name == "analyze_stl_geometry" or target_name == "analyze_stl":
            if "file_path" in target_args and "stl_path" not in target_args:
                target_args["stl_path"] = target_args.pop("file_path")

        elif tool_name == "slice_model_to_gcode" or target_name == "generate_safe_gcode":
            if "mode" in target_args and "mode_5d" not in target_args:
                mode_str = str(target_args.pop("mode")).upper()
                target_args["mode_5d"] = ("5D" in mode_str)
            if "slicer_params" in target_args:
                sp = target_args.pop("slicer_params")
                if isinstance(sp, dict):
                    for k, v in sp.items():
                        if k not in target_args:
                            target_args[k] = v
            if "infill" in target_args and "infill_percent" not in target_args:
                target_args["infill_percent"] = target_args.pop("infill")

        elif tool_name == "dispatch_print_job" or target_name == "send_to_printer":
            # Arguments are already printer_id and gcode_path
            pass

        return target_name, target_args

    # ------------------------------------------------------------------
    # Built-in Tool Definitions (Fallback)
    # ------------------------------------------------------------------

    def _get_default_tools_metadata(self) -> list[dict[str, Any]]:
        """Default schemas for the 7 Rosatom Additive Manufacturing tools."""
        return [
            {
                "name": "analyze_stl",
                "description": "Perform deep geometric analysis of an STL/OBJ/PLY file (dimensions, watertightness, overhangs, topple risk).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "stl_path": {"type": "string", "description": "Path to the 3D model file"}
                    },
                    "required": ["stl_path"],
                },
            },
            {
                "name": "recommend_print_parameters",
                "description": "Recommend physical and slicing parameters based on mesh analysis, industrial filament profile, and target strength.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "stl_path": {"type": "string", "description": "Path to STL file"},
                        "material": {"type": "string", "description": "Filament material name", "default": "PLA"},
                        "strength_priority": {"type": "string", "description": "Priority mode: balanced, max_strength, rapid_prototype", "default": "balanced"},
                    },
                    "required": ["stl_path"],
                },
            },
            {
                "name": "generate_safe_gcode",
                "description": "Slice 3D model with security guard audit, supporting 3D standard or Stereotech 5D continuous multi-axis kinematics.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "stl_path": {"type": "string", "description": "Path to STL file"},
                        "material": {"type": "string", "description": "Filament material", "default": "PLA"},
                        "layer_height": {"type": "number", "description": "Layer height in mm", "default": 0.2},
                        "infill_percent": {"type": "integer", "description": "Infill percentage (0-100)", "default": 20},
                        "mode_5d": {"type": "boolean", "description": "Enable Stereotech 5-axis continuous printing mode", "default": False},
                        "output_gcode_path": {"type": "string", "description": "Optional destination path for generated G-code"},
                    },
                    "required": ["stl_path"],
                },
            },
            {
                "name": "verify_gcode_security",
                "description": "Line-by-line static security inspection of a G-code file before transmission (bounds, temperatures, prohibited M-codes, SHA-256).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "gcode_path": {"type": "string", "description": "Path to G-code file"}
                    },
                    "required": ["gcode_path"],
                },
            },
            {
                "name": "diagnose_print_defect_or_spool",
                "description": "Diagnose print defect or spool anomaly, suggest parameter correction and root cause analysis.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "defect_type": {"type": "string", "description": "Type of defect (e.g. warping, stringing, under_extrusion)"},
                        "observed_symptoms": {"type": "string", "description": "Detailed description of observed defect or telemetry"},
                        "current_material": {"type": "string", "description": "Currently loaded filament material (optional)"},
                    },
                    "required": ["defect_type", "observed_symptoms"],
                },
            },
            {
                "name": "get_fleet_status",
                "description": "Retrieve real-time status and telemetry of all 3D/5D additive manufacturing cells in the Rosatom cluster.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "send_to_printer",
                "description": "Transmit validated and cryptographically verified G-code to a physical/simulated printer in the fleet.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "printer_id": {"type": "string", "description": "Unique identifier of target printer"},
                        "gcode_path": {"type": "string", "description": "Path to verified G-code file"},
                    },
                    "required": ["printer_id", "gcode_path"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # Convenience Methods
    # ------------------------------------------------------------------

    async def analyze_stl(self, stl_path: str | Path) -> dict[str, Any]:
        """Perform deep geometric analysis of an STL model."""
        return await self.call_tool("analyze_stl", {"stl_path": str(stl_path)})

    async def recommend_print_parameters(
        self,
        stl_path: str | Path,
        material: str = "PLA",
        strength_priority: str = "balanced",
    ) -> dict[str, Any]:
        """Recommend physical and slicing parameters."""
        return await self.call_tool(
            "recommend_print_parameters",
            {
                "stl_path": str(stl_path),
                "material": material,
                "strength_priority": strength_priority,
            },
        )

    async def generate_safe_gcode(
        self,
        stl_path: str | Path,
        material: str = "PLA",
        layer_height: float = 0.2,
        infill_percent: int = 20,
        mode_5d: bool = False,
        output_gcode_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Generate verified G-code in 3D or 5D mode."""
        args: dict[str, Any] = {
            "stl_path": str(stl_path),
            "material": material,
            "layer_height": layer_height,
            "infill_percent": infill_percent,
            "mode_5d": mode_5d,
        }
        if output_gcode_path is not None:
            args["output_gcode_path"] = str(output_gcode_path)
        return await self.call_tool("generate_safe_gcode", args)

    async def verify_gcode_security(self, gcode_path: str | Path) -> dict[str, Any]:
        """Run industrial security audit on G-code file."""
        return await self.call_tool("verify_gcode_security", {"gcode_path": str(gcode_path)})

    async def diagnose_print_defect_or_spool(
        self,
        defect_type: str,
        observed_symptoms: str,
        current_material: str | None = None,
    ) -> dict[str, Any]:
        """Diagnose print defect or spool anomaly."""
        args: dict[str, Any] = {
            "defect_type": defect_type,
            "observed_symptoms": observed_symptoms,
        }
        if current_material:
            args["current_material"] = current_material
        return await self.call_tool("diagnose_print_defect_or_spool", args)

    async def get_fleet_status(self) -> dict[str, Any]:
        """Get status and telemetry of printer fleet."""
        return await self.call_tool("get_fleet_status", {})

    async def send_to_printer(
        self,
        printer_id: str,
        gcode_path: str | Path,
    ) -> dict[str, Any]:
        """Transmit verified G-code to target printer."""
        return await self.call_tool(
            "send_to_printer",
            {
                "printer_id": printer_id,
                "gcode_path": str(gcode_path),
            },
        )