"""
AgentRouter — ReAct‑style reasoning loop that orchestrates MCP tool calls.

Full pipeline:
 1. Receive user prompt + optional STL + optional photo
 2. If STL → call analyze_stl_geometry + render_stl_model
 3. If photo or renders → call VLM inspector
 4. RAG retrieval for material/defect knowledge
 5. LLM reasons and calls MCP tools (recommend_print_profile, slice_model, etc.)
 6. Stream results back to the user
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI
from loguru import logger

from .mcp_client import MCPClient
from .vlm_inspector import VLMInspector
from .prompts import MAIN_SYSTEM, REPORT_SYSTEM, SAFETY_SYSTEM, load_prompt
from .prompts_register import MCP_TOOLS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))

MAX_TOOL_ROUNDS = 12          # safety: max number of LLM↔tool iterations
TOOL_CALL_TIMEOUT = 180.0     # seconds per MCP tool call


# ---------------------------------------------------------------------------
# Helper: build system prompt with safety + RAG context
# ---------------------------------------------------------------------------

def _build_system_prompt(rag_context: str = "") -> str:
    """Assemble the full system prompt from components."""
    parts = [
        MAIN_SYSTEM,
        "\n\n---\n\n",
        SAFETY_SYSTEM,
    ]
    if rag_context:
        parts.extend([
            "\n\n---\n\n## Справочные данные из базы знаний (RAG):\n\n",
            rag_context,
        ])
    return "".join(parts)


# ---------------------------------------------------------------------------
# Helper: convert MCP tool result to OpenAI message format
# ---------------------------------------------------------------------------

def _tool_result_message(tool_call_id: str, result: Any) -> dict:
    """Format a tool call result as an OpenAI tool response message."""
    if isinstance(result, dict):
        content = json.dumps(result, ensure_ascii=False, indent=2)
    elif isinstance(result, list):
        content = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        content = str(result)
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }


# ---------------------------------------------------------------------------
# AgentRouter
# ---------------------------------------------------------------------------

class AgentRouter:
    """Orchestrates the full agent pipeline: analysis → reasoning → tools → response."""

    def __init__(self):
        self.mcp = MCPClient()
        self.vlm = VLMInspector()
        self._llm: Optional[AsyncOpenAI] = None
        self._rag = None  # RAGRetriever, initialized lazily

        # Session state: conversation history per session_id
        self._sessions: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self):
        """Initialize connections to MCP server, LLM, and RAG."""
        await self.mcp.connect()

        self._llm = AsyncOpenAI(
            api_key=LLM_API_KEY or "sk-stub",
            base_url=LLM_BASE_URL,
        )

        # Lazy RAG init — don't fail if Qdrant is down
        try:
            from ..rag.retriever import RAGRetriever
            qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
            self._rag = RAGRetriever(qdrant_url=qdrant_url)
            logger.info("RAG retriever initialized (Qdrant: {})", qdrant_url)
        except Exception as e:
            logger.warning("RAG retriever unavailable: {} — proceeding without RAG", e)
            self._rag = None

        logger.info("AgentRouter started (LLM: {}, MCP: {})", LLM_MODEL, self.mcp._base_url)

    async def shutdown(self):
        await self.mcp.close()

    # ------------------------------------------------------------------
    # Pre-processing: auto-analyze STL and photos before LLM
    # ------------------------------------------------------------------

    async def _preprocess(
        self,
        prompt: str,
        files: dict[str, str],
        session_id: str,
    ) -> tuple[str, list[dict]]:
        """
        Run automatic analysis steps and build the enriched user message.

        Returns:
            - enriched_prompt: user prompt augmented with analysis results
            - events: list of event dicts for streaming
        """
        enriched_parts: list[str] = []
        events: list[dict] = []
        geometry_stats: dict | None = None
        render_paths: list[str] = []
        vlm_result: dict | None = None

        # --- Step 1: STL Geometry Analysis ---
        stl_path = files.get("stl") or files.get("stl_path")
        if stl_path and Path(stl_path).exists():
            events.append({
                "type": "tool_call",
                "tool": "analyze_stl_geometry",
                "status": "running",
            })
            try:
                geometry_stats = await self.mcp.call_tool_safe(
                    "analyze_stl_geometry", {"file_path": stl_path}
                )
                events.append({
                    "type": "tool_result",
                    "tool": "analyze_stl_geometry",
                    "result": geometry_stats,
                })
                enriched_parts.append(
                    f"### Результаты геометрического анализа STL:\n"
                    f"```json\n{json.dumps(geometry_stats, ensure_ascii=False, indent=2)}\n```"
                )
            except Exception as e:
                logger.error("STL analysis failed: {}", e)
                events.append({"type": "error", "tool": "analyze_stl_geometry", "content": str(e)})

            # --- Step 1b: Render STL for VLM ---
            try:
                events.append({
                    "type": "tool_call",
                    "tool": "render_stl_model",
                    "status": "running",
                })
                render_result = await self.mcp.call_tool_safe(
                    "render_stl_model", {"file_path": stl_path}
                )
                render_paths = render_result.get("render_paths", [])
                events.append({
                    "type": "tool_result",
                    "tool": "render_stl_model",
                    "result": {"render_count": len(render_paths)},
                })
            except Exception as e:
                logger.warning("STL rendering failed (non-critical): {}", e)

        # --- Step 2: Photo / VLM Analysis ---
        photo_path = files.get("photo") or files.get("photo_path")
        if photo_path or render_paths:
            events.append({
                "type": "tool_call",
                "tool": "analyze_visual_data",
                "status": "running",
            })
            try:
                vlm_result = await self.vlm.analyze_photo(
                    photo_path=photo_path or "",
                    context_prompt=prompt,
                    render_paths=render_paths,
                )
                events.append({
                    "type": "tool_result",
                    "tool": "analyze_visual_data",
                    "result": vlm_result,
                })
                enriched_parts.append(
                    f"### Результаты визуального анализа (VLM):\n"
                    f"```json\n{json.dumps(vlm_result, ensure_ascii=False, indent=2)}\n```"
                )
            except Exception as e:
                logger.error("VLM analysis failed: {}", e)
                events.append({"type": "error", "tool": "analyze_visual_data", "content": str(e)})

        # --- Step 3: RAG Context Retrieval ---
        rag_context = ""
        if self._rag:
            try:
                rag_results = await self._rag.search(prompt, top_k=5)
                if rag_results:
                    rag_context = self._rag.format_context(rag_results)
                    enriched_parts.append(
                        f"### Справочные данные (RAG):\n{rag_context}"
                    )
                    events.append({
                        "type": "rag",
                        "status": "retrieved",
                        "count": len(rag_results),
                    })
            except Exception as e:
                logger.warning("RAG retrieval failed: {}", e)

        # Combine everything
        if enriched_parts:
            enriched_prompt = (
                f"{prompt}\n\n"
                f"---\n\n"
                f"**Автоматический предварительный анализ:**\n\n"
                + "\n\n".join(enriched_parts)
            )
        else:
            enriched_prompt = prompt

        # Inject file paths into context for the LLM so it can reference them in tool calls
        if stl_path:
            enriched_prompt += f"\n\n[STL файл: `{stl_path}`]"
        if photo_path:
            enriched_prompt += f"\n\n[Фото: `{photo_path}`]"

        return enriched_prompt, events

    # ------------------------------------------------------------------
    # Core: ReAct Reasoning Loop (tool-calling)
    # ------------------------------------------------------------------

    async def _execute_tool_call(self, function_name: str, arguments: dict) -> dict:
        """Execute a single tool call via MCP (or VLM for visual analysis)."""
        t0 = time.monotonic()

        if function_name == "analyze_visual_data":
            # Route visual analysis to local VLM inspector
            result = await self.vlm.analyze_photo(
                photo_path=arguments.get("photo_path", ""),
                context_prompt=arguments.get("context", ""),
                render_paths=arguments.get("render_paths", []),
            )
        else:
            # Everything else goes to MCP server
            result = await self.mcp.call_tool_safe(function_name, arguments)

        elapsed = time.monotonic() - t0
        logger.info("Tool {} executed in {:.2f}s", function_name, elapsed)
        return result

    async def _react_loop(
        self,
        messages: list[dict],
        on_event: Any = None,
    ) -> str:
        """
        Run the ReAct loop:
        1. Send messages to LLM with tool definitions
        2. If LLM returns tool_calls → execute them → append results → repeat
        3. If LLM returns final text → return it
        """
        for round_idx in range(MAX_TOOL_ROUNDS):
            logger.debug("ReAct round {}/{}", round_idx + 1, MAX_TOOL_ROUNDS)

            try:
                response = await self._llm.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    tools=MCP_TOOLS,
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=4096,
                )
            except Exception as e:
                logger.error("LLM call failed: {}", e)
                return f"❌ Ошибка обращения к LLM: {e}"

            choice = response.choices[0]
            assistant_msg = choice.message

            # Append assistant message to history
            msg_dict: dict[str, Any] = {"role": "assistant"}
            if assistant_msg.content:
                msg_dict["content"] = assistant_msg.content
            if assistant_msg.tool_calls:
                msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_msg.tool_calls
                ]
            messages.append(msg_dict)

            # If no tool calls → we have the final answer
            if not assistant_msg.tool_calls:
                return assistant_msg.content or ""

            # Execute all tool calls in parallel
            tool_tasks = []
            for tc in assistant_msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                logger.info("LLM wants to call tool: {}({})", fn_name, fn_args)

                if on_event:
                    await on_event({
                        "type": "tool_call",
                        "tool": fn_name,
                        "arguments": fn_args,
                        "status": "running",
                    })

                tool_tasks.append((tc.id, fn_name, fn_args))

            # Run tool calls concurrently
            async def _run_one(call_id: str, name: str, args: dict):
                try:
                    result = await asyncio.wait_for(
                        self._execute_tool_call(name, args),
                        timeout=TOOL_CALL_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    result = {"error": f"Timeout: tool {name} exceeded {TOOL_CALL_TIMEOUT}s"}
                except Exception as e:
                    result = {"error": f"Tool {name} failed: {e}"}

                if on_event:
                    await on_event({
                        "type": "tool_result",
                        "tool": name,
                        "result": result,
                    })
                return call_id, result

            results = await asyncio.gather(
                *[_run_one(cid, name, args) for cid, name, args in tool_tasks]
            )

            # Append tool results to messages
            for call_id, result in results:
                messages.append(_tool_result_message(call_id, result))

        # Exceeded max rounds
        return "⚠️ Достигнут лимит итераций агента. Промежуточные результаты выше."

    # ------------------------------------------------------------------
    # Public API: synchronous run
    # ------------------------------------------------------------------

    async def run(
        self,
        prompt: str,
        files: dict[str, str],
        session_id: str,
    ) -> dict:
        """
        Non-streaming agent execution.
        Returns a dict with the final response and metadata.
        """
        t0 = time.monotonic()
        preprocess_events: list[dict] = []

        # Pre-processing: auto-analyze STL, photos, retrieve RAG
        enriched_prompt, preprocess_events = await self._preprocess(
            prompt, files, session_id
        )

        # Build conversation messages
        session_messages = self._sessions.get(session_id, [])
        if not session_messages:
            # New session — inject system prompt with RAG context
            rag_ctx = ""
            if self._rag:
                try:
                    rag_results = await self._rag.search(prompt, top_k=3)
                    if rag_results:
                        rag_ctx = self._rag.format_context(rag_results)
                except Exception:
                    pass
            session_messages = [
                {"role": "system", "content": _build_system_prompt(rag_ctx)},
            ]

        session_messages.append({"role": "user", "content": enriched_prompt})

        # Run ReAct loop
        final_answer = await self._react_loop(session_messages)

        # Store session
        session_messages.append({"role": "assistant", "content": final_answer})
        self._sessions[session_id] = session_messages

        elapsed = time.monotonic() - t0
        logger.info("Session {} completed in {:.1f}s", session_id, elapsed)

        return {
            "session_id": session_id,
            "response": final_answer,
            "preprocess_events": preprocess_events,
            "elapsed_seconds": round(elapsed, 2),
        }

    # ------------------------------------------------------------------
    # Public API: streaming run
    # ------------------------------------------------------------------

    async def run_stream(
        self,
        prompt: str,
        files: dict[str, str],
        session_id: str,
    ) -> AsyncIterator[dict]:
        """
        Streaming agent execution.
        Yields partial events:
         - {"type": "status", "content": "..."} — progress updates
         - {"type": "tool_call", "tool": "...", ...} — MCP tool invocation
         - {"type": "tool_result", "tool": "...", "result": {...}} — tool result
         - {"type": "rag", ...} — RAG retrieval info
         - {"type": "chunk", "content": "..."} — streamed LLM response tokens
         - {"type": "done", "content": "..."} — final assembled answer
         - {"type": "error", "content": "..."} — error
        """
        t0 = time.monotonic()
        collected_events: list[dict] = []

        yield {"type": "status", "content": "🔍 Запуск предварительного анализа..."}

        # Pre-processing
        try:
            enriched_prompt, preprocess_events = await self._preprocess(
                prompt, files, session_id
            )
            for evt in preprocess_events:
                yield evt
                collected_events.append(evt)
        except Exception as e:
            yield {"type": "error", "content": f"Ошибка предобработки: {e}"}
            return

        yield {"type": "status", "content": "🤖 Запуск агента рассуждения..."}

        # Build conversation
        session_messages = self._sessions.get(session_id, [])
        if not session_messages:
            rag_ctx = ""
            if self._rag:
                try:
                    rag_results = await self._rag.search(prompt, top_k=3)
                    if rag_results:
                        rag_ctx = self._rag.format_context(rag_results)
                except Exception:
                    pass
            session_messages = [
                {"role": "system", "content": _build_system_prompt(rag_ctx)},
            ]

        session_messages.append({"role": "user", "content": enriched_prompt})

        # Streaming ReAct loop
        event_queue: asyncio.Queue[dict] = asyncio.Queue()

        async def _on_event(evt: dict):
            await event_queue.put(evt)

        # Run ReAct loop in background task
        react_task = asyncio.create_task(
            self._react_loop_streaming(session_messages, event_queue)
        )

        # Yield events as they arrive
        try:
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                    yield event
                    if event.get("type") == "done" or event.get("type") == "error":
                        break
                except asyncio.TimeoutError:
                    if react_task.done():
                        # Drain remaining events
                        while not event_queue.empty():
                            yield event_queue.get_nowait()
                        break
        except Exception as e:
            yield {"type": "error", "content": f"Streaming error: {e}"}
        finally:
            if not react_task.done():
                react_task.cancel()

        elapsed = time.monotonic() - t0
        yield {"type": "meta", "elapsed_seconds": round(elapsed, 2)}

    async def _react_loop_streaming(
        self,
        messages: list[dict],
        event_queue: asyncio.Queue,
    ):
        """ReAct loop that pushes events to a queue for streaming."""
        try:
            for round_idx in range(MAX_TOOL_ROUNDS):
                await event_queue.put({
                    "type": "status",
                    "content": f"💭 Раунд рассуждения {round_idx + 1}...",
                })

                # --- Streaming LLM call ---
                try:
                    stream = await self._llm.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        tools=MCP_TOOLS,
                        tool_choice="auto",
                        temperature=0.3,
                        max_tokens=4096,
                        stream=True,
                    )
                except Exception as e:
                    await event_queue.put({
                        "type": "error",
                        "content": f"LLM error: {e}",
                    })
                    return

                # Accumulate streamed response
                content_chunks: list[str] = []
                tool_calls_acc: dict[int, dict] = {}  # index -> accumulated tool call

                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue

                    # Stream text content
                    if delta.content:
                        content_chunks.append(delta.content)
                        await event_queue.put({
                            "type": "chunk",
                            "content": delta.content,
                        })

                    # Accumulate tool calls from deltas
                    if delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index
                            if idx not in tool_calls_acc:
                                tool_calls_acc[idx] = {
                                    "id": tc_delta.id or "",
                                    "name": "",
                                    "arguments": "",
                                }
                            if tc_delta.id:
                                tool_calls_acc[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_calls_acc[idx]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    tool_calls_acc[idx]["arguments"] += tc_delta.function.arguments

                full_content = "".join(content_chunks) if content_chunks else None
                has_tool_calls = len(tool_calls_acc) > 0

                # Build assistant message for history
                msg_dict: dict[str, Any] = {"role": "assistant"}
                if full_content:
                    msg_dict["content"] = full_content
                if has_tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"],
                            },
                        }
                        for tc in tool_calls_acc.values()
                    ]
                messages.append(msg_dict)

                # No tool calls → final answer
                if not has_tool_calls:
                    final = full_content or ""
                    await event_queue.put({"type": "done", "content": final})
                    return

                # Execute tool calls
                for idx, tc in sorted(tool_calls_acc.items()):
                    fn_name = tc["name"]
                    try:
                        fn_args = json.loads(tc["arguments"])
                    except json.JSONDecodeError:
                        fn_args = {}

                    await event_queue.put({
                        "type": "tool_call",
                        "tool": fn_name,
                        "arguments": fn_args,
                        "status": "running",
                    })

                    try:
                        result = await asyncio.wait_for(
                            self._execute_tool_call(fn_name, fn_args),
                            timeout=TOOL_CALL_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        result = {"error": f"Timeout: {fn_name}"}
                    except Exception as e:
                        result = {"error": f"{fn_name} failed: {e}"}

                    await event_queue.put({
                        "type": "tool_result",
                        "tool": fn_name,
                        "result": result,
                    })

                    messages.append(_tool_result_message(tc["id"], result))

            # Max rounds reached
            await event_queue.put({
                "type": "done",
                "content": "⚠️ Достигнут лимит итераций. Промежуточные результаты отображены выше.",
            })

        except Exception as e:
            logger.error("ReAct loop error: {}\n{}", e, traceback.format_exc())
            await event_queue.put({
                "type": "error",
                "content": f"Internal error: {e}",
            })

    # ------------------------------------------------------------------
    # Public API: fleet status (convenience)
    # ------------------------------------------------------------------

    async def get_fleet_status(self) -> dict:
        """Get printer fleet status via MCP."""
        return await self.mcp.call_tool_safe("get_printer_fleet_status", {})

    # ------------------------------------------------------------------
    # Public API: compare 3D vs 5D (killer feature)
    # ------------------------------------------------------------------

    async def compare_3d_vs_5d(
        self,
        stl_path: str,
        slicer_params_3d: dict,
        slicer_params_5d: dict,
    ) -> dict:
        """
        Run slicing in both 3D and 5D modes and return a comparison card.
        This is the killer feature for the hackathon presentation.
        """
        results = await asyncio.gather(
            self.mcp.call_tool_safe("slice_model_to_gcode", {
                "stl_path": stl_path,
                "mode": "3D",
                "slicer_params": slicer_params_3d,
            }),
            self.mcp.call_tool_safe("slice_model_to_gcode", {
                "stl_path": stl_path,
                "mode": "5D",
                "slicer_params": slicer_params_5d,
            }),
        )

        result_3d, result_5d = results

        # Calculate savings
        time_3d = result_3d.get("print_time_minutes", 0)
        time_5d = result_5d.get("print_time_minutes", 0)
        mass_3d = result_3d.get("filament_used_grams", 0)
        mass_5d = result_5d.get("filament_used_grams", 0)

        time_savings_pct = round(
            ((time_3d - time_5d) / time_3d * 100) if time_3d > 0 else 0, 1
        )
        material_savings_pct = round(
            ((mass_3d - mass_5d) / mass_3d * 100) if mass_3d > 0 else 0, 1
        )

        comparison = {
            "mode_3d": {
                "print_time_minutes": time_3d,
                "filament_grams": mass_3d,
                "supports_needed": slicer_params_3d.get("support_material", True),
                "gcode_path": result_3d.get("gcode_path"),
                "layer_count": result_3d.get("layer_count", 0),
            },
            "mode_5d": {
                "print_time_minutes": time_5d,
                "filament_grams": mass_5d,
                "supports_needed": False,  # 5D avoids supports via tilting
                "gcode_path": result_5d.get("gcode_path"),
                "layer_count": result_5d.get("layer_count", 0),
                "has_5d_commands": result_5d.get("has_5d_commands", False),
            },
            "savings": {
                "time_savings_percent": time_savings_pct,
                "material_savings_percent": material_savings_pct,
                "material_saved_grams": round(mass_3d - mass_5d, 1),
            },
            "recommendation": (
                "5D_MULTIAXIS" if material_savings_pct > 10 or time_savings_pct > 10
                else "3D_PLANAR"
            ),
        }

        return comparison

    # ------------------------------------------------------------------
    # Public API: closed-loop defect correction (killer feature)
    # ------------------------------------------------------------------

    async def correct_defect(
        self,
        photo_path: str,
        stl_path: str,
        current_params: dict,
        session_id: str,
    ) -> dict:
        """
        Closed-loop defect correction:
        1. Analyze photo of failed print via VLM
        2. Identify defect and adjust parameters
        3. Re-slice with corrected parameters
        4. Find available printer and offer to re-print
        """
        # 1. Analyze the defect photo
        vlm_result = await self.vlm.analyze_photo(
            photo_path=photo_path,
            context_prompt=(
                "Это фото неудачной 3D-печати. Определи дефект и предложи "
                "корректировку параметров печати."
            ),
        )

        # 2. Apply corrections to parameters
        corrected_params = dict(current_params)
        temp_adj = vlm_result.get("temperature_adjustments", {})

        if temp_adj.get("nozzle_delta"):
            corrected_params["nozzle_temperature"] = (
                corrected_params.get("nozzle_temperature", 210) + temp_adj["nozzle_delta"]
            )
        if temp_adj.get("bed_delta"):
            corrected_params["bed_temperature"] = (
                corrected_params.get("bed_temperature", 60) + temp_adj["bed_delta"]
            )
        if temp_adj.get("fan_speed") is not None:
            corrected_params["fan_speed"] = temp_adj["fan_speed"]

        # Apply VLM suggestions
        if vlm_result.get("infill_suggestion"):
            corrected_params["infill_density"] = vlm_result["infill_suggestion"]

        # Detect warping → add brim
        defects = vlm_result.get("detected_defects", [])
        if "warping" in defects:
            corrected_params["brim_width"] = max(
                corrected_params.get("brim_width", 0), 10.0
            )
            corrected_params["fan_speed"] = 0

        # Detect stringing → increase retraction
        if "stringing" in defects:
            corrected_params["retraction_distance"] = max(
                corrected_params.get("retraction_distance", 0.8), 1.5
            )
            corrected_params["retraction_speed"] = max(
                corrected_params.get("retraction_speed", 45), 60
            )

        # 3. Re-slice with corrected params
        mode = corrected_params.pop("mode", "3D")
        slice_result = await self.mcp.call_tool_safe("slice_model_to_gcode", {
            "stl_path": stl_path,
            "mode": mode,
            "slicer_params": corrected_params,
        })

        # 4. Find available printer
        fleet = await self.mcp.call_tool_safe("get_printer_fleet_status", {})
        available = [
            p for p in (fleet if isinstance(fleet, list) else [])
            if p.get("status") == "IDLE"
        ]

        return {
            "detected_defects": defects,
            "vlm_analysis": vlm_result,
            "original_params": current_params,
            "corrected_params": corrected_params,
            "corrections_applied": {
                k: corrected_params[k]
                for k in corrected_params
                if corrected_params[k] != current_params.get(k)
            },
            "slice_result": slice_result,
            "available_printers": available,
        }