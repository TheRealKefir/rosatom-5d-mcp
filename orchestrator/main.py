"""
Rosatom 5D MCP Orchestrator — FastAPI backend.

Provides REST + WebSocket endpoints for:
 • uploading STL / photos
 • streaming agent responses (tool calls + LLM reasoning)
 • printer fleet dashboard
 • 3D vs 5D comparison
 • closed-loop defect correction
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field
import uvicorn

from agent.router import AgentRouter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STORAGE = Path(os.getenv("STORAGE_DIR", str(Path(__file__).resolve().parent.parent / "storage")))
STL_DIR = STORAGE / "uploads_stl"
RENDERS_DIR = STORAGE / "renders_png"
GCODE_DIR = STORAGE / "generated_gcode"

for d in (STL_DIR, RENDERS_DIR, GCODE_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# WebSocket Connection Manager (for live tool-call streaming to frontend)
# ---------------------------------------------------------------------------

class ConnectionManager:
    """Manages active WebSocket connections per session."""

    def __init__(self):
        self.active: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, ws: WebSocket):
        await ws.accept()
        self.active[session_id] = ws
        logger.info("WebSocket connected: session_id={}", session_id)

    def disconnect(self, session_id: str):
        self.active.pop(session_id, None)
        logger.info("WebSocket disconnected: session_id={}", session_id)

    async def send_event(self, session_id: str, event: dict):
        ws = self.active.get(session_id)
        if ws:
            try:
                await ws.send_json(event)
            except Exception as e:
                logger.warning("Failed to send WS event: {}", e)


ws_manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Pydantic schemas for REST API
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="User text prompt")
    session_id: Optional[str] = None
    material: Optional[str] = None
    auto_send_to_printer: bool = False


class ChatResponse(BaseModel):
    session_id: str
    response: str
    preprocess_events: list[dict] = []
    elapsed_seconds: float = 0.0
    stl_url: Optional[str] = None
    gcode_url: Optional[str] = None


class CompareRequest(BaseModel):
    stl_path: str
    slicer_params_3d: dict = {}
    slicer_params_5d: dict = {}


class DefectCorrectionRequest(BaseModel):
    photo_path: str
    stl_path: str
    current_params: dict
    session_id: Optional[str] = None


class DispatchRequest(BaseModel):
    printer_id: str
    gcode_path: str


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

_agent: Optional[AgentRouter] = None


async def _get_agent() -> AgentRouter:
    global _agent
    if _agent is None:
        _agent = AgentRouter()
        await _agent.startup()
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Rosatom 5D Orchestrator...")
    agent = await _get_agent()
    yield
    logger.info("🛑 Shutting down Rosatom 5D Orchestrator...")
    await agent.shutdown()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Rosatom 3D/5D Print MCP Orchestrator",
    description=(
        "API оркестратора для автоматического подбора параметров FDM 3D/5D-печати, "
        "генерации G-кода и управления парком принтеров через MCP-протокол"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve storage files for frontend (STL viewer, G-code previews)
if STORAGE.exists():
    app.mount("/storage", StaticFiles(directory=str(STORAGE)), name="storage")


# ---------------------------------------------------------------------------
# REST: Upload + Chat (main endpoint)
# ---------------------------------------------------------------------------

@app.post("/api/upload", response_model=ChatResponse)
async def upload_and_chat(
    stl: Optional[UploadFile] = File(None),
    photo: Optional[UploadFile] = File(None),
    prompt: str = Form(""),
    session_id: Optional[str] = Form(None),
    material: Optional[str] = Form(None),
    auto_send_to_printer: bool = Form(False),
):
    """
    Accept STL model, photo and/or text prompt.
    Runs the full agent pipeline: analysis → reasoning → slicing → response.
    """
    sid = session_id or uuid.uuid4().hex[:12]
    paths: dict[str, str] = {}
    stl_url: Optional[str] = None
    gcode_url: Optional[str] = None

    # Save uploaded files
    if stl:
        if not stl.filename.lower().endswith(".stl"):
            raise HTTPException(status_code=400, detail="Файл должен иметь расширение .stl")
        dest = STL_DIR / f"{sid}_{stl.filename}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(stl.file, f)
        paths["stl"] = str(dest)
        stl_url = f"/storage/uploads_stl/{dest.name}"

    if photo:
        dest = RENDERS_DIR / f"{sid}_{photo.filename}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(photo.file, f)
        paths["photo"] = str(dest)

    # Enrich prompt with material preference if specified
    full_prompt = prompt
    if material:
        full_prompt = f"{prompt}\n\nПользователь указал предпочтительный материал: {material}"
    if auto_send_to_printer:
        full_prompt += "\n\nПосле генерации G-кода автоматически отправь на подходящий принтер."

    agent = await _get_agent()
    result = await agent.run(prompt=full_prompt, files=paths, session_id=sid)

    # Extract gcode URL from response if available
    response_text = result.get("response", "")
    for evt in result.get("preprocess_events", []):
        if evt.get("tool") == "slice_model_to_gcode" and "result" in evt:
            gpath = evt["result"].get("gcode_path", "")
            if gpath:
                gcode_url = f"/storage/generated_gcode/{Path(gpath).name}"

    return ChatResponse(
        session_id=sid,
        response=response_text,
        preprocess_events=result.get("preprocess_events", []),
        elapsed_seconds=result.get("elapsed_seconds", 0),
        stl_url=stl_url,
        gcode_url=gcode_url,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_text_only(request: ChatRequest):
    """Text-only chat endpoint (no file uploads)."""
    sid = request.session_id or uuid.uuid4().hex[:12]
    prompt = request.prompt
    if request.material:
        prompt += f"\n\nМатериал: {request.material}"

    agent = await _get_agent()
    result = await agent.run(prompt=prompt, files={}, session_id=sid)

    return ChatResponse(
        session_id=sid,
        response=result.get("response", ""),
        preprocess_events=result.get("preprocess_events", []),
        elapsed_seconds=result.get("elapsed_seconds", 0),
    )


# ---------------------------------------------------------------------------
# REST: G-code download
# ---------------------------------------------------------------------------

@app.get("/api/gcode/{filename}")
async def download_gcode(filename: str):
    """Download generated G-code file."""
    path = GCODE_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="G-code file not found")
    return FileResponse(path, media_type="application/octet-stream", filename=filename)


# ---------------------------------------------------------------------------
# REST: Fleet status
# ---------------------------------------------------------------------------

@app.get("/api/fleet")
async def fleet_status():
    """Get status of all printers in the fleet."""
    agent = await _get_agent()
    return await agent.get_fleet_status()


@app.post("/api/fleet/dispatch")
async def dispatch_job(request: DispatchRequest):
    """Send G-code to a specific printer."""
    agent = await _get_agent()
    result = await agent.mcp.call_tool_safe("dispatch_print_job", {
        "printer_id": request.printer_id,
        "gcode_path": request.gcode_path,
    })
    return result


# ---------------------------------------------------------------------------
# REST: 3D vs 5D Comparison (killer feature)
# ---------------------------------------------------------------------------

@app.post("/api/compare")
async def compare_3d_vs_5d(request: CompareRequest):
    """
    Run slicing in both 3D and 5D modes and return a comparison card
    with material savings, time difference, and strength advantage.
    """
    agent = await _get_agent()
    return await agent.compare_3d_vs_5d(
        stl_path=request.stl_path,
        slicer_params_3d=request.slicer_params_3d,
        slicer_params_5d=request.slicer_params_5d,
    )


# ---------------------------------------------------------------------------
# REST: Defect Correction (killer feature)
# ---------------------------------------------------------------------------

@app.post("/api/defect/correct")
async def correct_defect(request: DefectCorrectionRequest):
    """
    Closed-loop defect correction:
    analyze photo → identify defect → adjust params → re-slice → find printer.
    """
    agent = await _get_agent()
    return await agent.correct_defect(
        photo_path=request.photo_path,
        stl_path=request.stl_path,
        current_params=request.current_params,
        session_id=request.session_id or uuid.uuid4().hex[:12],
    )


# ---------------------------------------------------------------------------
# WebSocket: Streaming chat with live tool-call events
# ---------------------------------------------------------------------------

@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    """
    Streaming chat via WebSocket.
    
    Client sends: {"prompt": "...", "stl_path": "...", "photo_path": "...", "session_id": "..."}
    Server yields: {"type": "status|tool_call|tool_result|chunk|done|error", ...}
    """
    await ws.accept()
    agent = await _get_agent()

    try:
        while True:
            data = await ws.receive_json()
            session_id = data.get("session_id", uuid.uuid4().hex[:12])

            files = {}
            if "stl_path" in data:
                files["stl"] = data["stl_path"]
            if "photo_path" in data:
                files["photo"] = data["photo_path"]

            async for event in agent.run_stream(
                prompt=data.get("prompt", ""),
                files=files,
                session_id=session_id,
            ):
                await ws.send_json(event)

    except WebSocketDisconnect:
        logger.info("Chat WebSocket disconnected")
    except Exception as e:
        logger.error("WebSocket error: {}", e)


@app.websocket("/ws/{session_id}")
async def ws_session(ws: WebSocket, session_id: str):
    """
    Per-session WebSocket for live telemetry and MCP tool-call tracing.
    Used by the frontend FleetDashboard and MCP trace panel.
    """
    await ws_manager.connect(session_id, ws)
    try:
        while True:
            # Keep alive — client can send pings
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(session_id)


# ---------------------------------------------------------------------------
# REST: Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "rosatom-5d-orchestrator"}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)