import inspect
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

# Добавляем корень монорепозитория в sys.path для прямого импорта mcp_server
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

STORAGE_DIR = BASE_DIR / "storage"
STL_DIR = STORAGE_DIR / "uploads_stl"
RENDERS_DIR = STORAGE_DIR / "renders_png"
GCODE_DIR = STORAGE_DIR / "generated_gcode"

for folder in (STL_DIR, RENDERS_DIR, GCODE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# Пробуем импортировать модуль mcp_server.server
try:
    from mcp_server import server as mcp_srv
    logger.success("Модуль mcp_server.server успешно подключен к Orchestrator!")
except Exception as e:
    mcp_srv = None
    logger.warning(f"mcp_server.server пока не доступен ({e}). Включен режим встроенных контрактных заглушек.")


# =====================================================================
# Менеджер WebSocket для стриминга вызовов @mcp.tool() на фронтенд
# =====================================================================
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket подключен: session_id={session_id}")

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)

    async def send_event(self, session_id: str, event_type: str, payload: dict[str, Any]):
        ws = self.active_connections.get(session_id)
        if ws:
            try:
                await ws.send_json({"event": event_type, "data": payload})
            except Exception as exc:
                logger.warning(f"Ошибка отправки WS-события: {exc}")


ws_manager = ConnectionManager()


# =====================================================================
# Универсальный вызыватель инструментов MCP-сервера (7 тулов из доки)
# =====================================================================
async def invoke_mcp_tool(
    tool_name: str,
    kwargs: dict[str, Any],
    session_id: Optional[str] = None,
    trace_log: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """
    Вызывает инструмент @mcp.tool() из mcp_server/server.py,
    стримит статус в WebSocket и сохраняет трассировку для UI.
    """
    logger.info(f"[MCP CALL] {tool_name}({kwargs})")
    if session_id:
        await ws_manager.send_event(session_id, "mcp_tool_start", {"tool": tool_name, "args": kwargs})

    result: Any = None

    # 1. Если mcp_server/server.py реализован — вызываем реальную функцию
    if mcp_srv is not None and hasattr(mcp_srv, tool_name):
        fn = getattr(mcp_srv, tool_name)
        try:
            if inspect.iscoroutinefunction(fn):
                result = await fn(**kwargs)
            else:
                result = fn(**kwargs)
        except Exception as exc:
            logger.error(f"Ошибка внутри MCP-инструмента {tool_name}: {exc}")
            result = {"error": str(exc), "status": "FAILED"}
    else:
        # 2. Контрактный fallback (строго по документации твоих 7 тулов)
        result = _fallback_mcp_execution(tool_name, kwargs)

    if trace_log is not None:
        trace_log.append({"tool": tool_name, "args": kwargs, "result": result})

    if session_id:
        await ws_manager.send_event(session_id, "mcp_tool_result", {"tool": tool_name, "result": result})

    return result


def _fallback_mcp_execution(tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Fallback-ответы, на 100% соответствующие документации 7 инструментов."""
    if tool_name == "analyze_stl":
        return {
            "stl_path": kwargs["stl_path"],
            "dimensions_mm": {"x": 85.0, "y": 62.5, "z": 44.0},
            "volume_cm3": 38.4,
            "surface_area_cm2": 142.6,
            "overhang_percent": 31.5,
            "bed_detachment_risk": "LOW",
            "recommended_mode": "5D",
            "verdict": "Выявлено 31.5% нависаний (>45°). Рекомендована 5D-печать без поддержек.",
        }

    if tool_name == "recommend_print_parameters":
        mat = kwargs.get("material", "PA12-CF").upper()
        strength = kwargs.get("strength_priority", False)
        temps = {
            "PLA": (210, 60),
            "PETG": (240, 75),
            "ABS": (255, 105),
            "PA12-CF": (280, 100),
            "PEEK": (400, 140),
        }
        nozzle, bed = temps.get(mat, (250, 90))
        return {
            "material": mat,
            "nozzle_temp": nozzle,
            "bed_temp": bed,
            "print_speed": 45 if strength else 65,
            "infill_percent": 80 if strength else 40,
            "layer_height": 0.15 if strength else 0.2,
            "supports_strategy": "NONE_5D_MULTIAXIS" if mat in ("PA12-CF", "PEEK") else "AUTO_TREE",
        }

    if tool_name == "generate_safe_gcode":
        out_path = Path(kwargs["output_gcode_path"])
        mode_5d = kwargs.get("mode_5d", False)
        gcode_content = (
            f"; Rosatom Safe G-Code | Material: {kwargs.get('material')} | 5D: {mode_5d}\n"
            "G28 X Y Z ; Safe Homing\n"
            f"G1 Z{kwargs.get('layer_height', 0.2)} F1200\n"
            + ("G1 X40 Y40 A25.0 B90.0 E4.2 F1800 ; 5D Stereotech Kinematics\n" if mode_5d else "G1 X40 Y40 E4.2 F1800\n")
        )
        out_path.write_text(gcode_content, encoding="utf-8")
        return {
            "status": "GENERATED",
            "output_gcode_path": str(out_path),
            "estimated_print_time": "1h 45m" if mode_5d else "2h 30m",
            "filament_weight_g": 68.4 if mode_5d else 96.2,
            "security_precheck": "PASSED",
        }

    if tool_name == "verify_gcode_security":
        return {
            "gcode_path": kwargs["gcode_path"],
            "is_safe": True,
            "warnings": [],
            "violations": [],
            "sha256_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        }

    if tool_name == "diagnose_print_defect_or_spool":
        defect = kwargs.get("defect_type", "warping")
        return {
            "defect_type": defect,
            "current_material": kwargs.get("current_material", "ABS"),
            "parameter_adjustments": {
                "bed_temp_delta": "+10°C",
                "cooling_fan_percent": 0,
                "retraction_speed_mm_s": 40,
                "brim_width_mm": 8,
            },
            "operator_actions": [
                "Обезжирить печатный стол изопропанолом и нанести адгезив",
                "Просушить катушку филамента при 70°C в течение 4 часов",
                "Закрыть дверцу термокамеры принтера",
            ],
        }

    if tool_name == "get_fleet_status":
        return {
            "printer_3d_1": {
                "name": "Voron 2.4 Industrial",
                "kinematics": "3D",
                "status": "IDLE",
                "max_nozzle_temp": 300,
                "max_bed_temp": 120,
            },
            "printer_5d_1": {
                "name": "Stereotech STE 530 Hybrid",
                "kinematics": "5D",
                "status": "IDLE",
                "max_nozzle_temp": 420,
                "max_bed_temp": 150,
            },
        }

    if tool_name == "send_to_printer":
        return {
            "status": "DISPATCHED",
            "printer_id": kwargs["printer_id"],
            "gcode_path": kwargs["gcode_path"],
            "verified_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "message": f"Задание успешно отправлено на {kwargs['printer_id']}. Начат прогрев стола и экструдера.",
        }

    return {"error": f"Неизвестный инструмент {tool_name}"}


# =====================================================================
# Вспомогательные парсеры интента из текста пользователя
# =====================================================================
def extract_material_from_text(text: str) -> str:
    t = text.upper()
    for mat in ("PA12-CF", "PEEK", "PETG", "ABS", "PLA", "ASA", "PC", "NYLON"):
        if mat in t:
            return mat
    if "УГЛЕ" in t or "КАРБОН" in t or "ПОЛИАМИД" in t:
        return "PA12-CF"
    if "ВЫСОКОТЕМП" in t or "АТОМ" in t:
        return "PEEK"
    return "PA12-CF"  # Дефолтный инженерный пластик для 5D


def extract_defect_from_text(text: str) -> Optional[str]:
    t = text.lower()
    if any(w in t for w in ("паутин", "волос", "сопл", "stringing")):
        return "stringing"
    if any(w in t for w in ("отрыв", "угл", "загиб", "отлип", "warping", "деламин", "трещин")):
        return "warping"
    if any(w in t for w in ("смещ", "сдвиг", "ступен", "layer_shift", "shift")):
        return "layer_shift"
    if any(w in t for w in ("недоэкструз", "пропуск", "дыр", "underextrusion", "катушк", "влажн")):
        return "underextrusion"
    return None


# =====================================================================
# Инициализация FastAPI
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск Rosatom 3D/5D Orchestrator API...")
    yield
    logger.info("Остановка Rosatom 3D/5D Orchestrator API...")


app = FastAPI(
    title="Rosatom 3D/5D Print MCP Orchestrator",
    description="API оркестратора, интегрированное с 7 инструментами @mcp.tool() из mcp_server/server.py",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Раздача статики (STL, рендеры, готовый G-код) для 3D/G-code вьювера на фронтенде
app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")


# =====================================================================
# Схемы запросов и ответов (Pydantic)
# =====================================================================
class ChatOrchestratorResponse(BaseModel):
    session_id: str
    reply: str
    stl_url: Optional[str] = None
    gcode_url: Optional[str] = None
    # Результаты работы каждого из 7 MCP-инструментов:
    stl_analysis: Optional[dict[str, Any]] = None
    recommended_parameters: Optional[dict[str, Any]] = None
    gcode_generation: Optional[dict[str, Any]] = None
    security_audit: Optional[dict[str, Any]] = None
    defect_diagnosis: Optional[dict[str, Any]] = None
    fleet_status: Optional[dict[str, Any]] = None
    dispatch_result: Optional[dict[str, Any]] = None
    mcp_tool_trace: list[dict[str, Any]] = Field(default_factory=list)


class DiagnoseRequest(BaseModel):
    defect_type: str = Field(..., examples=["warping", "stringing", "layer_shift", "underextrusion"])
    observed_symptoms: str = Field(..., examples=["Отрывает правый угол детали из ABS на 15-м слое"])
    current_material: Optional[str] = Field(default="ABS")


class VerifyGcodeRequest(BaseModel):
    gcode_path: str


class SendToPrinterRequest(BaseModel):
    printer_id: str = Field(..., examples=["printer_5d_1", "printer_3d_1"])
    gcode_path: str


# =====================================================================
# 1. ГЛАВНЫЙ ЭНДПОИНТ ЧАТА (Оркестрация всех 7 тулов MCP)
# =====================================================================
@app.post("/api/chat", response_model=ChatOrchestratorResponse)
async def chat_endpoint(
    message: str = Form(..., description="Текстовый запрос пользователя"),
    session_id: str = Form(default_factory=lambda: str(uuid.uuid4())),
    material: Optional[str] = Form(default=None, description="Явно выбранный пластик (опционально)"),
    strength_priority: Optional[bool] = Form(default=None, description="Приоритет максимальной прочности"),
    auto_send_to_printer: bool = Form(default=False, description="Сразу отправить на свободный принтер"),
    stl_file: Optional[UploadFile] = File(default=None, description="Файл 3D-модели (.stl)"),
    photo_file: Optional[UploadFile] = File(default=None, description="Фото дефекта печати, катушки или чертежа"),
):
    trace_log: list[dict[str, Any]] = []
    reply_parts: list[str] = []
    file_prefix = uuid.uuid4().hex[:8]

    stl_url: Optional[str] = None
    gcode_url: Optional[str] = None
    stl_analysis_res: Optional[dict[str, Any]] = None
    params_res: Optional[dict[str, Any]] = None
    gcode_res: Optional[dict[str, Any]] = None
    security_res: Optional[dict[str, Any]] = None
    defect_res: Optional[dict[str, Any]] = None
    fleet_res: Optional[dict[str, Any]] = None
    dispatch_res: Optional[dict[str, Any]] = None

    chosen_material = (material or extract_material_from_text(message)).upper()
    is_strength_priority = (
        strength_priority
        if strength_priority is not None
        else any(w in message.lower() for w in ("прочн", "нагрузк", "удар", "силов", "кронштейн"))
    )

    # -----------------------------------------------------------------
    # ВЕТКА А: Загружена STL-модель -> Анализ, Параметры, G-code, Аудит
    # -----------------------------------------------------------------
    if stl_file and stl_file.filename:
        if not stl_file.filename.lower().endswith(".stl"):
            raise HTTPException(status_code=400, detail="Файл модели должен иметь расширение .stl")

        saved_stl_path = STL_DIR / f"{file_prefix}_{stl_file.filename}"
        with saved_stl_path.open("wb") as buffer:
            shutil.copyfileobj(stl_file.file, buffer)
        stl_url = f"/storage/uploads_stl/{saved_stl_path.name}"

        # Тул №1: analyze_stl(stl_path)
        stl_analysis_res = await invoke_mcp_tool(
            "analyze_stl",
            {"stl_path": str(saved_stl_path)},
            session_id=session_id,
            trace_log=trace_log,
        )

        rec_mode_raw = str(stl_analysis_res.get("recommended_mode", "5D")).upper()
        mode_5d_flag = "5D" in rec_mode_raw

        # Тул №2: recommend_print_parameters(stl_path, material, strength_priority)
        params_res = await invoke_mcp_tool(
            "recommend_print_parameters",
            {
                "stl_path": str(saved_stl_path),
                "material": chosen_material,
                "strength_priority": is_strength_priority,
            },
            session_id=session_id,
            trace_log=trace_log,
        )

        layer_height = float(params_res.get("layer_height", 0.2))
        infill_int = int(params_res.get("infill_percent", 60 if is_strength_priority else 40))

        # Тул №3: generate_safe_gcode(...)
        output_gcode_filename = f"{saved_stl_path.stem}_{'5d' if mode_5d_flag else '3d'}.gcode"
        output_gcode_path = GCODE_DIR / output_gcode_filename

        gcode_res = await invoke_mcp_tool(
            "generate_safe_gcode",
            {
                "stl_path": str(saved_stl_path),
                "material": chosen_material,
                "layer_height": layer_height,
                "infill": infill_int,
                "mode_5d": mode_5d_flag,
                "output_gcode_path": str(output_gcode_path),
            },
            session_id=session_id,
            trace_log=trace_log,
        )

        if output_gcode_path.exists():
            gcode_url = f"/storage/generated_gcode/{output_gcode_filename}"

        # Тул №4: verify_gcode_security(gcode_path)
        security_res = await invoke_mcp_tool(
            "verify_gcode_security",
            {"gcode_path": str(output_gcode_path)},
            session_id=session_id,
            trace_log=trace_log,
        )

        reply_parts.append(
            f"**Геометрический анализ (`analyze_stl`):** габариты {stl_analysis_res.get('dimensions_mm')}, "
            f"объем {stl_analysis_res.get('volume_cm3')} см³, нависания >45°: {stl_analysis_res.get('overhang_percent')}%, "
            f"риск отрыва от стола: `{stl_analysis_res.get('bed_detachment_risk')}`. "
            f"Рекомендован режим: **{'5D (Stereotech)' if mode_5d_flag else '3D'}**.\n\n"
            f"**Параметры печати (`recommend_print_parameters`):** материал `{chosen_material}`, "
            f"сопло {params_res.get('nozzle_temp')}°C, стол {params_res.get('bed_temp')}°C, "
            f"слой {layer_height} мм, заполнение {infill_int}%.\n\n"
            f"**Генерация и аудит G-кода (`generate_safe_gcode` + `verify_gcode_security`):** "
            f"файл `{output_gcode_filename}`, безопасен: `{security_res.get('is_safe')}`, "
            f"SHA-256: `{str(security_res.get('sha256_hash', ''))[:16]}...`."
        )

    # -----------------------------------------------------------------
    # ВЕТКА Б: Диагностика дефекта или катушки (по фото или тексту)
    # -----------------------------------------------------------------
    detected_defect = extract_defect_from_text(message)
    if photo_file and photo_file.filename:
        saved_photo = RENDERS_DIR / f"{file_prefix}_{photo_file.filename}"
        with saved_photo.open("wb") as buffer:
            shutil.copyfileobj(photo_file.file, buffer)
        if not detected_defect:
            detected_defect = "warping"  # Дефолтный класс дефекта при прикреплении фото брака

    if detected_defect:
        # Тул №5: diagnose_print_defect_or_spool(defect_type, observed_symptoms, current_material)
        defect_res = await invoke_mcp_tool(
            "diagnose_print_defect_or_spool",
            {
                "defect_type": detected_defect,
                "observed_symptoms": message,
                "current_material": chosen_material,
            },
            session_id=session_id,
            trace_log=trace_log,
        )
        reply_parts.append(
            f"**Диагностика дефекта (`diagnose_print_defect_or_spool`):** выявлен `{detected_defect}` "
            f"на материале `{chosen_material}`. Рекомендации: `{defect_res}`."
        )

    # -----------------------------------------------------------------
    # ВЕТКА В: Опрос парка принтеров и отправка задания на печать
    # -----------------------------------------------------------------
    wants_fleet = any(w in message.lower() for w in ("принтер", "парк", "статус", "свобод", "отправ", "печат"))
    if stl_analysis_res or wants_fleet or auto_send_to_printer:
        # Тул №6: get_fleet_status()
        fleet_res = await invoke_mcp_tool(
            "get_fleet_status",
            {},
            session_id=session_id,
            trace_log=trace_log,
        )

        # Если нужно сразу отправить G-код на принтер
        wants_dispatch = auto_send_to_printer or any(
            w in message.lower() for w in ("отправь на печать", "запусти печать", "напечатай")
        )
        if wants_dispatch and gcode_res and security_res and security_res.get("is_safe", False):
            target_kinematics = "5D" if (stl_analysis_res and "5D" in str(stl_analysis_res.get("recommended_mode"))) else "3D"
            selected_printer_id: Optional[str] = None

            if isinstance(fleet_res, dict):
                for pid, pinfo in fleet_res.items():
                    if isinstance(pinfo, dict) and pinfo.get("status") == "IDLE":
                        if target_kinematics in str(pinfo.get("kinematics", "")).upper():
                            selected_printer_id = pid
                            break
                # Если не нашли по кинематике, берем любой свободный
                if not selected_printer_id:
                    for pid, pinfo in fleet_res.items():
                        if isinstance(pinfo, dict) and pinfo.get("status") == "IDLE":
                            selected_printer_id = pid
                            break

            if selected_printer_id:
                # Тул №7: send_to_printer(printer_id, gcode_path)
                dispatch_res = await invoke_mcp_tool(
                    "send_to_printer",
                    {
                        "printer_id": selected_printer_id,
                        "gcode_path": gcode_res.get("output_gcode_path") or str(output_gcode_path),
                    },
                    session_id=session_id,
                    trace_log=trace_log,
                )
                reply_parts.append(
                    f"**Диспетчеризация (`send_to_printer`):** задание отправлено на `{selected_printer_id}` "
                    f"(статус: `{dispatch_res.get('status')}`)."
                )

    if not reply_parts:
        reply_parts.append(
            "Готов к работе! Загрузите `.stl` модель для автоматического подбора параметров и генерации безопасного "
            "3D/5D G-кода, прикрепите фото дефекта печати или запросите статус парка принтеров."
        )

    return ChatOrchestratorResponse(
        session_id=session_id,
        reply="\n\n".join(reply_parts),
        stl_url=stl_url,
        gcode_url=gcode_url,
        stl_analysis=stl_analysis_res,
        recommended_parameters=params_res,
        gcode_generation=gcode_res,
        security_audit=security_res,
        defect_diagnosis=defect_res,
        fleet_status=fleet_res,
        dispatch_result=dispatch_res,
        mcp_tool_trace=trace_log,
    )


# =====================================================================
# 2. ПРЯМЫЕ ЭНДПОИНТЫ ПОД ИНСТРУМЕНТЫ MCP (Для кнопок на фронтенде)
# =====================================================================

@app.get("/api/fleet", summary="Вызов MCP: get_fleet_status()")
async def api_get_fleet_status():
    """Возвращает словарь парка 3D/5D-принтеров Росатома (`printer_3d_1`, `printer_5d_1` и т.д.)."""
    return await invoke_mcp_tool("get_fleet_status", {})


@app.post("/api/fleet/print", summary="Вызов MCP: send_to_printer(printer_id, gcode_path)")
async def api_send_to_printer(req: SendToPrinterRequest):
    """Финальная отправка проверенного G-кода на выбранный принтер из парка."""
    return await invoke_mcp_tool(
        "send_to_printer",
        {"printer_id": req.printer_id, "gcode_path": req.gcode_path},
    )


@app.post("/api/gcode/verify", summary="Вызов MCP: verify_gcode_security(gcode_path)")
async def api_verify_gcode(req: VerifyGcodeRequest):
    """Жесткий статический аудит G-кода и расчет SHA-256 хэша."""
    return await invoke_mcp_tool("verify_gcode_security", {"gcode_path": req.gcode_path})


@app.post("/api/diagnose", summary="Вызов MCP: diagnose_print_defect_or_spool(...)")
async def api_diagnose_defect(req: DiagnoseRequest):
    """Анализ дефекта печати (stringing, warping, layer_shift, underextrusion) или проблем с катушкой."""
    return await invoke_mcp_tool(
        "diagnose_print_defect_or_spool",
        {
            "defect_type": req.defect_type,
            "observed_symptoms": req.observed_symptoms,
            "current_material": req.current_material,
        },
    )


# =====================================================================
# 3. WEBSOCKET ДЛЯ ЖИВОЙ ТЕЛЕМЕТРИИ И ТРАССИРОВКИ MCP В UI
# =====================================================================
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await ws_manager.connect(session_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(session_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("orchestrator.main:app", host="0.0.0.0", port=8000, reload=True)