import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# Пути к директориям проекта
BASE_DIR = Path(__file__).resolve().parent.parent.parent
KB_DIR = Path(__file__).resolve().parent / "knowledge_base"
LOCAL_QDRANT_PATH = BASE_DIR / "storage" / "qdrant_local"

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "rosatom_3d_5d_kb")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE = 384  # Размерность для paraphrase-multilingual-MiniLM-L12-v2


# =====================================================================
# 1. Эталонные справочники Росатома (записываются, если файлы пустые)
# =====================================================================
DEFAULT_FILAMENTS = [
    {
        "material": "PLA",
        "full_name": "Polylactic Acid (Полилактид базовый)",
        "category": "standard",
        "nozzle_temp": 210,
        "bed_temp": 60,
        "chamber_temp": 25,
        "print_speed": 70,
        "layer_height": 0.2,
        "infill_percent_default": 30,
        "infill_percent_strength": 60,
        "cooling_fan_percent": 100,
        "bed_detachment_risk": "LOW",
        "supports_strategy": "AUTO_TREE",
        "recommended_kinematics": ["3D", "5D"],
        "drying_regime": "45°C в течение 4 часов при влажности >40%",
        "notes": "Подходит для прототипирования и ненагруженных корпусов. Низкая термостойкость (размягчение от 55°C).",
    },
    {
        "material": "PETG",
        "full_name": "Polyethylene Terephthalate Glycol (ПЭТГ химически стойкий)",
        "category": "engineering",
        "nozzle_temp": 240,
        "bed_temp": 75,
        "chamber_temp": 35,
        "print_speed": 55,
        "layer_height": 0.2,
        "infill_percent_default": 40,
        "infill_percent_strength": 75,
        "cooling_fan_percent": 40,
        "bed_detachment_risk": "LOW",
        "supports_strategy": "GRID_EASY_RELEASE",
        "recommended_kinematics": ["3D", "5D"],
        "drying_regime": "60°C в течение 5 часов (склонен к образованию паутины при влажности)",
        "notes": "Высокая межслойная адгезия, химстойкость к кислотам и щелочам.",
    },
    {
        "material": "ABS",
        "full_name": "Acrylonitrile Butadiene Styrene (АБС конструкционный)",
        "category": "engineering",
        "nozzle_temp": 255,
        "bed_temp": 105,
        "chamber_temp": 60,
        "print_speed": 50,
        "layer_height": 0.2,
        "infill_percent_default": 40,
        "infill_percent_strength": 80,
        "cooling_fan_percent": 0,
        "bed_detachment_risk": "HIGH",
        "supports_strategy": "NONE_5D_MULTIAXIS",
        "recommended_kinematics": ["5D", "3D"],
        "drying_regime": "70°C в течение 4 часов",
        "notes": "Высокая усадка (до 0.8%). Требуется закрытая термокамера и кайма (Brim 8 мм) для предотвращения отрыва углов.",
    },
    {
        "material": "PA12-CF",
        "full_name": "Polyamide 12 Carbon Fiber 15% (Углеродный полиамид для 5D Stereotech)",
        "category": "composite_5d",
        "nozzle_temp": 280,
        "bed_temp": 100,
        "chamber_temp": 70,
        "print_speed": 45,
        "layer_height": 0.15,
        "infill_percent_default": 60,
        "infill_percent_strength": 90,
        "cooling_fan_percent": 0,
        "bed_detachment_risk": "MEDIUM",
        "supports_strategy": "NONE_5D_MULTIAXIS",
        "recommended_kinematics": ["5D"],
        "drying_regime": "80°C в течение 8 часов в вакуумной сушилке непосредственно перед печатью",
        "notes": "Основной конструкционный композит для 5D-печати силовых кронштейнов, крыльчаток и оснастки без поддержек. Требуется закаленное стальное или рубиновое сопло.",
    },
    {
        "material": "PEEK",
        "full_name": "Polyether Ether Ketone (Полиэфирэфиркетон для атомной промышленности)",
        "category": "high_temp_nuclear",
        "nozzle_temp": 400,
        "bed_temp": 140,
        "chamber_temp": 120,
        "print_speed": 30,
        "layer_height": 0.15,
        "infill_percent_default": 80,
        "infill_percent_strength": 100,
        "cooling_fan_percent": 0,
        "bed_detachment_risk": "HIGH",
        "supports_strategy": "NONE_5D_MULTIAXIS",
        "recommended_kinematics": ["5D"],
        "drying_regime": "150°C в течение 6 часов в промышленном шкафу",
        "notes": "Сверхвысокотемпературный радиационно-стойкий полимер Росатома (рабочая температура до 250°C). Печать строго на высокотемпературных 5D-системах с контролем отжига.",
    },
]

DEFAULT_DEFECTS = [
    {
        "defect_type": "stringing",
        "title": "Образование паутины и волосков (Stringing / Oozing)",
        "observed_symptoms": "Тонкие нити пластика между холостыми перемещениями экструдера, капли на поверхности детали, влажный треск из сопла.",
        "root_causes": [
            "Влажная катушка филамента (особенно PA12-CF, PETG, Nylon)",
            "Завышенная температура сопла",
            "Недостаточная длина или скорость ретракта (отката нити)",
        ],
        "parameter_adjustments": {
            "nozzle_temp_delta": -10,
            "retraction_distance_mm": 1.8,
            "retraction_speed_mm_s": 45,
            "travel_speed_mm_s": 180,
        },
        "operator_actions": [
            "Просушить катушку филамента согласно регламенту материала (60–80°C, 4–8 часов)",
            "Снизить температуру экструдера на 5–10°C",
            "Активировать функцию Wipe Nozzle и избегать пересечения периметров при перемещении",
        ],
    },
    {
        "defect_type": "warping",
        "title": "Термическая деформация, отрыв углов от стола и деламинация (Warping / Cracking)",
        "observed_symptoms": "Углы детали загибаются вверх, деталь отрывается от печатного стола, горизонтальные трещины между слоями на большой высоте.",
        "root_causes": [
            "Высокая термическая усадка материала (ABS, PC, PEEK) при быстром охлаждении",
            "Недостаточная температура стола или сквозняк в камере",
            "Малая площадь первого слоя без использования каймы (Brim)",
        ],
        "parameter_adjustments": {
            "bed_temp_delta": +10,
            "cooling_fan_percent": 0,
            "brim_width_mm": 10,
            "first_layer_speed_mm_s": 20,
        },
        "operator_actions": [
            "Обезжирить стол изопропанолом и нанести высокотемпературный адгезив (клей-спрей или PVP/DIMAFix)",
            "Закрыть термокамеру и прогреть ее до 60–100°C перед стартом печати",
            "Переключить печать в 5D-режим для снижения внутренних межслойных напряжений",
        ],
    },
    {
        "defect_type": "layer_shift",
        "title": "Смещение слоев по осям X/Y/A/B (Layer Shifting)",
        "observed_symptoms": "Верхняя часть детали ступенчато сдвинута относительно нижней, слышны пропуски шагов шаговых двигателей.",
        "root_causes": [
            "Превышение допустимых ускорений и скорости печати",
            "Задевание соплом за загнувшийся вверх край нависания (Curling)",
            "Ослабление ремней кинематики или перегрев драйверов шаговых моторов",
        ],
        "parameter_adjustments": {
            "print_speed_delta_percent": -25,
            "max_acceleration_mm_s2": 1500,
            "z_hop_mm": 0.4,
        },
        "operator_actions": [
            "Снизить рабочую скорость печати и ускорения на 25%",
            "Включить Z-Hop (подъем сопла при перемещении) на 0.4 мм",
            "Проверить натяжение ремней и калибровку нулевых точек 5-осевого поворотного модуля (оси A и B)",
        ],
    },
    {
        "defect_type": "underextrusion",
        "title": "Недоэкструзия, пропуски в периметрах и запутывание катушки (Underextrusion / Spool Jam)",
        "observed_symptoms": "Рыхлые слои с щелями, губка вместо сплошной стенки, щелчки подающего механизма (экструдера), перехлест витков на катушке.",
        "root_causes": [
            "Частичный засор сопла (особенно при печати углеродным композитом PA12-CF на сопле < 0.4 мм)",
            "Слишком низкая температура сопла для выбранной объемной скорости потока",
            "Перехлест нити на катушке или повышенное трение в PTFE-трубке",
        ],
        "parameter_adjustments": {
            "nozzle_temp_delta": +10,
            "flow_rate_percent": 105,
            "print_speed_delta_percent": -20,
        },
        "operator_actions": [
            "Проверить свободный ход катушки филамента на держателе (исключить перехлест витков)",
            "Выполнить холодную прочистку сопла (Cold Pull) или заменить сопло на закаленное диаметром 0.6 мм для композитов",
            "Повысить температуру экструдера на 10°C и снизить скорость печати",
        ],
    },
]

DEFAULT_KINEMATICS_MD = """# Регламент выбора 3D vs 5D кинематики и безопасности G-кода (Росатом)

## 1. Критерии автоматического выбора 5D-печати (Stereotech STE 320 / 530)
Многоосевая 5D-печать (оси X, Y, Z + наклон стола A и вращение B) назначается системой в следующих случаях:
- **Критическая доля нависаний:** если процент граней с углом нависания более 45° (`overhang_percent`) превышает 15% от площади поверхности. Поворот стола на угол A позволяет печатать вылеты без поддержек.
- **Силовые детали (`strength_priority = true`):** при наличии изгибающих, крутящих или разрывных нагрузок 5D-укладка ориентирует полимерные нити вдоль силовых линий детали, повышая прочность на межслойный сдвиг на 40–60%.
- **Труднообрабатываемые полимеры (`PA12-CF`, `PEEK`):** удаление поддержек на высокопрочных композитах и PEEK повреждает рабочие поверхности, поэтому для них приоритетен 5D-режим без поддержек.

## 2. Критерии классической 3D-печати
- Плоские корпусные детали с большой площадью основания (`bed_detachment_risk = LOW`) и долей нависаний менее 15%.
- Сжимающие нагрузки, направленные строго перпендикулярно плоскости стола (вдоль оси Z).

## 3. Правила статического аудита безопасности G-кода (`verify_gcode_security`)
Перед отправкой задания через `send_to_printer` каждый `.gcode` файл проходит обязательную проверку:
1. **Безопасная парковка (`G28`):** запрещены неконтролируемые обнуления осей без предварительного подъема по Z при наклоненном 5D-столе (риск соударения сопла с поворотным кулаком).
2. **Температурные лимиты:** температура сопла (`M104`/`M109`) не должна превышать `max_nozzle_temp` целевого принтера (300°C для 3D Voron, 420°C для 5D Stereotech Hybrid).
3. **Габаритный коридор:** координаты X, Y, Z и углы наклона A (-15°..+95°) не должны выходить за физическую зону печати.
4. **Контроль целостности:** вычисляется криптографический хэш `SHA-256`, который сверяется диспетчером принтера перед стартом нагрева.
"""


# =====================================================================
# 2. Автозаполнение пустых файлов базы знаний
# =====================================================================
def ensure_knowledge_base_files() -> tuple[Path, Path, Path]:
    """Проверяет наличие и непустоту файлов БЗ; если они пустые — записывает эталонные данные."""
    KB_DIR.mkdir(parents=True, exist_ok=True)

    filaments_path = KB_DIR / "filaments_rosatom.json"
    defects_path = KB_DIR / "defects_guide.json"
    kinematics_path = KB_DIR / "kinematics_3d_vs_5d.md"

    if not filaments_path.exists() or filaments_path.stat().st_size == 0:
        filaments_path.write_text(json.dumps(DEFAULT_FILAMENTS, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Записан справочник материалов Росатома: {filaments_path}")

    if not defects_path.exists() or defects_path.stat().st_size == 0:
        defects_path.write_text(json.dumps(DEFAULT_DEFECTS, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"Записан справочник дефектов печати: {defects_path}")

    if not kinematics_path.exists() or kinematics_path.stat().st_size == 0:
        kinematics_path.write_text(DEFAULT_KINEMATICS_MD, encoding="utf-8")
        logger.info(f"Записан регламент 3D vs 5D и безопасности G-кода: {kinematics_path}")

    return filaments_path, defects_path, kinematics_path


# =====================================================================
# 3. Формирование чанков с метаданными для векторного поиска
# =====================================================================
def build_documents() -> list[dict[str, Any]]:
    """Преобразует JSON и Markdown справочники в список документов для индексации."""
    filaments_path, defects_path, kinematics_path = ensure_knowledge_base_files()
    docs: list[dict[str, Any]] = []

    # 1. Чанки по материалам (для recommend_print_parameters)
    filaments = json.loads(filaments_path.read_text(encoding="utf-8"))
    for item in filaments:
        mat_code = item["material"]
        text_chunk = (
            f"Материал для 3D/5D печати: {mat_code} ({item['full_name']}). "
            f"Температура сопла: {item['nozzle_temp']}°C, температура стола: {item['bed_temp']}°C, "
            f"термокамера: {item['chamber_temp']}°C. Скорость печати: {item['print_speed']} мм/с, "
            f"высота слоя: {item['layer_height']} мм, заполнение: {item['infill_percent_default']}% "
            f"(силовой режим: {item['infill_percent_strength']}%). "
            f"Риск отрыва от стола: {item['bed_detachment_risk']}. "
            f"Стратегия поддержек: {item['supports_strategy']}. Сушка: {item['drying_regime']}. "
            f"Особенности: {item['notes']}"
        )
        docs.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"filament_{mat_code}")),
                "doc_type": "filament",
                "key": mat_code,
                "text": text_chunk,
                "metadata": item,
            }
        )

    # 2. Чанки по дефектам и проблемам с катушкой (для diagnose_print_defect_or_spool)
    defects = json.loads(defects_path.read_text(encoding="utf-8"))
    for item in defects:
        d_type = item["defect_type"]
        text_chunk = (
            f"Дефект 3D/5D-печати: {d_type} — {item['title']}. "
            f"Симптомы: {item['observed_symptoms']} "
            f"Причины: {'; '.join(item['root_causes'])}. "
            f"Корректировка параметров: {json.dumps(item['parameter_adjustments'], ensure_ascii=False)}. "
            f"Действия оператора: {'; '.join(item['operator_actions'])}."
        )
        docs.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"defect_{d_type}")),
                "doc_type": "defect",
                "key": d_type,
                "text": text_chunk,
                "metadata": item,
            }
        )

    # 3. Чанки по правилам 3D vs 5D и безопасности G-кода
    md_text = kinematics_path.read_text(encoding="utf-8")
    sections = [sec.strip() for sec in md_text.split("## ") if sec.strip()]
    for idx, sec in enumerate(sections):
        title_line = sec.splitlines()[0].lstrip("# ").strip()
        docs.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"kinematics_sec_{idx}")),
                "doc_type": "kinematics_and_security",
                "key": title_line,
                "text": f"## {sec}",
                "metadata": {"section_title": title_line},
            }
        )

    return docs


# =====================================================================
# 4. Генерация эмбеддингов (FastEmbed с оффлайн-фолбэком)
# =====================================================================
def compute_embeddings(texts: list[str]) -> list[list[float]]:
    """Кодирует тексты через FastEmbed; если нет интернета для скачивания весов — включает детерминированный fallback."""
    try:
        from fastembed import TextEmbedding

        logger.info(f"Загрузка модели эмбеддингов FastEmbed ({EMBEDDING_MODEL_NAME})...")
        embedder = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        vectors = [vec.tolist() for vec in embedder.embed(texts)]
        logger.success(f"Сгенерировано {len(vectors)} векторов размерности {len(vectors[0])}.")
        return vectors
    except Exception as exc:
        logger.warning(f"FastEmbed недоступен ({exc}). Используется детерминированный хэш-эмбеддер (размерность {VECTOR_SIZE}).")
        fallback_vectors: list[list[float]] = []
        for t in texts:
            vec: list[float] = []
            for i in range(VECTOR_SIZE):
                h = hashlib.sha256(f"{i}:{t.lower()}".encode("utf-8")).digest()
                val = (int.from_bytes(h[:4], "big") / 0xFFFFFFFF) * 2.0 - 1.0
                vec.append(round(val, 6))
            fallback_vectors.append(vec)
        return fallback_vectors


# =====================================================================
# 5. Подключение к Qdrant (Docker сервер или локальная папка)
# =====================================================================
def get_qdrant_client() -> QdrantClient:
    """Пробует подключиться к Qdrant по HTTP, а если он не запущен — открывает локальную БД на диске."""
    try:
        client = QdrantClient(url=QDRANT_URL, timeout=2.0)
        client.get_collections()
        logger.info(f"Подключен к серверу Qdrant по адресу {QDRANT_URL}")
        return client
    except Exception:
        LOCAL_QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        logger.warning(f"Сервер Qdrant ({QDRANT_URL}) не отвечает. Используется локальная БД: {LOCAL_QDRANT_PATH}")
        return QdrantClient(path=str(LOCAL_QDRANT_PATH))


# =====================================================================
# 6. Главная функция индексации в Qdrant
# =====================================================================
def ingest_knowledge_base(recreate: bool = True) -> dict[str, Any]:
    """Полный пайплайн загрузки справочников Росатома в векторную БД Qdrant."""
    docs = build_documents()
    texts = [d["text"] for d in docs]
    vectors = compute_embeddings(texts)
    dim = len(vectors[0]) if vectors else VECTOR_SIZE

    client = get_qdrant_client()

    existing = [c.name for c in client.get_collections().collections]
    if recreate and COLLECTION_NAME in existing:
        client.delete_collection(collection_name=COLLECTION_NAME)
        logger.info(f"Старая коллекция '{COLLECTION_NAME}' удалена.")

    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE),
        )
        logger.info(f"Создана коллекция Qdrant '{COLLECTION_NAME}' (dim={dim}, metric=COSINE).")

    points = [
        rest.PointStruct(
            id=doc["id"],
            vector=vec,
            payload={
                "doc_type": doc["doc_type"],
                "key": doc["key"],
                "text": doc["text"],
                "metadata": doc["metadata"],
            },
        )
        for doc, vec in zip(docs, vectors)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.success(f"В коллекцию '{COLLECTION_NAME}' успешно загружено {len(points)} документов!")

    return {
        "collection": COLLECTION_NAME,
        "total_documents": len(points),
        "filaments_loaded": sum(1 for d in docs if d["doc_type"] == "filament"),
        "defects_loaded": sum(1 for d in docs if d["doc_type"] == "defect"),
        "rules_loaded": sum(1 for d in docs if d["doc_type"] == "kinematics_and_security"),
    }


if __name__ == "__main__":
    summary = ingest_knowledge_base(recreate=True)
    print("\n--- ИТОГ ИНДЕКСАЦИИ RAG ---")
    print(json.dumps(summary, ensure_ascii=False, indent=2))