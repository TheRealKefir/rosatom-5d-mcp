"""
Knowledge base ingestion pipeline for Rosatom 3D/5D printing RAG.

Loads filament properties, defect troubleshooting guides, and kinematics regulations,
chunks them into semantic documents, computes embeddings using FastEmbed,
and upserts them into Qdrant vector collections ('filaments', 'defects', 'kinematics').
"""

import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastembed import TextEmbedding
from loguru import logger
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import ResponseHandlingException

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
KB_DIR = Path(__file__).resolve().parent / "knowledge_base"
LOCAL_QDRANT_PATH = BASE_DIR / "storage" / "qdrant_local"

# Default configuration
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
VECTOR_SIZE = 384  # Default vector dimension for BAAI/bge-small-en-v1.5 and multilingual MiniLM

# Qdrant collection names
COLLECTION_FILAMENTS = "filaments"
COLLECTION_DEFECTS = "defects"
COLLECTION_KINEMATICS = "kinematics"
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "rosatom_3d_5d_kb")  # Unified / backward compatibility

TARGET_COLLECTIONS = [
    COLLECTION_FILAMENTS,
    COLLECTION_DEFECTS,
    COLLECTION_KINEMATICS,
    COLLECTION_NAME,
]


# =====================================================================
# 1. Reference Data Fallbacks (used if disk files are empty or missing)
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
        "density_g_cm3": 1.24,
        "tensile_strength_mpa": 50,
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
        "density_g_cm3": 1.27,
        "tensile_strength_mpa": 55,
        "notes": "Высокая межслойная адгезия, химстойкость к кислотам и щелочам, отличная спекаемость слоев.",
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
        "density_g_cm3": 1.04,
        "tensile_strength_mpa": 43,
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
        "density_g_cm3": 1.15,
        "tensile_strength_mpa": 110,
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
        "density_g_cm3": 1.30,
        "tensile_strength_mpa": 100,
        "notes": "Сверхвысокотемпературный радиационно-стойкий полимер Росатома (рабочая температура до 250°C). Печать строго на высокотемпературных 5D-системах с контролем отжига.",
    },
    {
        "material": "TPU",
        "full_name": "Thermoplastic Polyurethane 95A (Термополиуретан эластичный)",
        "category": "flexible",
        "nozzle_temp": 225,
        "bed_temp": 50,
        "chamber_temp": 25,
        "print_speed": 30,
        "layer_height": 0.2,
        "infill_percent_default": 25,
        "infill_percent_strength": 50,
        "cooling_fan_percent": 80,
        "bed_detachment_risk": "LOW",
        "supports_strategy": "TREE_SLIM",
        "recommended_kinematics": ["3D", "5D"],
        "drying_regime": "55°C в течение 5 часов (крайне гидрофильный)",
        "density_g_cm3": 1.21,
        "tensile_strength_mpa": 35,
        "notes": "Эластомер для виброгасителей, уплотнителей и демпферов. Требует прямой привод экструдера (Direct Drive) и отключение ретрактов.",
    },
    {
        "material": "PC",
        "full_name": "Polycarbonate (Поликарбонат ударопрочный)",
        "category": "engineering",
        "nozzle_temp": 285,
        "bed_temp": 115,
        "chamber_temp": 80,
        "print_speed": 40,
        "layer_height": 0.18,
        "infill_percent_default": 50,
        "infill_percent_strength": 85,
        "cooling_fan_percent": 0,
        "bed_detachment_risk": "HIGH",
        "supports_strategy": "NONE_5D_MULTIAXIS",
        "recommended_kinematics": ["5D", "3D"],
        "drying_regime": "100°C в течение 6 часов",
        "density_g_cm3": 1.20,
        "tensile_strength_mpa": 72,
        "notes": "Высокая ударная вязкость и оптическая прозрачность. Высокая температура размягчения (115°C). Требует жесткого температурного контроля камеры.",
    },
]

DEFAULT_DEFECTS = [
    {
        "defect_type": "stringing",
        "title": "Образование паутины и волосков (Stringing / Oozing)",
        "symptoms": "Тонкие нити пластика между холостыми перемещениями экструдера, капли на поверхности детали, влажный треск из сопла.",
        "observed_symptoms": "Тонкие нити пластика между холостыми перемещениями экструдера, капли на поверхности детали, влажный треск из сопла.",
        "causes": [
            "Влажная катушка филамента (особенно PA12-CF, PETG, TPU, Nylon)",
            "Завышенная температура сопла",
            "Недостаточная длина или скорость ретракта (отката нити)",
        ],
        "root_causes": [
            "Влажная катушка филамента (особенно PA12-CF, PETG, TPU, Nylon)",
            "Завышенная температура сопла",
            "Недостаточная длина или скорость ретракта (отката нити)",
        ],
        "fixes": [
            "Просушить катушку филамента согласно регламенту материала (60–80°C, 4–8 часов)",
            "Снизить температуру экструдера на 5–10°C",
            "Активировать функцию Wipe Nozzle и избегать пересечения периметров при перемещении",
        ],
        "operator_actions": [
            "Просушить катушку филамента согласно регламенту материала (60–80°C, 4–8 часов)",
            "Снизить температуру экструдера на 5–10°C",
            "Активировать функцию Wipe Nozzle и избегать пересечения периметров при перемещении",
        ],
        "parameter_adjustments": {
            "nozzle_temp_delta": -10,
            "retraction_distance_mm": 1.8,
            "retraction_speed_mm_s": 45,
            "travel_speed_mm_s": 180,
        },
    },
    {
        "defect_type": "warping",
        "title": "Термическая деформация, отрыв углов от стола и деламинация (Warping / Cracking)",
        "symptoms": "Углы детали загибаются вверх, деталь отрывается от печатного стола, горизонтальные трещины между слоями на большой высоте.",
        "observed_symptoms": "Углы детали загибаются вверх, деталь отрывается от печатного стола, горизонтальные трещины между слоями на большой высоте.",
        "causes": [
            "Высокая термическая усадка материала (ABS, PC, PEEK) при быстром охлаждении",
            "Недостаточная температура стола или сквозняк в камере",
            "Малая площадь первого слоя без использования каймы (Brim)",
        ],
        "root_causes": [
            "Высокая термическая усадка материала (ABS, PC, PEEK) при быстром охлаждении",
            "Недостаточная температура стола или сквозняк в камере",
            "Малая площадь первого слоя без использования каймы (Brim)",
        ],
        "fixes": [
            "Обезжирить стол изопропанолом и нанести высокотемпературный адгезив (клей-спрей или PVP/DIMAFix)",
            "Закрыть термокамеру и прогреть ее до 60–100°C перед стартом печати",
            "Переключить печать в 5D-режим для снижения внутренних межслойных напряжений",
        ],
        "operator_actions": [
            "Обезжирить стол изопропанолом и нанести высокотемпературный адгезив (клей-спрей или PVP/DIMAFix)",
            "Закрыть термокамеру и прогреть ее до 60–100°C перед стартом печати",
            "Переключить печать в 5D-режим для снижения внутренних межслойных напряжений",
        ],
        "parameter_adjustments": {
            "bed_temp_delta": 10,
            "cooling_fan_percent": 0,
            "brim_width_mm": 10,
            "first_layer_speed_mm_s": 20,
        },
    },
    {
        "defect_type": "layer_shift",
        "title": "Смещение слоев по осям X/Y/A/B (Layer Shifting)",
        "symptoms": "Верхняя часть детали ступенчато сдвинута относительно нижней, слышны пропуски шагов шаговых двигателей.",
        "observed_symptoms": "Верхняя часть детали ступенчато сдвинута относительно нижней, слышны пропуски шагов шаговых двигателей.",
        "causes": [
            "Превышение допустимых ускорений и скорости печати",
            "Задевание соплом за загнувшийся вверх край нависания (Curling)",
            "Ослабление ремней кинематики или перегрев драйверов шаговых моторов",
        ],
        "root_causes": [
            "Превышение допустимых ускорений и скорости печати",
            "Задевание соплом за загнувшийся вверх край нависания (Curling)",
            "Ослабление ремней кинематики или перегрев драйверов шаговых моторов",
        ],
        "fixes": [
            "Снизить рабочую скорость печати и ускорения на 25%",
            "Включить Z-Hop (подъем сопла при перемещении) на 0.4 мм",
            "Проверить натяжение ремней и калибровку нулевых точек 5-осевого поворотного модуля (оси A и B)",
        ],
        "operator_actions": [
            "Снизить рабочую скорость печати и ускорения на 25%",
            "Включить Z-Hop (подъем сопла при перемещении) на 0.4 мм",
            "Проверить натяжение ремней и калибровку нулевых точек 5-осевого поворотного модуля (оси A и B)",
        ],
        "parameter_adjustments": {
            "print_speed_delta_percent": -25,
            "max_acceleration_mm_s2": 1500,
            "z_hop_mm": 0.4,
        },
    },
    {
        "defect_type": "underextrusion",
        "title": "Недоэкструзия, пропуски в периметрах и запутывание катушки (Underextrusion / Spool Jam)",
        "symptoms": "Рыхлые слои с щелями, губка вместо сплошной стенки, щелчки подающего механизма (экструдера), перехлест витков на катушке.",
        "observed_symptoms": "Рыхлые слои с щелями, губка вместо сплошной стенки, щелчки подающего механизма (экструдера), перехлест витков на катушке.",
        "causes": [
            "Частичный засор сопла (особенно при печати углеродным композитом PA12-CF на сопле < 0.4 мм)",
            "Слишком низкая температура сопла для выбранной объемной скорости потока",
            "Перехлест нити на катушке или повышенное трение в PTFE-трубке",
        ],
        "root_causes": [
            "Частичный засор сопла (особенно при печати углеродным композитом PA12-CF на сопле < 0.4 мм)",
            "Слишком низкая температура сопла для выбранной объемной скорости потока",
            "Перехлест нити на катушке или повышенное трение в PTFE-трубке",
        ],
        "fixes": [
            "Проверить свободный ход катушки филамента на держателе (исключить перехлест витков)",
            "Выполнить холодную прочистку сопла (Cold Pull) или заменить сопло на закаленное диаметром 0.6 мм для композитов",
            "Повысить температуру экструдера на 10°C и снизить скорость печати",
        ],
        "operator_actions": [
            "Проверить свободный ход катушки филамента на держателе (исключить перехлест витков)",
            "Выполнить холодную прочистку сопла (Cold Pull) или заменить сопло на закаленное диаметром 0.6 мм для композитов",
            "Повысить температуру экструдера на 10°C и снизить скорость печати",
        ],
        "parameter_adjustments": {
            "nozzle_temp_delta": 10,
            "flow_rate_percent": 105,
            "print_speed_delta_percent": -20,
        },
    },
    {
        "defect_type": "delamination",
        "title": "Межслойное расслоение и трещины (Layer Delamination)",
        "symptoms": "Слои детали легко разделяются руками, визуальные горизонтальные трещины вдоль траекторий укладки нити.",
        "observed_symptoms": "Слои детали легко разделяются руками, визуальные горизонтальные трещины вдоль траекторий укладки нити.",
        "causes": [
            "Недостаточная температура экструзии и холодная камера",
            "Слишком сильный обдув модели вентилятором",
            "Чрезмерная высота слоя (> 70% диаметра сопла)",
        ],
        "root_causes": [
            "Недостаточная температура экструзии и холодная камера",
            "Слишком сильный обдув модели вентилятором",
            "Чрезмерная высота слоя (> 70% диаметра сопла)",
        ],
        "fixes": [
            "Повысить температуру экструдера на 10-15°C и прогреть термокамеру",
            "Отключить или уменьшить скорость вентилятора обдува детали до 0-20%",
            "Уменьшить высоту слоя до 0.15-0.2 мм для лучшего пятна контакта",
            "Перейти на 5D печать с пространственной ориентацией нити по траектории нагрузок",
        ],
        "operator_actions": [
            "Повысить температуру экструдера на 10-15°C и прогреть термокамеру",
            "Отключить или уменьшить скорость вентилятора обдува детали до 0-20%",
            "Уменьшить высоту слоя до 0.15-0.2 мм для лучшего пятна контакта",
            "Перейти на 5D печать с пространственной ориентацией нити по траектории нагрузок",
        ],
        "parameter_adjustments": {
            "nozzle_temp_delta": 15,
            "chamber_temp_delta": 15,
            "cooling_fan_percent": 0,
            "layer_height_delta": -0.05,
        },
    },
]

DEFAULT_KINEMATICS_MD = """# Регламент выбора 3D vs 5D кинематики и безопасности G-кода (Росатом)

## 1. Критерии автоматического выбора 5D-печати (Stereotech STE 320 / 530)
Многоосевая 5D-печать (оси X, Y, Z + наклон стола A и вращение B) назначается системой в следующих случаях:
- **Критическая доля нависаний:** если процент граней с углом нависания более 45° (`overhang_percent`) превышает 15% от площади поверхности. Поворот стола на угол A позволяет печатать вылеты без поддержек.
- **Силовые детали (`strength_priority = true`):** при наличии изгибающих, крутящих или разрывных нагрузок 5D-укладка ориентирует полимерные нити вдоль пространственных силовых линий детали, повышая прочность на межслойный сдвиг на 40–60%.
- **Труднообрабатываемые полимеры (`PA12-CF`, `PEEK`):** удаление поддержек на высокопрочных композитах и PEEK повреждает рабочие поверхности, поэтому для них приоритетен 5D-режим без поддержек.
- **Спиральные и трубчатые тела вращения:** импеллеры, шнеки, турбинные лопатки печатаются на непрерывном вращении оси B.

## 2. Критерии классической 3D-печати (Voron 2.4 / FDM)
- Плоские корпусные детали с большой площадью основания (`bed_detachment_risk = LOW`) и долей нависаний менее 15%.
- Сжимающие нагрузки, направленные строго перпендикулярно плоскости стола (вдоль оси Z).
- Быстрое черновое прототипирование габаритных макетов из базового PLA/PETG.

## 3. Специфика пространственной ориентации волокон при многоосевой 5D-печати
- В отличие от плоской послойной укладки (где ось Z является слабейшим звеном), 5D-печать укладывает непрерывные нити по изогнутым геодезическим траекториям.
- При печати углекомпозитом PA12-CF ориентация углеволокон вдоль траектории сопла дает прочность на разрыв до 110 МПа.
- Угол атаки сопла к поверхности поддерживается в диапазоне 75°–90° за счет согласованного движения осей A и B.

## 4. Правила статического аудита безопасности G-кода (verify_gcode_security)
Перед отправкой задания через `send_to_printer` каждый `.gcode` файл проходит обязательную проверку:
1. **Безопасная парковка (`G28`):** запрещены неконтролируемые обнуления осей без предварительного подъема по Z при наклоненном 5D-столе (риск соударения сопла с поворотным кулаком).
2. **Температурные лимиты:** температура сопла (`M104`/`M109`) не должна превышать `max_nozzle_temp` целевого принтера (300°C для 3D Voron, 420°C для 5D Stereotech Hybrid).
3. **Габаритный коридор:** координаты X, Y, Z и углы наклона A (-15°..+95°) не должны выходить за физическую зону печати.
4. **Контроль целостности:** вычисляется криптографический хэш `SHA-256`, который сверяется диспетчером принтера перед стартом нагрева.
"""


# =====================================================================
# 2. Knowledge Base Files Verification & Seeding
# =====================================================================
def ensure_knowledge_base_files(kb_dir: Optional[Path] = None) -> Tuple[Path, Path, Path]:
    """
    Ensures that the knowledge base directory and required files exist.
    If any file is missing or empty, seeds it with reference data.
    """
    target_dir = kb_dir or KB_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    filaments_path = target_dir / "filaments_rosatom.json"
    defects_path = target_dir / "defects_guide.json"
    kinematics_path = target_dir / "kinematics_3d_vs_5d.md"

    if not filaments_path.exists() or filaments_path.stat().st_size == 0:
        filaments_path.write_text(
            json.dumps(DEFAULT_FILAMENTS, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Записан справочник материалов Росатома: {filaments_path}")

    if not defects_path.exists() or defects_path.stat().st_size == 0:
        defects_path.write_text(
            json.dumps(DEFAULT_DEFECTS, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Записан справочник дефектов печати: {defects_path}")

    if not kinematics_path.exists() or kinematics_path.stat().st_size == 0:
        kinematics_path.write_text(DEFAULT_KINEMATICS_MD, encoding="utf-8")
        logger.info(f"Записан регламент 3D vs 5D и безопасности G-кода: {kinematics_path}")

    return filaments_path, defects_path, kinematics_path


# =====================================================================
# 3. Document Loaders & Chunking
# =====================================================================
def load_filaments(kb_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Loads filaments JSON and formats each filament as a RAG document
    with material properties, temperatures, speeds, kinematics, and notes.
    """
    filaments_path, _, _ = ensure_knowledge_base_files(kb_dir)
    try:
        data = json.loads(filaments_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Ошибка чтения {filaments_path} ({exc}), используются дефолтные филаменты.")
        data = DEFAULT_FILAMENTS

    docs: List[Dict[str, Any]] = []
    for item in data:
        mat_code = str(item.get("material", "UNKNOWN")).strip()
        full_name = item.get("full_name", mat_code)
        category = item.get("category", "engineering")
        kinematics = item.get("recommended_kinematics", ["3D", "5D"])
        kinematics_str = ", ".join(kinematics) if isinstance(kinematics, list) else str(kinematics)

        text_chunk = (
            f"Материал для 3D/5D печати: {mat_code} ({full_name}). "
            f"Категория материала: {category}. "
            f"Температурный регламент: температура сопла (экструдера) {item.get('nozzle_temp', 0)}°C, "
            f"температура печатного стола {item.get('bed_temp', 0)}°C, "
            f"температура термокамеры {item.get('chamber_temp', 0)}°C. "
            f"Скоростные и геометрические параметры: скорость печати {item.get('print_speed', 0)} мм/с, "
            f"базовая высота слоя {item.get('layer_height', 0.2)} мм, "
            f"плотность заполнения по умолчанию {item.get('infill_percent_default', 30)}% "
            f"(для силовых деталей: {item.get('infill_percent_strength', 70)}%), "
            f"мощность вентилятора обдува {item.get('cooling_fan_percent', 0)}%. "
            f"Риск термической усадки и отрыва от стола: {item.get('bed_detachment_risk', 'MEDIUM')}. "
            f"Рекомендуемая стратегия поддержек: {item.get('supports_strategy', 'AUTO')}. "
            f"Назначенная кинематика печати: {kinematics_str}. "
            f"Регламент предварительной сушки: {item.get('drying_regime', 'Не требуется')}. "
            f"Плотность: {item.get('density_g_cm3', 'н/д')} г/см³, предел прочности: {item.get('tensile_strength_mpa', 'н/д')} МПа. "
            f"Технологические нюансы и применение: {item.get('notes', '')}"
        )

        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"filament_{mat_code.lower()}"))
        docs.append(
            {
                "id": doc_id,
                "doc_type": "filament",
                "collection": COLLECTION_FILAMENTS,
                "key": mat_code,
                "text": text_chunk,
                "metadata": {
                    **item,
                    "doc_type": "filament",
                    "key": mat_code,
                    "collection": COLLECTION_FILAMENTS,
                    "source": filaments_path.name,
                },
            }
        )

    logger.debug(f"Сформировано {len(docs)} документов филаментов для коллекции '{COLLECTION_FILAMENTS}'.")
    return docs


def load_defects(kb_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Loads defects JSON and formats each defect as a RAG document
    with defect type, symptoms, causes, fixes, and parameter adjustments.
    """
    _, defects_path, _ = ensure_knowledge_base_files(kb_dir)
    try:
        data = json.loads(defects_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Ошибка чтения {defects_path} ({exc}), используются дефолтные дефекты.")
        data = DEFAULT_DEFECTS

    docs: List[Dict[str, Any]] = []
    for item in data:
        defect_type = str(item.get("defect_type", "unknown")).strip()
        title = item.get("title", defect_type)

        symptoms = item.get("symptoms") or item.get("observed_symptoms", "")
        causes = item.get("causes") or item.get("root_causes", [])
        fixes = item.get("fixes") or item.get("operator_actions", [])
        adjustments = item.get("parameter_adjustments", {})

        causes_str = "; ".join(causes) if isinstance(causes, list) else str(causes)
        fixes_str = "; ".join(fixes) if isinstance(fixes, list) else str(fixes)
        adjustments_str = (
            json.dumps(adjustments, ensure_ascii=False) if adjustments else "нет изменений"
        )

        text_chunk = (
            f"Дефект 3D/5D-печати: {defect_type} — {title}. "
            f"Внешние симптомы: {symptoms} "
            f"Первопричины возникновения: {causes_str}. "
            f"Действия оператора и регламент устранения: {fixes_str}. "
            f"Корректировка технологических параметров слайсинга: {adjustments_str}."
        )

        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"defect_{defect_type.lower()}"))
        docs.append(
            {
                "id": doc_id,
                "doc_type": "defect",
                "collection": COLLECTION_DEFECTS,
                "key": defect_type,
                "text": text_chunk,
                "metadata": {
                    **item,
                    "symptoms": symptoms,
                    "observed_symptoms": symptoms,
                    "causes": causes,
                    "root_causes": causes,
                    "fixes": fixes,
                    "operator_actions": fixes,
                    "parameter_adjustments": adjustments,
                    "doc_type": "defect",
                    "key": defect_type,
                    "collection": COLLECTION_DEFECTS,
                    "source": defects_path.name,
                },
            }
        )

    logger.debug(f"Сформировано {len(docs)} документов дефектов для коллекции '{COLLECTION_DEFECTS}'.")
    return docs


def load_kinematics(kb_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Loads kinematics markdown and splits it into section documents based on '## ' headers.
    """
    _, _, kinematics_path = ensure_knowledge_base_files(kb_dir)
    try:
        md_text = kinematics_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning(f"Ошибка чтения {kinematics_path} ({exc}), используется дефолтный markdown.")
        md_text = DEFAULT_KINEMATICS_MD

    raw_sections = [s.strip() for s in md_text.split("## ") if s.strip()]
    docs: List[Dict[str, Any]] = []

    for idx, sec in enumerate(raw_sections, start=1):
        lines = sec.splitlines()
        title_line = lines[0].strip().lstrip("# ").strip()
        body = "\n".join(lines[1:]).strip()
        formatted_text = f"## {title_line}\n\n{body}" if body else f"## {title_line}"

        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"kinematics_sec_{idx}"))
        docs.append(
            {
                "id": doc_id,
                "doc_type": "kinematics",
                "collection": COLLECTION_KINEMATICS,
                "key": title_line,
                "text": formatted_text,
                "metadata": {
                    "section_title": title_line,
                    "section_index": idx,
                    "doc_type": "kinematics",
                    "key": title_line,
                    "collection": COLLECTION_KINEMATICS,
                    "source": kinematics_path.name,
                },
            }
        )

    logger.debug(f"Сформировано {len(docs)} секций кинематики для коллекции '{COLLECTION_KINEMATICS}'.")
    return docs


def build_documents(kb_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Combines all documents (filaments, defects, kinematics) into a single list
    ready for embedding and indexing.
    """
    filaments = load_filaments(kb_dir)
    defects = load_defects(kb_dir)
    kinematics = load_kinematics(kb_dir)

    all_docs = filaments + defects + kinematics
    logger.info(
        f"Подготовлено {len(all_docs)} документов для RAG: "
        f"{len(filaments)} материалов, {len(defects)} дефектов, {len(kinematics)} секций кинематики."
    )
    return all_docs


# =====================================================================
# 4. Embeddings Generation (FastEmbed with Deterministic Fallback)
# =====================================================================
_embedder_instance: Optional[TextEmbedding] = None


def get_embedder(model_name: str = EMBEDDING_MODEL_NAME) -> TextEmbedding:
    """Returns a singleton FastEmbed TextEmbedding instance."""
    global _embedder_instance
    if _embedder_instance is None or getattr(_embedder_instance, "model_name", None) != model_name:
        logger.info(f"Инициализация FastEmbed (модель: {model_name})...")
        _embedder_instance = TextEmbedding(model_name=model_name)
    return _embedder_instance


def compute_embeddings(
    texts: List[str],
    model_name: str = EMBEDDING_MODEL_NAME,
) -> List[List[float]]:
    """
    Computes dense vector representations for input texts using FastEmbed.
    Falls back to a deterministic hash-based embedding in case FastEmbed is unavailable.
    """
    if not texts:
        return []

    try:
        embedder = get_embedder(model_name)
        raw_vectors = list(embedder.embed(texts))
        vectors = [vec.tolist() for vec in raw_vectors]
        logger.info(f"Сгенерировано {len(vectors)} эмбеддингов размерности {len(vectors[0])} через FastEmbed.")
        return vectors
    except Exception as exc:
        logger.warning(
            f"FastEmbed недоступен ({exc}). Включается детерминированный оффлайн-генератор (dim={VECTOR_SIZE})."
        )
        fallback_vectors: List[List[float]] = []
        for text in texts:
            vec: List[float] = []
            for i in range(VECTOR_SIZE):
                h = hashlib.sha256(f"{i}:{text.lower()}".encode("utf-8")).digest()
                val = (int.from_bytes(h[:4], "big") / 0xFFFFFFFF) * 2.0 - 1.0
                vec.append(round(val, 6))
            fallback_vectors.append(vec)
        return fallback_vectors


# =====================================================================
# 5. Qdrant Client Helpers (Sync & Async)
# =====================================================================
def get_qdrant_client(url: Optional[str] = None) -> QdrantClient:
    """
    Returns a synchronous QdrantClient.
    Attempts HTTP connection to `url` or `QDRANT_URL`, falls back to local on-disk storage.
    """
    target_url = url or DEFAULT_QDRANT_URL
    try:
        client = QdrantClient(url=target_url, timeout=2.0)
        client.get_collections()
        logger.info(f"Синхронное подключение к серверу Qdrant: {target_url}")
        return client
    except Exception as exc:
        LOCAL_QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        logger.warning(
            f"Сервер Qdrant ({target_url}) недоступен ({exc}). "
            f"Переключение на локальную БД: {LOCAL_QDRANT_PATH}"
        )
        return QdrantClient(path=str(LOCAL_QDRANT_PATH))


def get_async_qdrant_client(url: Optional[str] = None) -> AsyncQdrantClient:
    """
    Returns an AsyncQdrantClient.
    Attempts HTTP connection; if unreachable, creates a disk-backed local client.
    """
    target_url = url or DEFAULT_QDRANT_URL
    try:
        # Check connectivity using a fast sync ping
        sync_probe = QdrantClient(url=target_url, timeout=1.5)
        sync_probe.get_collections()
        sync_probe.close()
        logger.info(f"Асинхронное подключение к серверу Qdrant: {target_url}")
        return AsyncQdrantClient(url=target_url, timeout=10.0)
    except Exception as exc:
        LOCAL_QDRANT_PATH.mkdir(parents=True, exist_ok=True)
        logger.warning(
            f"Сервер Qdrant ({target_url}) недоступен ({exc}). "
            f"Используется локальное асинхронное хранилище: {LOCAL_QDRANT_PATH}"
        )
        return AsyncQdrantClient(path=str(LOCAL_QDRANT_PATH))


# =====================================================================
# 6. Core Asynchronous Ingestion to Qdrant
# =====================================================================
async def ingest_to_qdrant(
    qdrant_url: Optional[str] = None,
    recreate: bool = False,
    skip_existing: bool = False,
    kb_dir: Optional[Path] = None,
    client: Optional[AsyncQdrantClient] = None,
) -> Dict[str, Any]:
    """
    Main asynchronous pipeline:
    1. Loads filaments, defects, and kinematics files.
    2. Chunks and prepares documents.
    3. Computes vector embeddings using FastEmbed.
    4. Connects to Qdrant.
    5. Manages collections ('filaments', 'defects', 'kinematics', and unified 'rosatom_3d_5d_kb').
       Handles existing collections (recreates if `recreate=True`, skips if `skip_existing=True`).
    6. Upserts points into each collection.
    7. Returns indexing summary.
    """
    target_url = qdrant_url or DEFAULT_QDRANT_URL
    logger.info(
        f"Начало индексации базы знаний в Qdrant (URL: {target_url}, recreate={recreate}, skip_existing={skip_existing})"
    )

    # 1. Load documents
    docs = build_documents(kb_dir=kb_dir)
    if not docs:
        logger.warning("База знаний пуста, нечего индексировать.")
        return {"status": "empty", "total_documents": 0, "collections": {}}

    # 2. Compute embeddings
    texts = [d["text"] for d in docs]
    vectors = compute_embeddings(texts, model_name=EMBEDDING_MODEL_NAME)
    vector_dim = len(vectors[0]) if vectors else VECTOR_SIZE

    # Map each document to its computed vector
    doc_vector_map = {doc["id"]: vec for doc, vec in zip(docs, vectors)}

    # Group documents by target collection
    collection_docs: Dict[str, List[Dict[str, Any]]] = {
        COLLECTION_FILAMENTS: [d for d in docs if d["doc_type"] == "filament"],
        COLLECTION_DEFECTS: [d for d in docs if d["doc_type"] == "defect"],
        COLLECTION_KINEMATICS: [d for d in docs if d["doc_type"] == "kinematics"],
        COLLECTION_NAME: docs,  # Unified collection containing everything
    }

    # 3. Obtain async client
    owns_client = client is None
    q_client = client or get_async_qdrant_client(url=target_url)

    results_summary: Dict[str, Any] = {}

    try:
        # Get existing collections
        cols_response = await q_client.get_collections()
        existing_names = {c.name for c in cols_response.collections}
        logger.info(f"Существующие коллекции в Qdrant: {list(existing_names)}")

        for col_name, items in collection_docs.items():
            if not items:
                continue

            exists = col_name in existing_names

            # Handle existing collection logic
            if exists:
                if recreate:
                    logger.info(f"Коллекция '{col_name}' уже существует. Удаление для пересоздания...")
                    await q_client.delete_collection(collection_name=col_name)
                    exists = False
                elif skip_existing:
                    logger.info(f"Коллекция '{col_name}' уже существует. Пропуск индексации (--skip-existing).")
                    results_summary[col_name] = {
                        "status": "skipped",
                        "points_upserted": 0,
                        "total_items": len(items),
                    }
                    continue
                else:
                    logger.info(f"Коллекция '{col_name}' уже существует. Выполняется обновление (upsert)...")

            # Create collection if it doesn't exist
            if not exists:
                logger.info(f"Создание коллекции Qdrant '{col_name}' (размерность={vector_dim}, метрика=COSINE)...")
                await q_client.create_collection(
                    collection_name=col_name,
                    vectors_config=rest.VectorParams(
                        size=vector_dim,
                        distance=rest.Distance.COSINE,
                    ),
                )
                existing_names.add(col_name)

            # Build PointStruct objects
            points = [
                rest.PointStruct(
                    id=item["id"],
                    vector=doc_vector_map[item["id"]],
                    payload={
                        "doc_type": item["doc_type"],
                        "key": item["key"],
                        "text": item["text"],
                        "collection": col_name,
                        "metadata": item["metadata"],
                        **item["metadata"],
                    },
                )
                for item in items
            ]

            # Upsert into Qdrant
            await q_client.upsert(
                collection_name=col_name,
                points=points,
            )
            logger.success(f"Коллекция '{col_name}': успешно сохранено {len(points)} векторов.")

            results_summary[col_name] = {
                "status": "upserted",
                "points_upserted": len(points),
                "total_items": len(items),
            }

    finally:
        if owns_client:
            await q_client.close()

    summary: Dict[str, Any] = {
        "status": "success",
        "total_documents": len(docs),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vector_size": vector_dim,
        "collections": results_summary,
    }

    logger.info("Индексация базы знаний Росатома завершена успешно.")
    return summary


# =====================================================================
# 7. Synchronous Compatibility Wrapper
# =====================================================================
def ingest_knowledge_base(
    recreate: bool = False,
    qdrant_url: Optional[str] = None,
    skip_existing: bool = False,
    kb_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for ingest_to_qdrant.
    Safe to call from synchronous code, Celery tasks, or active asyncio event loops.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Running inside an existing event loop — delegate to worker thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                asyncio.run,
                ingest_to_qdrant(
                    qdrant_url=qdrant_url,
                    recreate=recreate,
                    skip_existing=skip_existing,
                    kb_dir=kb_dir,
                ),
            )
            return future.result()
    else:
        return asyncio.run(
            ingest_to_qdrant(
                qdrant_url=qdrant_url,
                recreate=recreate,
                skip_existing=skip_existing,
                kb_dir=kb_dir,
            )
        )


# =====================================================================
# 8. Command-Line Interface (CLI Entry Point)
# =====================================================================
def main() -> None:
    """CLI entry point for ingesting knowledge base into Qdrant."""
    parser = argparse.ArgumentParser(
        description="Индексация базы знаний Росатома 3D/5D печати в векторную БД Qdrant."
    )
    parser.add_argument(
        "--url",
        "-u",
        default=DEFAULT_QDRANT_URL,
        help=f"URL сервера Qdrant (по умолчанию: {DEFAULT_QDRANT_URL})",
    )
    parser.add_argument(
        "--recreate",
        "-r",
        action="store_true",
        help="Удалить существующие коллекции и создать заново",
    )
    parser.add_argument(
        "--skip-existing",
        "-s",
        action="store_true",
        help="Пропускать существующие коллекции без изменения",
    )
    parser.add_argument(
        "--kb-dir",
        default=None,
        help="Путь к директории knowledge_base (по умолчанию: orchestrator/rag/knowledge_base)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=EMBEDDING_MODEL_NAME,
        help=f"Модель FastEmbed для эмбеддингов (по умолчанию: {EMBEDDING_MODEL_NAME})",
    )

    args = parser.parse_args()

    custom_kb = Path(args.kb_dir).resolve() if args.kb_dir else None

    logger.info("Запуск индексатора базы знаний Росатома...")
    summary = asyncio.run(
        ingest_to_qdrant(
            qdrant_url=args.url,
            recreate=args.recreate,
            skip_existing=args.skip_existing,
            kb_dir=custom_kb,
        )
    )

    print("\n" + "=" * 50)
    print("ИТОГИ ИНДЕКСАЦИИ БАЗЫ ЗНАНИЙ РОСАТОМА (QDRANT):")
    print("=" * 50)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()