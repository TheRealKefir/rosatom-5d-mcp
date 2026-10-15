import json
import sys
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

# Добавляем корень монорепозитория в sys.path для запуска файла напрямую
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from orchestrator.rag.ingest import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    build_documents,
    compute_embeddings,
    ensure_knowledge_base_files,
    get_qdrant_client,
    ingest_knowledge_base,
)


class RosatomKnowledgeRetriever:
    """
    Ретривер базы знаний 3D/5D-печати Росатома (Qdrant + FastEmbed + In-Memory Fallback).
    """

    def __init__(self) -> None:
        self._client: Optional[QdrantClient] = None
        self._embedder: Any = None
        self._in_memory_docs: list[dict[str, Any]] = []
        self._init_storage()

    def _init_storage(self) -> None:
        """Гарантирует наличие файлов БЗ и подключается к Qdrant (с резервной копией в памяти)."""
        ensure_knowledge_base_files()
        self._in_memory_docs = build_documents()

        try:
            self._client = get_qdrant_client()
            collections = [c.name for c in self._client.get_collections().collections]
            if COLLECTION_NAME not in collections:
                logger.info(f"Коллекция '{COLLECTION_NAME}' не найдена. Запускаю авто-индексацию...")
                self._client.close()
                ingest_knowledge_base(recreate=False)
                self._client = get_qdrant_client()
        except Exception as exc:
            logger.warning(
                f"Qdrant временно заблокирован другим процессом ({exc}). "
                "Активен быстрый In-Memory поиск по справочникам Росатома."
            )
            self._client = None

    def _encode_query(self, query: str) -> list[float]:
        """Кодирует поисковый запрос в вектор размерности 384."""
        if self._embedder is None:
            try:
                from fastembed import TextEmbedding
                self._embedder = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
            except Exception:
                self._embedder = False

        if self._embedder:
            try:
                return list(self._embedder.embed([query]))[0].tolist()
            except Exception as exc:
                logger.warning(f"Ошибка FastEmbed при кодировании запроса ({exc}), используем fallback.")

        return compute_embeddings([query])[0]

    # =====================================================================
    # 1. Универсальный векторный поиск с фильтрацией по типу документа
    # =====================================================================
    def search(
        self,
        query: str,
        doc_type: Optional[str] = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Ищет релевантные документы в Qdrant.
        doc_type может быть: 'filament', 'defect', 'kinematics_and_security' или None (искать везде).
        """
        query_vector = self._encode_query(query)

        if self._client is not None:
            try:
                query_filter = None
                if doc_type:
                    query_filter = rest.Filter(
                        must=[
                            rest.FieldCondition(
                                key="doc_type",
                                match=rest.MatchValue(value=doc_type),
                            )
                        ]
                    )

                # Поддержка нового API query_points (qdrant-client >= 1.10) и классического search
                if hasattr(self._client, "query_points"):
                    response = self._client.query_points(
                        collection_name=COLLECTION_NAME,
                        query=query_vector,
                        query_filter=query_filter,
                        limit=top_k,
                        with_payload=True,
                    )
                    hits = response.points
                else:
                    hits = self._client.search(
                        collection_name=COLLECTION_NAME,
                        query_vector=query_vector,
                        query_filter=query_filter,
                        limit=top_k,
                        with_payload=True,
                    )

                results: list[dict[str, Any]] = []
                for hit in hits:
                    payload = hit.payload or {}
                    results.append(
                        {
                            "score": round(float(hit.score), 4),
                            "doc_type": payload.get("doc_type"),
                            "key": payload.get("key"),
                            "text": payload.get("text", ""),
                            "metadata": payload.get("metadata", {}),
                        }
                    )
                if results:
                    return results
            except Exception as exc:
                logger.warning(f"Ошибка векторного поиска Qdrant ({exc}), переключение на In-Memory поиск.")

        # Fallback: поиск по документам в памяти
        return self._fallback_in_memory_search(query=query, doc_type=doc_type, top_k=top_k)

    def _fallback_in_memory_search(
        self,
        query: str,
        doc_type: Optional[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Резервный поиск по ключевым словам и подстрокам, если файловый Qdrant занят."""
        candidates = [
            d for d in self._in_memory_docs if (doc_type is None or d["doc_type"] == doc_type)
        ]
        q_words = [w.lower() for w in query.split() if len(w) > 2]

        scored: list[tuple[float, dict[str, Any]]] = []
        for doc in candidates:
            haystack = f"{doc['key']} {doc['text']}".lower()
            score = sum(1.0 for w in q_words if w in haystack)
            if doc["key"].lower() in query.lower():
                score += 5.0
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "score": round(s, 2),
                "doc_type": d["doc_type"],
                "key": d["key"],
                "text": d["text"],
                "metadata": d["metadata"],
            }
            for s, d in scored[:top_k]
        ]

    # =====================================================================
    # 2. Специализированный метод под тул `recommend_print_parameters`
    # =====================================================================
    def get_filament_profile(
        self,
        material: str,
        strength_priority: bool = False,
    ) -> dict[str, Any]:
        """
        Возвращает точный технологический профиль материала из базы Росатома.
        Если точное имя не найдено, ищет ближайший аналог через векторный поиск.
        """
        mat_clean = material.strip().upper()

        # 1. Сначала проверяем точное совпадение по коду материала (PLA, ABS, PETG, PA12-CF, PEEK)
        exact_meta: Optional[dict[str, Any]] = None
        for doc in self._in_memory_docs:
            if doc["doc_type"] == "filament" and doc["key"].upper() == mat_clean:
                exact_meta = doc["metadata"]
                break

        # 2. Если точного совпадения нет (например, запросили "карбоновый нейлон"), ищем семантически
        if exact_meta is None:
            hits = self.search(query=f"Материал пластик {material}", doc_type="filament", top_k=1)
            if hits and hits[0].get("metadata"):
                exact_meta = hits[0]["metadata"]

        if exact_meta is None:
            # Дефолтный профиль PA12-CF на случай неизвестного полимера
            exact_meta = self._in_memory_docs[0]["metadata"]

        infill = (
            exact_meta.get("infill_percent_strength", 80)
            if strength_priority
            else exact_meta.get("infill_percent_default", 40)
        )
        base_speed = int(exact_meta.get("print_speed", 50))
        layer_h = float(exact_meta.get("layer_height", 0.2))

        return {
            "material": exact_meta.get("material", mat_clean),
            "full_name": exact_meta.get("full_name", ""),
            "nozzle_temp": exact_meta.get("nozzle_temp", 250),
            "bed_temp": exact_meta.get("bed_temp", 90),
            "chamber_temp": exact_meta.get("chamber_temp", 45),
            "print_speed": max(25, base_speed - 10) if strength_priority else base_speed,
            "infill_percent": infill,
            "layer_height": round(max(0.12, layer_h - 0.05), 2) if strength_priority else layer_h,
            "cooling_fan_percent": exact_meta.get("cooling_fan_percent", 0),
            "supports_strategy": exact_meta.get("supports_strategy", "NONE_5D_MULTIAXIS"),
            "recommended_kinematics": exact_meta.get("recommended_kinematics", ["5D"]),
            "drying_regime": exact_meta.get("drying_regime", ""),
            "notes": exact_meta.get("notes", ""),
        }

    # =====================================================================
    # 3. Специализированный метод под тул `diagnose_print_defect_or_spool`
    # =====================================================================
    def get_defect_solution(
        self,
        defect_type: str,
        observed_symptoms: str = "",
        current_material: str = "ABS",
    ) -> dict[str, Any]:
        """
        Находит регламент устранения дефекта печати или проблемы с катушкой
        и адаптирует его под текущий материал.
        """
        defect_clean = defect_type.strip().lower()
        matched_defect: Optional[dict[str, Any]] = None

        # 1. Точное совпадение по коду дефекта (stringing, warping, layer_shift, underextrusion)
        for doc in self._in_memory_docs:
            if doc["doc_type"] == "defect" and doc["key"].lower() == defect_clean:
                matched_defect = doc["metadata"]
                break

        # 2. Семантический поиск по описанию симптомов
        if matched_defect is None:
            search_query = f"{defect_type} {observed_symptoms}".strip()
            hits = self.search(query=search_query, doc_type="defect", top_k=1)
            if hits and hits[0].get("metadata"):
                matched_defect = hits[0]["metadata"]

        # Подтягиваем также профиль текущего материала, чтобы дать точный режим сушки катушки
        mat_profile = self.get_filament_profile(current_material)

        if matched_defect is None:
            return {
                "defect_type": defect_type,
                "current_material": mat_profile["material"],
                "parameter_adjustments": {"print_speed_delta_percent": -15},
                "operator_actions": [
                    f"Просушить катушку ({mat_profile.get('drying_regime', '70°C 4 часа')})",
                    "Проверить калибровку первого слоя и чистоту сопла",
                ],
            }

        actions = list(matched_defect.get("operator_actions", []))
        if mat_profile.get("drying_regime"):
            actions.append(
                f"Режим сушки катушки для {mat_profile['material']}: {mat_profile['drying_regime']}"
            )

        return {
            "defect_type": matched_defect.get("defect_type", defect_clean),
            "title": matched_defect.get("title", ""),
            "current_material": mat_profile["material"],
            "root_causes": matched_defect.get("root_causes", []),
            "parameter_adjustments": matched_defect.get("parameter_adjustments", {}),
            "operator_actions": actions,
        }

    # =====================================================================
    # 4. Сборка контекста для промпта LLM-оркестратора
    # =====================================================================
    def build_rag_context(
        self,
        user_query: str,
        material: Optional[str] = None,
        defect_type: Optional[str] = None,
    ) -> str:
        """
        Формирует структурированную выжимку из базы знаний Росатома
        для подачи в системный промпт LLM перед вызовом инструментов MCP.
        """
        sections: list[str] = ["### КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ РОСАТОМА (RAG):"]

        if material:
            prof = self.get_filament_profile(material)
            sections.append(
                f"- **Профиль материала `{prof['material']}` ({prof['full_name']}):** "
                f"сопло {prof['nozzle_temp']}°C, стол {prof['bed_temp']}°C, камера {prof['chamber_temp']}°C, "
                f"скорость {prof['print_speed']} мм/с, слой {prof['layer_height']} мм, "
                f"поддержки `{prof['supports_strategy']}`. Сушка: {prof['drying_regime']}."
            )

        if defect_type:
            sol = self.get_defect_solution(defect_type, user_query, material or "ABS")
            sections.append(
                f"- **Регламент по дефекту `{sol['defect_type']}` ({sol.get('title', '')}):** "
                f"корректировки={json.dumps(sol['parameter_adjustments'], ensure_ascii=False)}; "
                f"действия оператора={'; '.join(sol['operator_actions'])}."
            )

        general_hits = self.search(query=user_query, top_k=2)
        for idx, hit in enumerate(general_hits, start=1):
            sections.append(f"- **Справка #{idx} [{hit['doc_type']} / {hit['key']}]:** {hit['text']}")

        return "\n".join(sections)


# Глобальный Singleton-экземпляр, чтобы не открывать несколько соединений к локальному Qdrant
_retriever_instance: Optional[RosatomKnowledgeRetriever] = None


def get_retriever() -> RosatomKnowledgeRetriever:
    """Возвращает инициализированный Singleton-ретривер для использования в main.py и agent/router.py."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RosatomKnowledgeRetriever()
    return _retriever_instance


# =====================================================================
# Самопроверка модуля при запуске из консоли
# =====================================================================
if __name__ == "__main__":
    r = get_retriever()

    print("\n1. Тест подбора профиля материала (PA12-CF, силовой режим):")
    print(json.dumps(r.get_filament_profile("PA12-CF", strength_priority=True), ensure_ascii=False, indent=2))

    print("\n2. Тест диагностики дефекта по симптомам:")
    print(
        json.dumps(
            r.get_defect_solution(
                defect_type="warping",
                observed_symptoms="Отрывает углы детали от стола и пошли трещины между слоями",
                current_material="ABS",
            ),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n3. Тест сборки RAG-контекста для LLM:")
    print(r.build_rag_context("Нужно напечатать кронштейн с большими нависаниями без поддержек", material="PA12-CF"))