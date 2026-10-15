"""
orchestrator/rag/retriever.py

RAG retriever module for querying the Qdrant vector database.
Retrieves relevant engineering knowledge on 3D/5D printing filaments and defect solutions.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

try:
    from fastembed import TextEmbedding
except ImportError:  # pragma: no cover
    TextEmbedding = None  # type: ignore

# Add root of monorepo to sys.path for direct module execution
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from orchestrator.rag.ingest import (
    COLLECTION_DEFECTS,
    COLLECTION_FILAMENTS,
    COLLECTION_KINEMATICS,
    COLLECTION_NAME,
    DEFAULT_DEFECTS,
    DEFAULT_FILAMENTS,
    EMBEDDING_MODEL_NAME,
    VECTOR_SIZE,
    build_documents,
    ensure_knowledge_base_files,
)


class RAGRetriever:
    """
    Asynchronous RAG retriever for Rosatom 3D/5D additive manufacturing knowledge base.
    Embeds queries using FastEmbed and retrieves matching records from Qdrant vector store.
    """

    DEFAULT_COLLECTIONS: list[str] = [COLLECTION_FILAMENTS, COLLECTION_DEFECTS]

    def __init__(
        self,
        qdrant_url: str = "http://qdrant:6333",
        embedding_model: str = EMBEDDING_MODEL_NAME,
        timeout: float = 3.0,
        client: Optional[AsyncQdrantClient] = None,
    ) -> None:
        """
        Initialize the RAG retriever.

        Args:
            qdrant_url: URL or local path to Qdrant instance. Defaults to 'http://qdrant:6333'.
            embedding_model: HuggingFace model name for FastEmbed query embedding.
            timeout: Network request timeout in seconds for Qdrant client.
            client: Optional pre-configured AsyncQdrantClient instance (useful for testing/mocking).
        """
        self.qdrant_url = qdrant_url
        self.embedding_model_name = embedding_model
        self.timeout = timeout

        # 1. Initialize AsyncQdrantClient
        if client is not None:
            self.client = client
        elif qdrant_url == ":memory:":
            self.client = AsyncQdrantClient(location=":memory:")
        elif os.path.exists(qdrant_url) and os.path.isdir(qdrant_url):
            self.client = AsyncQdrantClient(path=qdrant_url)
        else:
            self.client = AsyncQdrantClient(url=qdrant_url, timeout=timeout)

        # 2. Initialize FastEmbed TextEmbedding
        self._embedder: Optional[TextEmbedding] = None
        self._init_embedder()

        # In-memory documents fallback for exact lookup helpers
        self._in_memory_docs: list[dict[str, Any]] = []
        try:
            ensure_knowledge_base_files()
            self._in_memory_docs = build_documents()
        except Exception as exc:
            logger.debug(f"Knowledge base directory initialization note: {exc}")

    def _init_embedder(self) -> None:
        """Load the FastEmbed TextEmbedding model, or configure deterministic fallback on error."""
        if TextEmbedding is None:
            logger.warning("fastembed is not installed. Deterministic hash embedding will be used as fallback.")
            self._embedder = None
            return

        try:
            logger.info(f"Loading FastEmbed model: {self.embedding_model_name}")
            self._embedder = TextEmbedding(model_name=self.embedding_model_name)
            logger.success(f"FastEmbed model '{self.embedding_model_name}' loaded successfully.")
        except Exception as exc:
            logger.warning(
                f"Failed to load FastEmbed model '{self.embedding_model_name}' ({exc}). "
                "Deterministic hash embedding will be used as fallback."
            )
            self._embedder = None

    def _fallback_hash_embedding(self, text: str) -> list[float]:
        """Deterministic fallback embedding generator matching ingest.py."""
        dim = VECTOR_SIZE
        vec: list[float] = []
        text_lower = text.lower()
        for i in range(dim):
            h = hashlib.sha256(f"{i}:{text_lower}".encode("utf-8")).digest()
            val = (int.from_bytes(h[:4], "big") / 0xFFFFFFFF) * 2.0 - 1.0
            vec.append(round(val, 6))
        return vec

    def _encode_query(self, query: str) -> list[float]:
        """Encode a text query into a 384-dimensional dense vector."""
        if self._embedder is not None:
            try:
                embeddings = list(self._embedder.embed([query]))
                if embeddings:
                    return embeddings[0].tolist()
            except Exception as exc:
                logger.warning(f"FastEmbed error during encoding ({exc}), falling back to hash embedding.")

        return self._fallback_hash_embedding(query)

    async def _search_single_collection(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """
        Query a single Qdrant collection, with automatic fallback to unified KB collection.
        Handles connection and collection errors gracefully.
        """
        try:
            # Query the requested collection
            points = await self._execute_query_points(
                collection_name=collection_name,
                query_vector=query_vector,
                top_k=top_k,
            )
            if points is not None:
                return self._format_points(points, collection_name)
        except (ResponseHandlingException, OSError, TimeoutError, ConnectionError) as exc:
            logger.warning(f"Qdrant connection error while querying '{collection_name}': {exc}")
            return []
        except Exception as exc:
            logger.debug(f"Direct query on collection '{collection_name}' failed: {exc}. Attempting fallback...")

        # If collection doesn't exist directly, check fallback in unified COLLECTION_NAME (e.g. rosatom_3d_5d_kb)
        try:
            fallback_points = await self._query_unified_kb_fallback(
                target_collection=collection_name,
                query_vector=query_vector,
                top_k=top_k,
            )
            if fallback_points:
                return self._format_points(fallback_points, collection_name)
        except (ResponseHandlingException, OSError, TimeoutError, ConnectionError) as exc:
            logger.warning(f"Qdrant connection error during fallback query: {exc}")
            return []
        except Exception as exc:
            logger.debug(f"Fallback query on unified collection failed: {exc}")

        return []

    async def _execute_query_points(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int,
        query_filter: Optional[rest.Filter] = None,
    ) -> Optional[list[Any]]:
        """Execute query_points or search on client."""
        if hasattr(self.client, "query_points"):
            response = await self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            return response.points
        elif hasattr(self.client, "search"):
            return await self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        return None

    async def _query_unified_kb_fallback(
        self,
        target_collection: str,
        query_vector: list[float],
        top_k: int,
    ) -> list[Any]:
        """Query unified COLLECTION_NAME with doc_type filter if specific collection is absent."""
        doc_type_filter: Optional[str] = None
        col_lower = target_collection.lower()
        if "filament" in col_lower:
            doc_type_filter = "filament"
        elif "defect" in col_lower:
            doc_type_filter = "defect"
        elif "kinematic" in col_lower or "security" in col_lower:
            doc_type_filter = "kinematics_and_security"

        qf = None
        if doc_type_filter:
            qf = rest.Filter(
                must=[rest.FieldCondition(key="doc_type", match=rest.MatchValue(value=doc_type_filter))]
            )

        points = await self._execute_query_points(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            top_k=top_k,
            query_filter=qf,
        )
        return points or []

    def _format_points(self, points: list[Any], collection_name: str) -> list[dict[str, Any]]:
        """Format Qdrant points into structured dicts with score, payload, and text."""
        results: list[dict[str, Any]] = []
        for p in points:
            payload = p.payload or {}
            score = round(float(p.score), 4) if getattr(p, "score", None) is not None else 0.0
            results.append({
                "id": p.id,
                "score": score,
                "collection": collection_name,
                "payload": payload,
                "text": payload.get("text", ""),
                "metadata": payload.get("metadata", {}),
            })
        return results

    async def search_filaments(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Search the 'filaments' collection for materials, printing regimes, and drying guidelines.

        Args:
            query: Natural language query (e.g. 'высокотемпературный пластик для 5D').
            top_k: Number of top documents to return.

        Returns:
            List of structured dicts with score, payload, text, and collection metadata.
        """
        return await self.search(query=query, collections=[COLLECTION_FILAMENTS], top_k=top_k)

    async def search_defects(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Search the 'defects' collection for print failures, root causes, and troubleshooting actions.

        Args:
            query: Natural language query (e.g. 'паутина между деталями', 'отрыв углов от стола').
            top_k: Number of top documents to return.

        Returns:
            List of structured dicts with score, payload, text, and collection metadata.
        """
        return await self.search(query=query, collections=[COLLECTION_DEFECTS], top_k=top_k)

    async def search(
        self,
        query: str,
        collections: list[str] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Unified semantic search across specified collections or all standard knowledge collections.

        Args:
            query: Natural language query string.
            collections: List of collection names to search. Defaults to ['filaments', 'defects'].
            top_k: Total number of most relevant documents to return across all collections.

        Returns:
            List of structured result dicts sorted by relevance score in descending order.
        """
        clean_query = query.strip() if query else ""
        if not clean_query:
            return []

        target_collections = collections if collections is not None else self.DEFAULT_COLLECTIONS
        if isinstance(target_collections, str):
            target_collections = [target_collections]

        try:
            query_vector = await asyncio.to_thread(self._encode_query, clean_query)
        except Exception as exc:
            logger.warning(f"Error encoding query '{clean_query}': {exc}")
            return []

        # Execute searches across all target collections concurrently
        tasks = [
            self._search_single_collection(col, query_vector=query_vector, top_k=top_k)
            for col in target_collections
        ]
        results_nested = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: list[dict[str, Any]] = []
        for res in results_nested:
            if isinstance(res, list):
                all_results.extend(res)
            elif isinstance(res, Exception):
                logger.warning(f"Collection search task encountered an error: {res}")

        # Sort combined results by similarity score descending
        all_results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return all_results[:top_k]

    def format_context(self, results: list[dict[str, Any]]) -> str:
        """
        Format retrieved documents into a clean Markdown context string for LLM prompt injection.

        Args:
            results: List of structured search result dicts.

        Returns:
            Formatted string ready for inclusion in system or user prompt.
        """
        if not results:
            return ""

        doc_blocks: list[str] = []
        for idx, item in enumerate(results, start=1):
            score = item.get("score", 0.0)
            collection = item.get("collection", "knowledge_base")
            payload = item.get("payload", {})
            text = item.get("text") or payload.get("text")

            header = f"[{idx}] Справка из базы знаний: `{collection}` (релевантность: {score:.2f})"

            if text and str(text).strip():
                content = str(text).strip()
            else:
                lines: list[str] = []
                for k, v in payload.items():
                    if k in ("id", "vector", "doc_type"):
                        continue
                    if isinstance(v, (dict, list)):
                        lines.append(f"- **{k}**: {json.dumps(v, ensure_ascii=False)}")
                    else:
                        lines.append(f"- **{k}**: {v}")
                content = "\n".join(lines) if lines else json.dumps(payload, ensure_ascii=False)

            doc_blocks.append(f"{header}\n{content}")

        return "\n\n".join(doc_blocks)

    async def close(self) -> None:
        """Close connection to Qdrant client."""
        try:
            if hasattr(self.client, "close"):
                await self.client.close()
        except Exception as exc:
            logger.debug(f"Error closing Qdrant client: {exc}")

    # =========================================================================
    # Compatibility helpers (for direct profile & defect queries)
    # =========================================================================

    def get_filament_profile(
        self,
        material: str,
        strength_priority: bool = False,
    ) -> dict[str, Any]:
        """
        Synchronous helper returning standard Rosatom technical parameters for a material.
        Falls back to default filaments database if Qdrant is offline.
        """
        mat_clean = material.strip().upper()
        exact_meta: Optional[dict[str, Any]] = None

        for doc in self._in_memory_docs:
            if doc.get("doc_type") == "filament" and str(doc.get("key", "")).upper() == mat_clean:
                exact_meta = doc.get("metadata")
                break

        if exact_meta is None:
            for item in DEFAULT_FILAMENTS:
                if item["material"].upper() == mat_clean:
                    exact_meta = item
                    break

        if exact_meta is None:
            exact_meta = DEFAULT_FILAMENTS[0]  # Default to PLA

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

    def get_defect_solution(
        self,
        defect_type: str,
        observed_symptoms: str = "",
        current_material: str = "ABS",
    ) -> dict[str, Any]:
        """
        Synchronous helper returning standard troubleshooting guide for a print defect.
        """
        defect_clean = defect_type.strip().lower()
        matched: Optional[dict[str, Any]] = None

        for doc in self._in_memory_docs:
            if doc.get("doc_type") == "defect" and str(doc.get("key", "")).lower() == defect_clean:
                matched = doc.get("metadata")
                break

        if matched is None:
            for item in DEFAULT_DEFECTS:
                if item["defect_type"].lower() == defect_clean:
                    matched = item
                    break

        mat_profile = self.get_filament_profile(current_material)

        if matched is None:
            return {
                "defect_type": defect_type,
                "current_material": mat_profile["material"],
                "parameter_adjustments": {"print_speed_delta_percent": -15},
                "operator_actions": [
                    f"Просушить катушку ({mat_profile.get('drying_regime', '70°C 4 часа')})",
                    "Проверить калибровку первого слоя и чистоту сопла",
                ],
            }

        actions = list(matched.get("operator_actions", []))
        if mat_profile.get("drying_regime"):
            actions.append(f"Режим сушки катушки для {mat_profile['material']}: {mat_profile['drying_regime']}")

        return {
            "defect_type": matched.get("defect_type", defect_clean),
            "title": matched.get("title", ""),
            "current_material": mat_profile["material"],
            "root_causes": matched.get("root_causes", []),
            "parameter_adjustments": matched.get("parameter_adjustments", {}),
            "operator_actions": actions,
        }

    def build_rag_context(
        self,
        user_query: str,
        material: Optional[str] = None,
        defect_type: Optional[str] = None,
    ) -> str:
        """
        Synchronous context assembler for agent reasoning prompt.
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

        return "\n".join(sections)


# Backwards compatibility alias
RosatomKnowledgeRetriever = RAGRetriever

_retriever_instance: Optional[RAGRetriever] = None


def get_retriever() -> RAGRetriever:
    """Return a singleton instance of RAGRetriever."""
    global _retriever_instance
    if _retriever_instance is None:
        qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
        _retriever_instance = RAGRetriever(qdrant_url=qdrant_url)
    return _retriever_instance