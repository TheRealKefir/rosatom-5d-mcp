"""
RAG-пакет базы знаний Росатома по 3D/5D-печати.
Экспортирует индексатор Qdrant и семантический ретривер профилей материалов и дефектов.
"""

from orchestrator.rag.ingest import (
    COLLECTION_NAME,
    DEFAULT_DEFECTS,
    DEFAULT_FILAMENTS,
    DEFAULT_KINEMATICS_MD,
    build_documents,
    compute_embeddings,
    ensure_knowledge_base_files,
    get_qdrant_client,
    ingest_knowledge_base,
)
from orchestrator.rag.retriever import (
    RosatomKnowledgeRetriever,
    get_retriever,
)

__all__ = [
    "COLLECTION_NAME",
    "DEFAULT_FILAMENTS",
    "DEFAULT_DEFECTS",
    "DEFAULT_KINEMATICS_MD",
    "ensure_knowledge_base_files",
    "build_documents",
    "compute_embeddings",
    "get_qdrant_client",
    "ingest_knowledge_base",
    "RosatomKnowledgeRetriever",
    "get_retriever",
]