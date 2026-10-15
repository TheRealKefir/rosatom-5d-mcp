"""
RAG-пакет базы знаний Росатома по 3D/5D-печати.
Экспортирует индексатор Qdrant и семантический ретривер профилей материалов и дефектов.
"""

from orchestrator.rag.ingest import (
    COLLECTION_DEFECTS,
    COLLECTION_FILAMENTS,
    COLLECTION_KINEMATICS,
    COLLECTION_NAME,
    DEFAULT_DEFECTS,
    DEFAULT_FILAMENTS,
    DEFAULT_KINEMATICS_MD,
    build_documents,
    compute_embeddings,
    ensure_knowledge_base_files,
    get_async_qdrant_client,
    get_qdrant_client,
    ingest_knowledge_base,
    ingest_to_qdrant,
    load_defects,
    load_filaments,
    load_kinematics,
)
from orchestrator.rag.retriever import (
    RAGRetriever,
    RosatomKnowledgeRetriever,
    get_retriever,
)

__all__ = [
    "COLLECTION_NAME",
    "COLLECTION_FILAMENTS",
    "COLLECTION_DEFECTS",
    "COLLECTION_KINEMATICS",
    "DEFAULT_FILAMENTS",
    "DEFAULT_DEFECTS",
    "DEFAULT_KINEMATICS_MD",
    "ensure_knowledge_base_files",
    "build_documents",
    "compute_embeddings",
    "get_qdrant_client",
    "get_async_qdrant_client",
    "ingest_knowledge_base",
    "ingest_to_qdrant",
    "load_filaments",
    "load_defects",
    "load_kinematics",
    "RAGRetriever",
    "RosatomKnowledgeRetriever",
    "get_retriever",
]