"""Compatibility exports for local Qdrant vector-store infrastructure."""

from rag_pipeline.infrastructure.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_VECTOR_STORE_PATH,
    DENSE_VECTOR_NAME,
    HYBRID_VECTOR_STORE_SCHEMA_VERSION,
    SPARSE_VECTOR_NAME,
    VECTOR_STORE_SCHEMA_VERSION,
    IndexingResult,
    LocalVectorStore,
    SearchMode,
    VectorStoreConfig,
    build_chunk_point_id,
)

__all__ = [
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_VECTOR_STORE_PATH",
    "DENSE_VECTOR_NAME",
    "HYBRID_VECTOR_STORE_SCHEMA_VERSION",
    "SPARSE_VECTOR_NAME",
    "VECTOR_STORE_SCHEMA_VERSION",
    "IndexingResult",
    "LocalVectorStore",
    "SearchMode",
    "VectorStoreConfig",
    "build_chunk_point_id",
]
