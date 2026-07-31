"""Compatibility exports for local dense embedding infrastructure."""

from rag_pipeline.infrastructure.embeddings import (
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    EmbeddedDocument,
    EmbeddingService,
    LocalEmbeddingConfig,
    create_local_embedding_service,
)

__all__ = [
    "DEFAULT_LOCAL_EMBEDDING_MODEL",
    "EmbeddedDocument",
    "EmbeddingService",
    "LocalEmbeddingConfig",
    "create_local_embedding_service",
]
