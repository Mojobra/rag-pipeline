"""Compatibility exports for local sparse embedding infrastructure."""

from rag_pipeline.infrastructure.sparse_embeddings import (
    DEFAULT_FASTEMBED_CACHE_DIR,
    DEFAULT_LOCAL_SPARSE_MODEL,
    LocalSparseEmbeddingConfig,
    SparseEmbeddingService,
    SparseEmbeddingVector,
    create_local_sparse_embedding_service,
)

__all__ = [
    "DEFAULT_FASTEMBED_CACHE_DIR",
    "DEFAULT_LOCAL_SPARSE_MODEL",
    "LocalSparseEmbeddingConfig",
    "SparseEmbeddingService",
    "SparseEmbeddingVector",
    "create_local_sparse_embedding_service",
]
