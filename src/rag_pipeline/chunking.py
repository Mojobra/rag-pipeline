"""Compatibility exports for recursive and experimental chunking policies."""

from rag_pipeline.ingestion.chunking import (
    ChunkingConfig,
    StructureAwareChunkingConfig,
    chunk_documents,
    chunk_documents_with_structure,
)
from rag_pipeline.ingestion.semantic_chunking import (
    SemanticChunkingConfig,
    chunk_documents_semantically,
)

__all__ = [
    "ChunkingConfig",
    "SemanticChunkingConfig",
    "StructureAwareChunkingConfig",
    "chunk_documents",
    "chunk_documents_semantically",
    "chunk_documents_with_structure",
]
