"""Build validated HTTP-service settings from explicit environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from rag_pipeline.application.retrieval import RetrievalPipelineConfig
from rag_pipeline.generation import (
    DEFAULT_LOCAL_GENERATION_MODEL,
    GenerationConfig,
    LocalGenerationConfig,
)
from rag_pipeline.infrastructure.embeddings import (
    DEFAULT_LOCAL_EMBEDDING_MODEL,
    LocalEmbeddingConfig,
)
from rag_pipeline.infrastructure.sparse_embeddings import (
    DEFAULT_LOCAL_SPARSE_MODEL,
    LocalSparseEmbeddingConfig,
)
from rag_pipeline.infrastructure.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_VECTOR_STORE_PATH,
    SearchMode,
    VectorStoreConfig,
)
from rag_pipeline.retrieval import RetrievalConfig


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """Bind one local collection and model pair for the HTTP process.

    The collection is read-only through this adapter. Component contracts
    enforce model and storage compatibility when the first query runs.
    """

    retrieval: RetrievalPipelineConfig
    generation_model: LocalGenerationConfig
    generation: GenerationConfig = field(default_factory=GenerationConfig)


def load_api_settings() -> ApiSettings:
    """Read server-owned settings at startup, without loading models or Qdrant.

    ``RAG_API_*`` values select the same collection and provider identities used
    during CLI indexing. Invalid values fail startup via existing config checks.
    An optional sparse model is used only for a hybrid collection.
    """
    store = VectorStoreConfig(
        path=os.getenv("RAG_API_STORE_PATH", str(DEFAULT_VECTOR_STORE_PATH)),
        collection_name=os.getenv("RAG_API_COLLECTION_NAME", DEFAULT_COLLECTION_NAME),
        search_mode=os.getenv("RAG_API_SEARCH_MODE", SearchMode.DENSE.value),
    )
    sparse = (
        LocalSparseEmbeddingConfig(
            model_name=os.getenv("RAG_API_SPARSE_MODEL", DEFAULT_LOCAL_SPARSE_MODEL)
        )
        if store.search_mode is SearchMode.HYBRID
        else None
    )
    retrieval = RetrievalPipelineConfig(
        embedding=LocalEmbeddingConfig(
            model_name=os.getenv(
                "RAG_API_EMBEDDING_MODEL", DEFAULT_LOCAL_EMBEDDING_MODEL
            ),
            model_revision=os.getenv("RAG_API_EMBEDDING_REVISION"),
        ),
        vector_store=store,
        sparse_embedding=sparse,
        retrieval=RetrievalConfig(),
        local_reranker=None,
        reranking=None,
    )
    return ApiSettings(
        retrieval=retrieval,
        generation_model=LocalGenerationConfig(
            model_name=os.getenv(
                "RAG_API_GENERATION_MODEL", DEFAULT_LOCAL_GENERATION_MODEL
            ),
            model_revision=os.getenv("RAG_API_GENERATION_REVISION"),
        ),
    )
