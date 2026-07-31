"""Application use cases for local embedding previews and collection indexing.

These functions compose ingestion, chunking, dense and optional sparse
embedding, and Qdrant persistence while keeping those side effects out of CLI
handlers. Domain services remain independently testable and reusable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_pipeline.exceptions import InvalidPipelineConfigurationError
from rag_pipeline.infrastructure.embeddings import (
    EmbeddedDocument,
    LocalEmbeddingConfig,
)
from rag_pipeline.infrastructure.sparse_embeddings import LocalSparseEmbeddingConfig
from rag_pipeline.infrastructure.vector_store import (
    IndexingResult,
    LocalVectorStore,
    SearchMode,
    VectorStoreConfig,
)
from rag_pipeline.ingestion import load_documents
from rag_pipeline.ingestion.chunking import ChunkingConfig, chunk_documents


@dataclass(frozen=True, slots=True)
class IndexingPipelineConfig:
    """Validated settings for one local document-indexing workflow.

    Dense collections must omit sparse settings, while hybrid collections must
    provide them. Component configurations perform their own field validation;
    this object enforces only the cross-stage search-mode contract.
    """

    chunking: ChunkingConfig
    embedding: LocalEmbeddingConfig
    vector_store: VectorStoreConfig
    sparse_embedding: LocalSparseEmbeddingConfig | None = None

    def __post_init__(self) -> None:
        """Reject invalid component types and dense/hybrid configuration drift."""
        _validate_component_types(self)
        is_hybrid = self.vector_store.search_mode is SearchMode.HYBRID
        if is_hybrid != (self.sparse_embedding is not None):
            raise InvalidPipelineConfigurationError(
                "Hybrid indexing requires sparse embedding settings, and dense "
                "indexing must omit them."
            )


@dataclass(frozen=True, slots=True)
class EmbeddingPreview:
    """Summarize dense vectors created without writing a collection."""

    chunk_count: int
    dimension: int | None
    model_identifier: str


def preview_local_embeddings(
    paths: Sequence[str | Path],
    *,
    recursive: bool,
    chunking: ChunkingConfig,
    embedding: LocalEmbeddingConfig,
) -> EmbeddingPreview:
    """Extract, chunk, and densely embed local documents without persistence.

    Model initialization may download artifacts and perform inference. The
    returned dimension is ``None`` when the input produces no chunks; no vector
    database is opened or modified.
    """
    embedded_documents, model_identifier = _prepare_dense_embeddings(
        paths,
        recursive=recursive,
        chunking=chunking,
        embedding=embedding,
    )
    dimension = None if not embedded_documents else embedded_documents[0].dimension
    return EmbeddingPreview(
        chunk_count=len(embedded_documents),
        dimension=dimension,
        model_identifier=model_identifier,
    )


def index_local_documents(
    paths: Sequence[str | Path],
    *,
    recursive: bool,
    config: IndexingPipelineConfig,
) -> IndexingResult:
    """Build and persist dense or hybrid vectors for local documents.

    The use case performs filesystem reads, model initialization and inference,
    creates or validates the configured Qdrant collection, and synchronously
    upserts deterministic points. Lower-level services enforce input,
    provenance, and collection-compatibility invariants.
    """
    if not isinstance(config, IndexingPipelineConfig):
        raise TypeError("config must be an IndexingPipelineConfig.")

    documents = load_documents(paths, recursive=recursive)
    chunks = chunk_documents(documents, config=config.chunking)

    # Import factories at execution time so constructing application objects
    # remains side-effect free and provider boundaries stay replaceable in tests.
    from rag_pipeline.infrastructure.embeddings import create_local_embedding_service
    from rag_pipeline.infrastructure.sparse_embeddings import (
        create_local_sparse_embedding_service,
    )

    embedding_service = create_local_embedding_service(config.embedding)
    embedded_documents = embedding_service.embed_documents(chunks)
    sparse_service = (
        None
        if config.sparse_embedding is None
        else create_local_sparse_embedding_service(config.sparse_embedding)
    )
    sparse_vectors = (
        None if sparse_service is None else sparse_service.embed_documents(chunks)
    )

    with LocalVectorStore(config.vector_store) as vector_store:
        return vector_store.index(
            embedded_documents,
            model_identifier=embedding_service.model_identifier,
            sparse_vectors=sparse_vectors,
            sparse_model_identifier=(
                None if sparse_service is None else sparse_service.model_identifier
            ),
        )


def _prepare_dense_embeddings(
    paths: Sequence[str | Path],
    *,
    recursive: bool,
    chunking: ChunkingConfig,
    embedding: LocalEmbeddingConfig,
) -> tuple[list[EmbeddedDocument], str]:
    """Load one document snapshot and return its dense embeddings and model ID."""
    if not isinstance(chunking, ChunkingConfig):
        raise TypeError("chunking must be a ChunkingConfig.")
    if not isinstance(embedding, LocalEmbeddingConfig):
        raise TypeError("embedding must be a LocalEmbeddingConfig.")

    from rag_pipeline.infrastructure.embeddings import create_local_embedding_service

    documents = load_documents(paths, recursive=recursive)
    chunks = chunk_documents(documents, config=chunking)
    service = create_local_embedding_service(embedding)
    return service.embed_documents(chunks), service.model_identifier


def _validate_component_types(config: IndexingPipelineConfig) -> None:
    """Fail early when callers bypass static typing with wrong config objects."""
    _require_component_type("chunking", config.chunking, ChunkingConfig)
    _require_component_type("embedding", config.embedding, LocalEmbeddingConfig)
    _require_component_type(
        "vector_store",
        config.vector_store,
        VectorStoreConfig,
    )
    if config.sparse_embedding is not None and not isinstance(
        config.sparse_embedding,
        LocalSparseEmbeddingConfig,
    ):
        raise TypeError(
            "sparse_embedding must be a LocalSparseEmbeddingConfig or None."
        )


def _require_component_type(
    name: str,
    value: object,
    expected_type: type[object],
) -> None:
    """Raise a consistent type error for one required workflow component."""
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be a {expected_type.__name__}.")
