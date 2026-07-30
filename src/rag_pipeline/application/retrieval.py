"""Application service for reusable local retrieval and optional reranking.

The module owns model and Qdrant lifecycles for one retrieval session. Callers
can run many queries through the same initialized services without depending on
CLI parsing or terminal rendering.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from rag_pipeline.embeddings import LocalEmbeddingConfig
from rag_pipeline.exceptions import InvalidPipelineConfigurationError
from rag_pipeline.reranking import LocalRerankerConfig, RerankingConfig
from rag_pipeline.retrieval import RetrievalConfig, RetrievalResult
from rag_pipeline.sparse_embeddings import LocalSparseEmbeddingConfig
from rag_pipeline.vector_store import (
    LocalVectorStore,
    SearchMode,
    VectorStoreConfig,
)


class _Retriever(Protocol):
    """Structural contract used by the application retrieval coordinator."""

    def retrieve(
        self,
        query: str,
        *,
        config: RetrievalConfig | None = None,
    ) -> list[RetrievalResult]:
        """Return ranked first-stage results for one query."""


class _Reranker(Protocol):
    """Structural contract for optional second-stage result ordering."""

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        config: RerankingConfig | None = None,
    ) -> list[RetrievalResult]:
        """Return reranked results for one query and candidate list."""


@dataclass(frozen=True, slots=True)
class RetrievalPipelineConfig:
    """Validated local provider settings for a reusable retrieval session.

    The contract binds dense or hybrid collection mode to the matching model
    settings and requires reranker model and result-limit settings together.
    Individual configuration objects retain their own validation behavior.
    """

    embedding: LocalEmbeddingConfig
    vector_store: VectorStoreConfig
    sparse_embedding: LocalSparseEmbeddingConfig | None
    retrieval: RetrievalConfig
    local_reranker: LocalRerankerConfig | None
    reranking: RerankingConfig | None

    def __post_init__(self) -> None:
        """Validate component types and cross-stage optional-service pairs."""
        _validate_config_types(self)
        is_hybrid = self.vector_store.search_mode is SearchMode.HYBRID
        if is_hybrid != (self.sparse_embedding is not None):
            raise InvalidPipelineConfigurationError(
                "Hybrid retrieval requires sparse embedding settings, and dense "
                "retrieval must omit them."
            )
        if (self.local_reranker is None) != (self.reranking is None):
            raise InvalidPipelineConfigurationError(
                "Reranker model and reranking result settings must be configured "
                "together."
            )


class RetrievalPipeline:
    """Coordinate first-stage retrieval and optional cross-encoder reranking.

    Provider construction and resource ownership live in
    :func:`open_local_retrieval_pipeline`; this class contains only the reusable
    query use case, which also makes it straightforward to test with doubles.
    """

    def __init__(
        self,
        retriever: _Retriever,
        *,
        retrieval_config: RetrievalConfig,
        reranker: _Reranker | None = None,
        reranker_factory: Callable[[], _Reranker] | None = None,
        reranking_config: RerankingConfig | None = None,
    ) -> None:
        """Bind initialized services and an optional lazy reranker factory."""
        if not callable(getattr(retriever, "retrieve", None)):
            raise TypeError("retriever must provide a retrieve method.")
        if not isinstance(retrieval_config, RetrievalConfig):
            raise TypeError("retrieval_config must be a RetrievalConfig.")
        if reranker is not None and reranker_factory is not None:
            raise InvalidPipelineConfigurationError(
                "Provide either reranker or reranker_factory, not both."
            )
        has_reranker = reranker is not None or reranker_factory is not None
        if has_reranker != (reranking_config is not None):
            raise InvalidPipelineConfigurationError(
                "A reranker service or factory and reranking_config must be "
                "provided together."
            )
        if reranker is not None and not callable(getattr(reranker, "rerank", None)):
            raise TypeError("reranker must provide a rerank method.")
        if reranker_factory is not None and not callable(reranker_factory):
            raise TypeError("reranker_factory must be callable.")

        self._retriever = retriever
        self._retrieval_config = retrieval_config
        self._reranker = reranker
        self._reranker_factory = reranker_factory
        self._reranking_config = reranking_config

    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Run one query through the configured retrieval and reranking stages.

        The underlying services validate the query and provider responses.
        Result order and provenance are preserved when reranking is disabled.
        """
        results = self._retriever.retrieve(
            query,
            config=self._retrieval_config,
        )
        if not results or self._reranking_config is None:
            return results
        if self._reranker is None:
            if self._reranker_factory is None:
                raise RuntimeError("Reranking is configured without a service.")
            reranker = self._reranker_factory()
            if not callable(getattr(reranker, "rerank", None)):
                raise TypeError(
                    "reranker_factory must return an object with a rerank method."
                )
            self._reranker = reranker
        return self._reranker.rerank(
            query,
            results,
            config=self._reranking_config,
        )


@contextmanager
def open_local_retrieval_pipeline(
    config: RetrievalPipelineConfig,
) -> Iterator[RetrievalPipeline]:
    """Initialize and share local retrieval resources for a query session.

    Entering may download/cache embedding or reranking models and opens the
    configured Qdrant database. The yielded pipeline reuses all services across
    queries; exiting always closes the vector store and releases file locks.
    """
    if not isinstance(config, RetrievalPipelineConfig):
        raise TypeError("config must be a RetrievalPipelineConfig.")

    # Lazy imports keep module import and CLI help free from model or database
    # initialization, while preserving explicit provider factory boundaries.
    from rag_pipeline.embeddings import create_local_embedding_service
    from rag_pipeline.reranking import create_local_reranker_service
    from rag_pipeline.retrieval import RetrieverService
    from rag_pipeline.sparse_embeddings import (
        create_local_sparse_embedding_service,
    )

    embedding_service = create_local_embedding_service(config.embedding)
    sparse_service = (
        None
        if config.sparse_embedding is None
        else create_local_sparse_embedding_service(config.sparse_embedding)
    )
    reranker_factory = (
        None
        if config.local_reranker is None
        else lambda: create_local_reranker_service(config.local_reranker)
    )

    with LocalVectorStore(config.vector_store) as vector_store:
        retriever = RetrieverService(
            embedding_service,
            vector_store,
            sparse_service,
        )
        yield RetrievalPipeline(
            retriever,
            retrieval_config=config.retrieval,
            reranker_factory=reranker_factory,
            reranking_config=config.reranking,
        )


def _validate_config_types(config: RetrievalPipelineConfig) -> None:
    """Reject wrong component objects before any provider side effects occur."""
    _require_component_type("embedding", config.embedding, LocalEmbeddingConfig)
    _require_component_type(
        "vector_store",
        config.vector_store,
        VectorStoreConfig,
    )
    _require_component_type("retrieval", config.retrieval, RetrievalConfig)
    _require_optional_component_type(
        "sparse_embedding",
        config.sparse_embedding,
        LocalSparseEmbeddingConfig,
    )
    _require_optional_component_type(
        "local_reranker",
        config.local_reranker,
        LocalRerankerConfig,
    )
    _require_optional_component_type(
        "reranking",
        config.reranking,
        RerankingConfig,
    )


def _require_component_type(
    name: str,
    value: object,
    expected_type: type[object],
) -> None:
    """Raise a consistent type error for one required workflow component."""
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be a {expected_type.__name__}.")


def _require_optional_component_type(
    name: str,
    value: object,
    expected_type: type[object],
) -> None:
    """Validate one optional workflow component without broad casts."""
    if value is not None and not isinstance(value, expected_type):
        raise TypeError(f"{name} must be a {expected_type.__name__} or None.")
