"""Model-safe semantic retrieval from the local LangChain vector store."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from langchain_core.documents import Document

from rag_pipeline.embeddings import EmbeddingService
from rag_pipeline.exceptions import (
    InvalidRetrievalConfigurationError,
    RetrievalInputError,
    RetrievalProviderError,
    VectorStoreError,
)
from rag_pipeline.vector_store import LocalVectorStore


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Controls how many cosine-similar chunks are accepted."""

    top_k: int = 4
    score_threshold: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise InvalidRetrievalConfigurationError("top_k must be an integer.")
        if self.top_k <= 0:
            raise InvalidRetrievalConfigurationError(
                "top_k must be greater than zero."
            )

        if self.score_threshold is None:
            return
        if isinstance(self.score_threshold, bool) or not isinstance(
            self.score_threshold, Real
        ):
            raise InvalidRetrievalConfigurationError(
                "score_threshold must be a number."
            )
        threshold = float(self.score_threshold)
        if not isfinite(threshold):
            raise InvalidRetrievalConfigurationError(
                "score_threshold must be finite."
            )
        if not -1.0 <= threshold <= 1.0:
            raise InvalidRetrievalConfigurationError(
                "score_threshold must be between -1 and 1 for cosine search."
            )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One ranked document returned by semantic retrieval."""

    document: Document
    score: float
    rank: int


class RetrieverService:
    """Embed queries and retrieve compatible chunks through LangChain."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: LocalVectorStore,
    ) -> None:
        if not isinstance(embedding_service, EmbeddingService):
            raise TypeError("embedding_service must be an EmbeddingService.")
        if not isinstance(vector_store, LocalVectorStore):
            raise TypeError("vector_store must be a LocalVectorStore.")

        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        config: RetrievalConfig | None = None,
    ) -> list[RetrievalResult]:
        """Return chunks ordered by descending Qdrant cosine similarity."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not query.strip():
            raise RetrievalInputError("query cannot be empty.")

        settings = config or RetrievalConfig()
        query_embedding = self._embedding_service.embed_query(query)
        self._vector_store.validate_compatibility(
            model_identifier=self._embedding_service.model_identifier,
            dimension=len(query_embedding),
        )

        try:
            langchain_store = self._vector_store.as_langchain_vector_store()
            provider_results = langchain_store.similarity_search_with_score_by_vector(
                embedding=list(query_embedding),
                k=settings.top_k,
                score_threshold=settings.score_threshold,
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise RetrievalProviderError(
                "LangChain Qdrant similarity search failed."
            ) from exc

        results = []
        for provider_index, provider_result in enumerate(provider_results):
            if not isinstance(provider_result, (tuple, list)) or len(provider_result) != 2:
                raise RetrievalProviderError(
                    f"Search result {provider_index} is not a document-score pair."
                )
            document, score = provider_result
            if not isinstance(document, Document):
                raise RetrievalProviderError(
                    f"Search result {provider_index} is not a LangChain Document."
                )
            if isinstance(score, bool) or not isinstance(score, Real):
                raise RetrievalProviderError(
                    f"Search result {provider_index} has a non-numeric score."
                )
            numeric_score = float(score)
            if not isfinite(numeric_score):
                raise RetrievalProviderError(
                    f"Search result {provider_index} has a non-finite score."
                )
            if (
                settings.score_threshold is not None
                and numeric_score < settings.score_threshold
            ):
                continue

            results.append(
                RetrievalResult(
                    document=document,
                    score=numeric_score,
                    rank=len(results) + 1,
                )
            )

        return results
