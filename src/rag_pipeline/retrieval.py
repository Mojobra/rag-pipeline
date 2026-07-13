"""Model-safe semantic retrieval from the local LangChain vector store."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import TypeAlias

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models

from rag_pipeline.embeddings import EmbeddingService
from rag_pipeline.exceptions import (
    InvalidRetrievalConfigurationError,
    RetrievalInputError,
    RetrievalProviderError,
    VectorStoreError,
)
from rag_pipeline.vector_store import LocalVectorStore


MetadataFilterValue: TypeAlias = str | int | bool

_METADATA_FIELD_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)*"
)
_MIN_QDRANT_INTEGER = -(2**63)
_MAX_QDRANT_INTEGER = 2**63 - 1


@dataclass(frozen=True, slots=True)
class MetadataFilter:
    """One exact-match condition against indexed document metadata."""

    field: str
    value: MetadataFilterValue

    def __post_init__(self) -> None:
        if not isinstance(self.field, str):
            raise InvalidRetrievalConfigurationError(
                "metadata filter field must be a string."
            )
        normalized_field = self.field.strip()
        if not normalized_field or not _METADATA_FIELD_PATTERN.fullmatch(
            normalized_field
        ):
            raise InvalidRetrievalConfigurationError(
                "metadata filter field must contain dot-separated letters, "
                "numbers, underscores, or hyphens."
            )
        object.__setattr__(self, "field", normalized_field)

        if isinstance(self.value, str):
            if not self.value.strip():
                raise InvalidRetrievalConfigurationError(
                    "metadata filter string value cannot be empty."
                )
            return
        if isinstance(self.value, bool):
            return
        if isinstance(self.value, int):
            if not _MIN_QDRANT_INTEGER <= self.value <= _MAX_QDRANT_INTEGER:
                raise InvalidRetrievalConfigurationError(
                    "metadata filter integer must fit in a signed 64-bit value."
                )
            return
        raise InvalidRetrievalConfigurationError(
            "metadata filter value must be a string, integer, or boolean."
        )


def parse_metadata_filter(expression: str) -> MetadataFilter:
    """Parse a CLI ``KEY=VALUE`` expression into a typed exact-match filter."""
    if not isinstance(expression, str):
        raise TypeError("metadata filter expression must be a string.")

    field, separator, raw_value = expression.partition("=")
    raw_value = raw_value.strip()
    if not separator or not field.strip() or not raw_value:
        raise InvalidRetrievalConfigurationError(
            "metadata filter must use KEY=VALUE, for example file_extension=.pdf."
        )

    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return MetadataFilter(field=field, value=value)


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Controls how many cosine-similar chunks are accepted."""

    top_k: int = 4
    score_threshold: float | None = None
    metadata_filters: tuple[MetadataFilter, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int):
            raise InvalidRetrievalConfigurationError("top_k must be an integer.")
        if self.top_k <= 0:
            raise InvalidRetrievalConfigurationError(
                "top_k must be greater than zero."
            )

        if self.score_threshold is not None:
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

        if isinstance(self.metadata_filters, (str, bytes)):
            raise InvalidRetrievalConfigurationError(
                "metadata_filters must contain MetadataFilter objects."
            )
        try:
            metadata_filters = tuple(self.metadata_filters)
        except TypeError as exc:
            raise InvalidRetrievalConfigurationError(
                "metadata_filters must be an iterable of MetadataFilter objects."
            ) from exc

        seen_filter_keys: set[
            tuple[str, type[object], MetadataFilterValue]
        ] = set()
        for metadata_filter in metadata_filters:
            if not isinstance(metadata_filter, MetadataFilter):
                raise InvalidRetrievalConfigurationError(
                    "metadata_filters must contain MetadataFilter objects."
                )
            filter_key = (
                metadata_filter.field,
                type(metadata_filter.value),
                metadata_filter.value,
            )
            if filter_key in seen_filter_keys:
                raise InvalidRetrievalConfigurationError(
                    f"duplicate metadata filter {metadata_filter.field!r}."
                )
            seen_filter_keys.add(filter_key)
        object.__setattr__(self, "metadata_filters", metadata_filters)


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
                filter=_build_qdrant_metadata_filter(settings.metadata_filters),
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


def _build_qdrant_metadata_filter(
    metadata_filters: tuple[MetadataFilter, ...],
) -> models.Filter | None:
    if not metadata_filters:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key=f"{QdrantVectorStore.METADATA_KEY}.{condition.field}",
                match=models.MatchValue(value=condition.value),
            )
            for condition in metadata_filters
        ]
    )
