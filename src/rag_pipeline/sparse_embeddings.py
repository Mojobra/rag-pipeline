"""Generate validated sparse vectors for optional lexical retrieval.

The service adapts LangChain Qdrant sparse embeddings to immutable, sorted
index/value tuples used by hybrid indexing and query fusion.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import SparseEmbeddings, SparseVector

from rag_pipeline.exceptions import (
    EmbeddingInputError,
    EmbeddingProviderError,
    InvalidEmbeddingConfigurationError,
)


DEFAULT_LOCAL_SPARSE_MODEL = "Qdrant/bm25"
DEFAULT_FASTEMBED_CACHE_DIR = Path(".rag_data/fastembed")


@dataclass(frozen=True, slots=True)
class LocalSparseEmbeddingConfig:
    """Validated runtime settings for the local FastEmbed sparse model.

    The configuration controls model identity, cache location, batch size, and
    optional CPU threading before any model artifacts are initialized.
    """

    model_name: str = DEFAULT_LOCAL_SPARSE_MODEL
    cache_dir: str | Path | None = DEFAULT_FASTEMBED_CACHE_DIR
    batch_size: int = 256
    threads: int | None = None

    def __post_init__(self) -> None:
        """Validate sparse model, cache, batching, and thread settings eagerly."""
        _validate_non_empty_string("model_name", self.model_name)
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)):
                raise InvalidEmbeddingConfigurationError(
                    "sparse cache_dir must be a string, Path, or None."
                )
            if isinstance(self.cache_dir, str) and not self.cache_dir.strip():
                raise InvalidEmbeddingConfigurationError(
                    "sparse cache_dir must be non-empty when provided."
                )
        _validate_positive_integer("sparse batch_size", self.batch_size)
        if self.threads is not None:
            _validate_positive_integer("sparse threads", self.threads)

    @property
    def resolved_cache_dir(self) -> Path | None:
        if self.cache_dir is None:
            return None
        return Path(self.cache_dir).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class SparseEmbeddingVector:
    """Immutable sparse indices and weights for one document or query.

    Service-produced instances have aligned, unique, sorted indices; the vector
    store revalidates externally constructed records. An empty service-produced
    vector is valid and triggers dense-only query fallback.
    """

    indices: tuple[int, ...]
    values: tuple[float, ...]

    @property
    def is_empty(self) -> bool:
        return not self.indices


class SparseEmbeddingService:
    """Coordinate one sparse provider for hybrid indexing and retrieval.

    The service enforces input ordering and validates each provider vector before
    it reaches Qdrant. It is used only when a collection has hybrid schema.
    """

    def __init__(
        self,
        embeddings: SparseEmbeddings,
        *,
        model_name: str,
    ) -> None:
        if not isinstance(embeddings, SparseEmbeddings):
            raise TypeError(
                "embeddings must implement LangChain's SparseEmbeddings interface."
            )
        _validate_non_empty_string("model_name", model_name)
        self._embeddings = embeddings
        self._model_name = model_name

    @property
    def model_identifier(self) -> str:
        return self._model_name

    def embed_documents(
        self,
        documents: Iterable[Document],
    ) -> list[SparseEmbeddingVector]:
        """Embed non-empty documents into sparse vectors in input order.

        The provider performs local model inference. Output cardinality and each
        index/value sequence are validated so vectors stay aligned with the
        dense embeddings passed to hybrid indexing.
        """
        source_documents = list(documents)
        if not source_documents:
            return []

        for index, document in enumerate(source_documents):
            if not isinstance(document, Document):
                raise TypeError(
                    f"documents[{index}] must be a LangChain Document object."
                )
            if not document.page_content.strip():
                raise EmbeddingInputError(
                    f"documents[{index}] has empty page_content."
                )

        try:
            provider_vectors = list(
                self._embeddings.embed_documents(
                    [document.page_content for document in source_documents]
                )
            )
        except Exception as exc:
            raise EmbeddingProviderError(
                f"Sparse embedding model {self.model_identifier} failed for documents."
            ) from exc

        if len(provider_vectors) != len(source_documents):
            raise EmbeddingProviderError(
                "Sparse embedding provider returned "
                f"{len(provider_vectors)} vector(s) for "
                f"{len(source_documents)} document(s)."
            )
        return [
            _normalize_sparse_vector(vector, context=f"document vector {index}")
            for index, vector in enumerate(provider_vectors)
        ]

    def embed_query(self, query: str) -> SparseEmbeddingVector:
        """Embed one non-empty query for the sparse hybrid-search branch.

        Provider inference may legitimately return an empty vector; retrieval
        handles that edge case by falling back to dense search. Malformed sparse
        output raises ``EmbeddingProviderError``.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not query.strip():
            raise EmbeddingInputError("query cannot be empty.")

        try:
            provider_vector = self._embeddings.embed_query(query)
        except Exception as exc:
            raise EmbeddingProviderError(
                f"Sparse embedding model {self.model_identifier} failed for a query."
            ) from exc
        return _normalize_sparse_vector(provider_vector, context="query vector")


def create_local_sparse_embedding_service(
    config: LocalSparseEmbeddingConfig | None = None,
) -> SparseEmbeddingService:
    """Initialize the local FastEmbed BM25 service through LangChain.

    Construction may populate the configured model cache and allocate inference
    resources. Import and provider initialization failures are normalized to
    ``EmbeddingProviderError``.
    """
    settings = config or LocalSparseEmbeddingConfig()

    try:
        from langchain_qdrant import FastEmbedSparse
    except ImportError as exc:
        raise EmbeddingProviderError(
            "Local sparse embeddings require langchain-qdrant and fastembed."
        ) from exc

    cache_dir = settings.resolved_cache_dir
    try:
        embeddings = FastEmbedSparse(
            model_name=settings.model_name,
            batch_size=settings.batch_size,
            cache_dir=None if cache_dir is None else str(cache_dir),
            threads=settings.threads,
        )
    except Exception as exc:
        raise EmbeddingProviderError(
            f"Failed to initialize sparse embedding model {settings.model_name}."
        ) from exc

    return SparseEmbeddingService(embeddings, model_name=settings.model_name)


def _normalize_sparse_vector(
    raw_vector: SparseVector,
    *,
    context: str,
) -> SparseEmbeddingVector:
    """Validate and sort one provider sparse vector.

    Index/value lengths must match; indices must be unique non-negative integers;
    weights must be finite numbers. Empty vectors are retained because retrieval
    has an explicit dense fallback for them.
    """
    if not isinstance(raw_vector, SparseVector):
        raise EmbeddingProviderError(
            f"Sparse {context} is not a LangChain SparseVector."
        )
    if len(raw_vector.indices) != len(raw_vector.values):
        raise EmbeddingProviderError(
            f"Sparse {context} has different index and value counts."
        )

    pairs: list[tuple[int, float]] = []
    seen_indices: set[int] = set()
    for position, (index, value) in enumerate(
        zip(raw_vector.indices, raw_vector.values, strict=True)
    ):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise EmbeddingProviderError(
                f"Sparse {context} has an invalid index at position {position}."
            )
        if index in seen_indices:
            raise EmbeddingProviderError(
                f"Sparse {context} contains duplicate index {index}."
            )
        if isinstance(value, bool) or not isinstance(value, Real):
            raise EmbeddingProviderError(
                f"Sparse {context} has a non-numeric value at position {position}."
            )
        numeric_value = float(value)
        if not isfinite(numeric_value):
            raise EmbeddingProviderError(
                f"Sparse {context} has a non-finite value at position {position}."
            )
        seen_indices.add(index)
        pairs.append((index, numeric_value))

    pairs.sort(key=lambda item: item[0])
    return SparseEmbeddingVector(
        indices=tuple(index for index, _ in pairs),
        values=tuple(value for _, value in pairs),
    )


def _validate_non_empty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidEmbeddingConfigurationError(
            f"{name} must be a non-empty string."
        )


def _validate_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidEmbeddingConfigurationError(f"{name} must be an integer.")
    if value <= 0:
        raise InvalidEmbeddingConfigurationError(
            f"{name} must be greater than zero."
        )
