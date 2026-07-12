"""LangChain-based embedding generation for document chunks and queries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from numbers import Real

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_pipeline.exceptions import (
    EmbeddingInputError,
    EmbeddingProviderError,
    InvalidEmbeddingConfigurationError,
)


DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True, slots=True)
class LocalEmbeddingConfig:
    """Settings for the local Hugging Face embedding backend."""

    model_name: str = DEFAULT_LOCAL_EMBEDDING_MODEL
    model_revision: str | None = None
    device: str = "cpu"
    batch_size: int = 32
    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        _validate_non_empty_string("model_name", self.model_name)
        _validate_non_empty_string("device", self.device)

        if self.model_revision is not None:
            _validate_non_empty_string("model_revision", self.model_revision)
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise InvalidEmbeddingConfigurationError(
                "batch_size must be an integer."
            )
        if self.batch_size <= 0:
            raise InvalidEmbeddingConfigurationError(
                "batch_size must be greater than zero."
            )
        if not isinstance(self.normalize_embeddings, bool):
            raise InvalidEmbeddingConfigurationError(
                "normalize_embeddings must be a boolean."
            )


@dataclass(frozen=True, slots=True)
class EmbeddedDocument:
    """A source document paired with its immutable embedding vector."""

    document: Document
    embedding: tuple[float, ...]

    @property
    def dimension(self) -> int:
        """Return the number of values in the embedding vector."""
        return len(self.embedding)


class EmbeddingService:
    """Validate and coordinate a LangChain embedding model."""

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        model_name: str,
        model_revision: str | None = None,
    ) -> None:
        if not isinstance(embeddings, Embeddings):
            raise TypeError("embeddings must implement LangChain's Embeddings interface.")
        _validate_non_empty_string("model_name", model_name)
        if model_revision is not None:
            _validate_non_empty_string("model_revision", model_revision)

        self._embeddings = embeddings
        self._model_name = model_name
        self._model_revision = model_revision
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_revision(self) -> str | None:
        return self._model_revision

    @property
    def model_identifier(self) -> str:
        if self._model_revision is None:
            return self._model_name
        return f"{self._model_name}@{self._model_revision}"

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def embed_documents(
        self, documents: Iterable[Document]
    ) -> list[EmbeddedDocument]:
        """Embed non-empty documents in input order and validate the response."""
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
                f"Embedding model {self.model_identifier} failed for documents."
            ) from exc

        if len(provider_vectors) != len(source_documents):
            raise EmbeddingProviderError(
                "Embedding provider returned "
                f"{len(provider_vectors)} vector(s) for "
                f"{len(source_documents)} document(s)."
            )

        normalized_vectors = [
            _normalize_vector(vector, context=f"document vector {index}")
            for index, vector in enumerate(provider_vectors)
        ]
        dimensions = {len(vector) for vector in normalized_vectors}
        if len(dimensions) != 1:
            raise EmbeddingProviderError(
                "Embedding provider returned inconsistent document dimensions."
            )

        batch_dimension = dimensions.pop()
        self._set_or_validate_dimension(batch_dimension)
        return [
            EmbeddedDocument(document=document, embedding=vector)
            for document, vector in zip(
                source_documents, normalized_vectors, strict=True
            )
        ]

    def embed_query(self, query: str) -> tuple[float, ...]:
        """Embed one non-empty query with the same model and dimension contract."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not query.strip():
            raise EmbeddingInputError("query cannot be empty.")

        try:
            provider_vector = self._embeddings.embed_query(query)
        except Exception as exc:
            raise EmbeddingProviderError(
                f"Embedding model {self.model_identifier} failed for a query."
            ) from exc

        vector = _normalize_vector(provider_vector, context="query vector")
        self._set_or_validate_dimension(len(vector))
        return vector

    def _set_or_validate_dimension(self, dimension: int) -> None:
        if self._dimension is None:
            self._dimension = dimension
            return
        if dimension != self._dimension:
            raise EmbeddingProviderError(
                f"Embedding dimension changed from {self._dimension} to {dimension}."
            )


def create_local_embedding_service(
    config: LocalEmbeddingConfig | None = None,
) -> EmbeddingService:
    """Create the default local LangChain Hugging Face embedding service."""
    settings = config or LocalEmbeddingConfig()

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise EmbeddingProviderError(
            "Local embeddings require langchain-huggingface and sentence-transformers."
        ) from exc

    model_kwargs: dict[str, str] = {"device": settings.device}
    if settings.model_revision is not None:
        model_kwargs["revision"] = settings.model_revision

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={
                "batch_size": settings.batch_size,
                "normalize_embeddings": settings.normalize_embeddings,
            },
            query_encode_kwargs={
                "normalize_embeddings": settings.normalize_embeddings,
            },
            show_progress=False,
        )
    except Exception as exc:
        raise EmbeddingProviderError(
            f"Failed to initialize local embedding model {settings.model_name}."
        ) from exc

    return EmbeddingService(
        embeddings,
        model_name=settings.model_name,
        model_revision=settings.model_revision,
    )


def _normalize_vector(
    raw_vector: Iterable[Real],
    *,
    context: str,
) -> tuple[float, ...]:
    try:
        values = list(raw_vector)
    except TypeError as exc:
        raise EmbeddingProviderError(f"{context} is not iterable.") from exc

    if not values:
        raise EmbeddingProviderError(f"{context} is empty.")

    vector = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise EmbeddingProviderError(
                f"{context} contains a non-numeric value at index {index}."
            )
        numeric_value = float(value)
        if not isfinite(numeric_value):
            raise EmbeddingProviderError(
                f"{context} contains a non-finite value at index {index}."
            )
        vector.append(numeric_value)

    return tuple(vector)


def _validate_non_empty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidEmbeddingConfigurationError(
            f"{name} must be a non-empty string."
        )
