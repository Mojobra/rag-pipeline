"""Persistent local vector storage backed by Qdrant and LangChain."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from numbers import Real
from pathlib import Path
import sqlite3
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import (
    QdrantVectorStore,
    RetrievalMode,
    SparseEmbeddings,
)
from qdrant_client import QdrantClient, models

from rag_pipeline.embeddings import EmbeddedDocument
from rag_pipeline.exceptions import (
    InvalidVectorStoreConfigurationError,
    VectorStoreCollectionNotFoundError,
    VectorStoreCompatibilityError,
    VectorStoreInputError,
    VectorStoreProviderError,
)
from rag_pipeline.sparse_embeddings import SparseEmbeddingVector


DEFAULT_VECTOR_STORE_PATH = Path(".rag_data/qdrant")
DEFAULT_COLLECTION_NAME = "rag_documents"
VECTOR_STORE_SCHEMA_VERSION = 1
HYBRID_VECTOR_STORE_SCHEMA_VERSION = 2
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
_POINT_ID_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "https://example.local/rag-pipeline/chunk",
)


class SearchMode(str, Enum):
    """Vector representations available in one Qdrant collection."""

    DENSE = "dense"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class VectorStoreConfig:
    """Settings for the embedded Qdrant database."""

    path: str | Path | None = DEFAULT_VECTOR_STORE_PATH
    collection_name: str = DEFAULT_COLLECTION_NAME
    write_batch_size: int = 64
    search_mode: SearchMode | str = SearchMode.DENSE

    def __post_init__(self) -> None:
        if self.path is not None:
            if not isinstance(self.path, (str, Path)):
                raise InvalidVectorStoreConfigurationError(
                    "path must be a string, Path, or None."
                )
            if isinstance(self.path, str) and not self.path.strip():
                raise InvalidVectorStoreConfigurationError(
                    "path must be non-empty when provided."
                )
        if not isinstance(self.collection_name, str) or not self.collection_name.strip():
            raise InvalidVectorStoreConfigurationError(
                "collection_name must be a non-empty string."
            )
        if isinstance(self.write_batch_size, bool) or not isinstance(
            self.write_batch_size, int
        ):
            raise InvalidVectorStoreConfigurationError(
                "write_batch_size must be an integer."
            )
        if self.write_batch_size <= 0:
            raise InvalidVectorStoreConfigurationError(
                "write_batch_size must be greater than zero."
            )
        try:
            search_mode = SearchMode(self.search_mode)
        except (TypeError, ValueError) as exc:
            supported_modes = ", ".join(mode.value for mode in SearchMode)
            raise InvalidVectorStoreConfigurationError(
                f"search_mode must be one of: {supported_modes}."
            ) from exc
        object.__setattr__(self, "search_mode", search_mode)

    @property
    def resolved_path(self) -> Path | None:
        if self.path is None:
            return None
        return Path(self.path).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class IndexingResult:
    """Summary of one idempotent vector-store upsert."""

    collection_name: str
    indexed_count: int
    total_count: int
    point_ids: tuple[str, ...]
    embedding_model: str
    embedding_dimension: int | None
    search_mode: SearchMode
    sparse_embedding_model: str | None


class LocalVectorStore:
    """Manage a local Qdrant collection containing validated chunk vectors."""

    def __init__(self, config: VectorStoreConfig | None = None) -> None:
        self.config = config or VectorStoreConfig()
        path = self.config.resolved_path

        try:
            if path is None:
                self._client = QdrantClient(":memory:")
            else:
                path.mkdir(parents=True, exist_ok=True)
                # Avoid Qdrant's redundant SQLite thread-mode probe on Python 3.11+.
                self._client = QdrantClient(
                    path=str(path),
                    force_disable_check_same_thread=sqlite3.threadsafety == 3,
                )
        except Exception as exc:
            raise VectorStoreProviderError(
                "Failed to initialize the local Qdrant database."
            ) from exc

    def __enter__(self) -> LocalVectorStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Release local database resources and file locks."""
        self._client.close()

    def index(
        self,
        embedded_documents: Iterable[EmbeddedDocument],
        *,
        model_identifier: str,
        sparse_vectors: Iterable[SparseEmbeddingVector] | None = None,
        sparse_model_identifier: str | None = None,
    ) -> IndexingResult:
        """Upsert precomputed embeddings using deterministic chunk IDs."""
        _validate_model_identifier(model_identifier)
        records = list(embedded_documents)
        prepared_sparse_vectors = self._prepare_sparse_inputs(
            records,
            sparse_vectors=sparse_vectors,
            sparse_model_identifier=sparse_model_identifier,
        )

        if not records:
            return IndexingResult(
                collection_name=self.config.collection_name,
                indexed_count=0,
                total_count=self.count(),
                point_ids=(),
                embedding_model=model_identifier,
                embedding_dimension=None,
                search_mode=self.config.search_mode,
                sparse_embedding_model=sparse_model_identifier,
            )

        dimension, points, point_ids = self._prepare_points(
            records,
            model_identifier=model_identifier,
            sparse_vectors=prepared_sparse_vectors,
            sparse_model_identifier=sparse_model_identifier,
        )
        self._ensure_compatible_collection(
            model_identifier=model_identifier,
            dimension=dimension,
            sparse_model_identifier=sparse_model_identifier,
        )

        try:
            for start in range(0, len(points), self.config.write_batch_size):
                self._client.upsert(
                    collection_name=self.config.collection_name,
                    points=points[start : start + self.config.write_batch_size],
                    wait=True,
                )
        except Exception as exc:
            raise VectorStoreProviderError(
                f"Failed to upsert vectors into {self.config.collection_name}."
            ) from exc

        return IndexingResult(
            collection_name=self.config.collection_name,
            indexed_count=len(points),
            total_count=self.count(),
            point_ids=tuple(point_ids),
            embedding_model=model_identifier,
            embedding_dimension=dimension,
            search_mode=self.config.search_mode,
            sparse_embedding_model=sparse_model_identifier,
        )

    def count(self) -> int:
        """Return the number of points in the configured collection."""
        try:
            if not self._client.collection_exists(self.config.collection_name):
                return 0
            return int(
                self._client.count(
                    collection_name=self.config.collection_name,
                    exact=True,
                ).count
            )
        except Exception as exc:
            raise VectorStoreProviderError(
                f"Failed to count vectors in {self.config.collection_name}."
            ) from exc

    def as_langchain_vector_store(
        self,
        *,
        embedding: Embeddings | None = None,
        sparse_embedding: SparseEmbeddings | None = None,
        retrieval_mode: RetrievalMode = RetrievalMode.DENSE,
        validate: bool | None = None,
    ) -> QdrantVectorStore:
        """Expose the collection through LangChain's Qdrant integration."""
        try:
            if not self._client.collection_exists(self.config.collection_name):
                raise VectorStoreCollectionNotFoundError(
                    f"Collection {self.config.collection_name} does not exist."
                )
            if (
                self.config.search_mode == SearchMode.DENSE
                and retrieval_mode != RetrievalMode.DENSE
            ):
                raise VectorStoreCompatibilityError(
                    "Dense collections cannot be opened in hybrid retrieval mode."
                )
            should_validate = embedding is not None if validate is None else validate
            return QdrantVectorStore(
                client=self._client,
                collection_name=self.config.collection_name,
                embedding=embedding,
                distance=models.Distance.COSINE,
                retrieval_mode=retrieval_mode,
                vector_name=(
                    QdrantVectorStore.VECTOR_NAME
                    if self.config.search_mode == SearchMode.DENSE
                    else DENSE_VECTOR_NAME
                ),
                sparse_embedding=sparse_embedding,
                sparse_vector_name=SPARSE_VECTOR_NAME,
                validate_embeddings=should_validate,
                validate_collection_config=should_validate,
            )
        except (
            VectorStoreCollectionNotFoundError,
            VectorStoreCompatibilityError,
            VectorStoreProviderError,
        ):
            raise
        except Exception as exc:
            raise VectorStoreProviderError(
                "Failed to create the LangChain Qdrant vector store."
            ) from exc

    def validate_compatibility(
        self,
        *,
        model_identifier: str,
        dimension: int,
        sparse_model_identifier: str | None = None,
    ) -> None:
        """Validate an existing collection without creating or mutating it."""
        _validate_model_identifier(model_identifier)
        _validate_dimension(dimension)
        if self.config.search_mode == SearchMode.HYBRID:
            _validate_sparse_model_identifier(sparse_model_identifier)
        elif sparse_model_identifier is not None:
            raise VectorStoreInputError(
                "sparse_model_identifier is only valid for hybrid collections."
            )

        try:
            if not self._client.collection_exists(self.config.collection_name):
                raise VectorStoreCollectionNotFoundError(
                    f"Collection {self.config.collection_name} does not exist; "
                    "index documents before retrieval."
                )
            info = self._client.get_collection(self.config.collection_name)
        except VectorStoreCollectionNotFoundError:
            raise
        except Exception as exc:
            raise VectorStoreProviderError(
                f"Failed to inspect collection {self.config.collection_name}."
            ) from exc

        self._validate_collection_info(
            info,
            model_identifier=model_identifier,
            dimension=dimension,
            sparse_model_identifier=sparse_model_identifier,
        )

    def _prepare_sparse_inputs(
        self,
        records: list[EmbeddedDocument],
        *,
        sparse_vectors: Iterable[SparseEmbeddingVector] | None,
        sparse_model_identifier: str | None,
    ) -> list[SparseEmbeddingVector] | None:
        if self.config.search_mode == SearchMode.DENSE:
            if sparse_vectors is not None or sparse_model_identifier is not None:
                raise VectorStoreInputError(
                    "Sparse vectors are only valid for hybrid collections."
                )
            return None

        _validate_sparse_model_identifier(sparse_model_identifier)
        if sparse_vectors is None:
            raise VectorStoreInputError(
                "Hybrid indexing requires one sparse vector per document."
            )
        prepared_vectors = list(sparse_vectors)
        if len(prepared_vectors) != len(records):
            raise VectorStoreInputError(
                "Hybrid indexing received "
                f"{len(prepared_vectors)} sparse vector(s) for "
                f"{len(records)} document(s)."
            )
        normalized_vectors = []
        for index, vector in enumerate(prepared_vectors):
            if not isinstance(vector, SparseEmbeddingVector):
                raise TypeError(
                    f"sparse_vectors[{index}] must be a SparseEmbeddingVector."
                )
            normalized_vectors.append(_validate_sparse_vector(vector, index=index))
        return normalized_vectors

    def _prepare_points(
        self,
        records: list[EmbeddedDocument],
        *,
        model_identifier: str,
        sparse_vectors: list[SparseEmbeddingVector] | None,
        sparse_model_identifier: str | None,
    ) -> tuple[int, list[models.PointStruct], list[str]]:
        points = []
        point_ids = []
        seen_ids: set[str] = set()
        dimension: int | None = None

        for index, record in enumerate(records):
            if not isinstance(record, EmbeddedDocument):
                raise TypeError(
                    f"embedded_documents[{index}] must be an EmbeddedDocument."
                )
            document = record.document
            if not document.page_content.strip():
                raise VectorStoreInputError(
                    f"embedded_documents[{index}] contains empty page_content."
                )

            vector = _validate_vector(record.embedding, index=index)
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise VectorStoreInputError(
                    "Embedded documents have inconsistent vector dimensions."
                )

            point_id = build_chunk_point_id(document)
            if point_id in seen_ids:
                raise VectorStoreInputError(
                    f"Duplicate deterministic point ID at embedded_documents[{index}]."
                )
            seen_ids.add(point_id)

            metadata = _json_safe_metadata(document.metadata, index=index)
            metadata.update(
                {
                    "chunk_id": point_id,
                    "embedding_model": model_identifier,
                    "embedding_dimension": len(vector),
                }
            )
            if self.config.search_mode == SearchMode.HYBRID:
                if sparse_vectors is None:
                    raise VectorStoreInputError(
                        "Hybrid point preparation requires sparse vectors."
                    )
                metadata["sparse_embedding_model"] = sparse_model_identifier
                sparse_vector = sparse_vectors[index]
                point_vector: list[float] | dict[str, object] = {
                    DENSE_VECTOR_NAME: vector,
                }
                if not sparse_vector.is_empty:
                    point_vector[SPARSE_VECTOR_NAME] = models.SparseVector(
                        indices=list(sparse_vector.indices),
                        values=list(sparse_vector.values),
                    )
            else:
                point_vector = vector
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=point_vector,
                    payload={
                        QdrantVectorStore.CONTENT_KEY: document.page_content,
                        QdrantVectorStore.METADATA_KEY: metadata,
                    },
                )
            )
            point_ids.append(point_id)

        if dimension is None:
            raise VectorStoreInputError("No embedding dimension was available.")
        return dimension, points, point_ids

    def _ensure_compatible_collection(
        self,
        *,
        model_identifier: str,
        dimension: int,
        sparse_model_identifier: str | None,
    ) -> None:
        try:
            if not self._client.collection_exists(self.config.collection_name):
                if self.config.search_mode == SearchMode.DENSE:
                    self._client.create_collection(
                        collection_name=self.config.collection_name,
                        vectors_config=models.VectorParams(
                            size=dimension,
                            distance=models.Distance.COSINE,
                        ),
                        metadata={
                            "rag_pipeline_schema_version": (
                                VECTOR_STORE_SCHEMA_VERSION
                            ),
                            "embedding_model": model_identifier,
                            "embedding_dimension": dimension,
                        },
                    )
                else:
                    self._client.create_collection(
                        collection_name=self.config.collection_name,
                        vectors_config={
                            DENSE_VECTOR_NAME: models.VectorParams(
                                size=dimension,
                                distance=models.Distance.COSINE,
                            )
                        },
                        sparse_vectors_config={
                            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                                modifier=models.Modifier.IDF,
                            )
                        },
                        metadata={
                            "rag_pipeline_schema_version": (
                                HYBRID_VECTOR_STORE_SCHEMA_VERSION
                            ),
                            "search_mode": SearchMode.HYBRID.value,
                            "embedding_model": model_identifier,
                            "embedding_dimension": dimension,
                            "sparse_embedding_model": sparse_model_identifier,
                            "dense_vector_name": DENSE_VECTOR_NAME,
                            "sparse_vector_name": SPARSE_VECTOR_NAME,
                            "fusion": "rrf",
                        },
                    )
                return

        except Exception as exc:
            raise VectorStoreProviderError(
                f"Failed to inspect collection {self.config.collection_name}."
            ) from exc

        self.validate_compatibility(
            model_identifier=model_identifier,
            dimension=dimension,
            sparse_model_identifier=sparse_model_identifier,
        )

    def _validate_collection_info(
        self,
        info: Any,
        *,
        model_identifier: str,
        dimension: int,
        sparse_model_identifier: str | None,
    ) -> None:
        vectors_config = info.config.params.vectors
        if self.config.search_mode == SearchMode.DENSE:
            if not isinstance(vectors_config, models.VectorParams):
                raise VectorStoreCompatibilityError(
                    "The existing collection is not a dense schema v1 collection "
                    "with one unnamed vector."
                )
            dense_params = vectors_config
            expected_metadata = {
                "rag_pipeline_schema_version": VECTOR_STORE_SCHEMA_VERSION,
                "embedding_model": model_identifier,
                "embedding_dimension": dimension,
            }
        else:
            if (
                not isinstance(vectors_config, dict)
                or set(vectors_config) != {DENSE_VECTOR_NAME}
                or not isinstance(
                    vectors_config.get(DENSE_VECTOR_NAME),
                    models.VectorParams,
                )
            ):
                raise VectorStoreCompatibilityError(
                    "The existing collection is not a hybrid schema v2 collection "
                    f"with a {DENSE_VECTOR_NAME!r} dense vector."
                )
            dense_params = vectors_config[DENSE_VECTOR_NAME]
            sparse_config = info.config.params.sparse_vectors
            if (
                not isinstance(sparse_config, dict)
                or set(sparse_config) != {SPARSE_VECTOR_NAME}
                or not isinstance(
                    sparse_config.get(SPARSE_VECTOR_NAME),
                    models.SparseVectorParams,
                )
            ):
                raise VectorStoreCompatibilityError(
                    "The existing collection does not contain the required "
                    f"{SPARSE_VECTOR_NAME!r} sparse vector."
                )
            sparse_params = sparse_config[SPARSE_VECTOR_NAME]
            if sparse_params.modifier != models.Modifier.IDF:
                raise VectorStoreCompatibilityError(
                    "The existing sparse vector does not use Qdrant's IDF modifier."
                )
            expected_metadata = {
                "rag_pipeline_schema_version": HYBRID_VECTOR_STORE_SCHEMA_VERSION,
                "search_mode": SearchMode.HYBRID.value,
                "embedding_model": model_identifier,
                "embedding_dimension": dimension,
                "sparse_embedding_model": sparse_model_identifier,
                "dense_vector_name": DENSE_VECTOR_NAME,
                "sparse_vector_name": SPARSE_VECTOR_NAME,
                "fusion": "rrf",
            }

        if dense_params.size != dimension:
            raise VectorStoreCompatibilityError(
                f"Collection dimension is {dense_params.size}, but incoming "
                f"embeddings use {dimension}."
            )
        if dense_params.distance != models.Distance.COSINE:
            raise VectorStoreCompatibilityError(
                "The existing collection does not use cosine distance."
            )

        metadata = info.config.metadata or {}
        for key, expected_value in expected_metadata.items():
            actual_value = metadata.get(key)
            if actual_value != expected_value:
                raise VectorStoreCompatibilityError(
                    f"Collection metadata {key!r} is {actual_value!r}; "
                    f"expected {expected_value!r}."
                )


def build_chunk_point_id(document: Document) -> str:
    """Build a stable UUID for a chunk's logical source location."""
    if not isinstance(document, Document):
        raise TypeError("document must be a LangChain Document object.")

    source = document.metadata.get("source")
    chunk_index = document.metadata.get("chunk_index")
    identity = {
        "source": str(source) if source is not None else None,
        "page": document.metadata.get("page"),
        "chunk_index": chunk_index,
    }

    if source is None or isinstance(chunk_index, bool) or not isinstance(
        chunk_index, int
    ):
        identity["content_sha256"] = sha256(
            document.page_content.encode("utf-8")
        ).hexdigest()

    canonical_identity = json.dumps(
        identity,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return str(uuid5(_POINT_ID_NAMESPACE, canonical_identity))


def _validate_vector(vector: Iterable[Real], *, index: int) -> list[float]:
    values = list(vector)
    if not values:
        raise VectorStoreInputError(
            f"embedded_documents[{index}] has an empty vector."
        )

    normalized = []
    for value_index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise VectorStoreInputError(
                f"embedded_documents[{index}] has a non-numeric vector value "
                f"at index {value_index}."
            )
        numeric_value = float(value)
        if not isfinite(numeric_value):
            raise VectorStoreInputError(
                f"embedded_documents[{index}] has a non-finite vector value "
                f"at index {value_index}."
            )
        normalized.append(numeric_value)
    return normalized


def _validate_sparse_vector(
    vector: SparseEmbeddingVector,
    *,
    index: int,
) -> SparseEmbeddingVector:
    if len(vector.indices) != len(vector.values):
        raise VectorStoreInputError(
            f"sparse_vectors[{index}] has different index and value counts."
        )

    pairs = []
    seen_indices: set[int] = set()
    for value_index, (sparse_index, value) in enumerate(
        zip(vector.indices, vector.values, strict=True)
    ):
        if (
            isinstance(sparse_index, bool)
            or not isinstance(sparse_index, int)
            or sparse_index < 0
        ):
            raise VectorStoreInputError(
                f"sparse_vectors[{index}] has an invalid sparse index at "
                f"position {value_index}."
            )
        if sparse_index in seen_indices:
            raise VectorStoreInputError(
                f"sparse_vectors[{index}] contains duplicate sparse index "
                f"{sparse_index}."
            )
        if isinstance(value, bool) or not isinstance(value, Real):
            raise VectorStoreInputError(
                f"sparse_vectors[{index}] has a non-numeric value at "
                f"position {value_index}."
            )
        numeric_value = float(value)
        if not isfinite(numeric_value):
            raise VectorStoreInputError(
                f"sparse_vectors[{index}] has a non-finite value at "
                f"position {value_index}."
            )
        seen_indices.add(sparse_index)
        pairs.append((sparse_index, numeric_value))

    pairs.sort(key=lambda item: item[0])
    return SparseEmbeddingVector(
        indices=tuple(sparse_index for sparse_index, _ in pairs),
        values=tuple(value for _, value in pairs),
    )


def _json_safe_metadata(metadata: dict[str, Any], *, index: int) -> dict[str, Any]:
    try:
        serialized = json.dumps(metadata, ensure_ascii=True, allow_nan=False)
        normalized = json.loads(serialized)
    except (TypeError, ValueError) as exc:
        raise VectorStoreInputError(
            f"embedded_documents[{index}] metadata must be JSON-serializable."
        ) from exc

    if not isinstance(normalized, dict):
        raise VectorStoreInputError(
            f"embedded_documents[{index}] metadata must be a dictionary."
        )
    return normalized


def _validate_model_identifier(model_identifier: object) -> None:
    if not isinstance(model_identifier, str) or not model_identifier.strip():
        raise VectorStoreInputError(
            "model_identifier must be a non-empty string."
        )


def _validate_sparse_model_identifier(model_identifier: object) -> None:
    if not isinstance(model_identifier, str) or not model_identifier.strip():
        raise VectorStoreInputError(
            "sparse_model_identifier must be a non-empty string for hybrid mode."
        )


def _validate_dimension(dimension: object) -> None:
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise VectorStoreInputError("dimension must be an integer.")
    if dimension <= 0:
        raise VectorStoreInputError("dimension must be greater than zero.")
