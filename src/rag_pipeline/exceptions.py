"""Shared exceptions for the RAG pipeline."""

from __future__ import annotations


class RagPipelineError(Exception):
    """Base error for pipeline failures."""


class IngestionError(RagPipelineError):
    """Base error for document ingestion failures."""


class IngestionPathNotFoundError(IngestionError):
    """Raised when an input path does not exist."""


class UnsupportedDocumentTypeError(IngestionError):
    """Raised when a document type is not supported yet."""


class TextExtractionError(IngestionError):
    """Raised when text extraction fails for a supported document type."""


class ChunkingError(RagPipelineError):
    """Base error for document chunking failures."""


class InvalidChunkingConfigurationError(ChunkingError, ValueError):
    """Raised when chunk size or overlap settings are invalid."""


class EmbeddingError(RagPipelineError):
    """Base error for embedding generation failures."""


class InvalidEmbeddingConfigurationError(EmbeddingError, ValueError):
    """Raised when local embedding settings are invalid."""


class EmbeddingInputError(EmbeddingError, ValueError):
    """Raised when content cannot safely be sent to an embedding model."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when an embedding provider fails or returns invalid vectors."""


class VectorStoreError(RagPipelineError):
    """Base error for vector database failures."""


class InvalidVectorStoreConfigurationError(VectorStoreError, ValueError):
    """Raised when local vector-store settings are invalid."""


class VectorStoreInputError(VectorStoreError, ValueError):
    """Raised when embedded documents are not safe to index."""


class VectorStoreCompatibilityError(VectorStoreError):
    """Raised when an existing collection uses an incompatible configuration."""


class VectorStoreCollectionNotFoundError(VectorStoreError):
    """Raised when a requested vector collection has not been indexed yet."""


class VectorStoreProviderError(VectorStoreError):
    """Raised when the vector database client fails."""


class RetrievalError(RagPipelineError):
    """Base error for semantic retrieval failures."""


class InvalidRetrievalConfigurationError(RetrievalError, ValueError):
    """Raised when retrieval limits or thresholds are invalid."""


class RetrievalInputError(RetrievalError, ValueError):
    """Raised when a retrieval query is invalid."""


class RetrievalProviderError(RetrievalError):
    """Raised when a vector search fails or returns an invalid response."""


class GenerationError(RagPipelineError):
    """Base error for grounded answer-generation failures."""


class InvalidGenerationConfigurationError(GenerationError, ValueError):
    """Raised when local model or context settings are invalid."""


class GenerationInputError(GenerationError, ValueError):
    """Raised when a question or retrieved context is invalid."""


class GenerationProviderError(GenerationError):
    """Raised when a language model fails or returns invalid output."""


class CitationError(RagPipelineError):
    """Base error for answer citation failures."""


class InvalidCitationConfigurationError(CitationError, ValueError):
    """Raised when citation formatting settings are invalid."""


class CitationInputError(CitationError, ValueError):
    """Raised when retrieved evidence lacks valid citation provenance."""
