"""Split extracted documents with recursive or structure-aware policies.

This module owns deterministic LangChain character splitting and preserves the
source positions required by indexing, evaluation, and citations. Embedding-
aware semantic splitting lives in :mod:`rag_pipeline.ingestion.semantic_chunking`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from rag_pipeline.exceptions import ChunkingError, InvalidChunkingConfigurationError


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Validated character-splitting settings for indexing workflows.

    The configuration supplies LangChain's maximum chunk length and target
    overlap. Construction rejects ranges that would make splitting ambiguous or
    invalid before any document processing begins.
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200

    def __post_init__(self) -> None:
        """Reject non-integer or unsafe splitter ranges at construction time.

        Overlap may be zero but must remain strictly smaller than chunk size.
        """
        _validate_chunking_limits(self.chunk_size, self.chunk_overlap)


@dataclass(frozen=True, slots=True)
class StructureAwareChunkingConfig:
    """Settings for markup-aware recursive splitting in benchmark experiments.

    Markdown and HTML documents use LangChain's language-specific separator
    priorities; other formats fall back to the normal recursive policy. The
    size and overlap limits keep output bounded and comparable.
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200

    def __post_init__(self) -> None:
        """Enforce the same hard size contract as recursive chunking."""
        _validate_chunking_limits(self.chunk_size, self.chunk_overlap)


def chunk_documents(
    documents: Iterable[Document],
    *,
    config: ChunkingConfig | None = None,
) -> list[Document]:
    """Create retrieval chunks while preserving source provenance.

    Blank documents are skipped. Each returned LangChain document inherits its
    source metadata and gains stable chunk counts plus start/end character
    positions. Invalid input types fail before they can reach embedding or
    persistent indexing stages.
    """
    settings = config or ChunkingConfig()
    splitter = _recursive_splitter(settings)
    return _split_documents(
        documents,
        splitter_for_document=lambda _document: (splitter, None),
    )


def chunk_documents_with_structure(
    documents: Iterable[Document],
    *,
    config: StructureAwareChunkingConfig | None = None,
) -> list[Document]:
    """Split markup at structural boundaries before smaller text boundaries.

    Markdown and HTML use LangChain's language-aware recursive separator lists,
    while PDF, DOCX, plain text, and unknown metadata use the baseline policy.
    The function performs no model or storage I/O and preserves exact character
    offsets for every emitted chunk.
    """
    settings = config or StructureAwareChunkingConfig()
    return _split_documents(
        documents,
        splitter_for_document=lambda document: _structure_splitter(
            document,
            settings,
        ),
        strategy="structure_aware",
    )


def _split_documents(
    documents: Iterable[Document],
    *,
    splitter_for_document: Callable[
        [Document],
        tuple[RecursiveCharacterTextSplitter, str | None],
    ],
    strategy: str | None = None,
) -> list[Document]:
    """Apply one splitter per source document and attach stable provenance."""
    chunks: list[Document] = []
    for document in documents:
        if not isinstance(document, Document):
            raise TypeError("documents must contain LangChain Document objects.")
        if not document.page_content.strip():
            continue

        splitter, structure_language = splitter_for_document(document)
        document_chunks = splitter.split_documents([document])
        chunks.extend(
            _annotate_chunks(
                document_chunks,
                strategy=strategy,
                structure_language=structure_language,
            )
        )
    return chunks


def _annotate_chunks(
    chunks: list[Document],
    *,
    strategy: str | None,
    structure_language: str | None,
) -> list[Document]:
    """Validate LangChain offsets and add per-document chunk metadata in place."""
    chunk_count = len(chunks)
    for chunk_index, chunk in enumerate(chunks):
        start_index = chunk.metadata.get("start_index")
        if (
            isinstance(start_index, bool)
            or not isinstance(start_index, int)
            or start_index < 0
        ):
            raise ChunkingError("LangChain did not provide a valid start_index.")

        chunk.metadata.update(
            {
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "end_index": start_index + len(chunk.page_content),
                "chunk_char_count": len(chunk.page_content),
            }
        )
        if strategy is not None:
            chunk.metadata["chunking_strategy"] = strategy
            chunk.metadata["structure_language"] = (
                structure_language or "recursive_fallback"
            )
    return chunks


def _recursive_splitter(
    config: ChunkingConfig | StructureAwareChunkingConfig,
) -> RecursiveCharacterTextSplitter:
    """Create the stable baseline splitter without performing document work."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
        add_start_index=True,
    )


def _structure_splitter(
    document: Document,
    config: StructureAwareChunkingConfig,
) -> tuple[RecursiveCharacterTextSplitter, str | None]:
    """Select a markup-aware LangChain splitter from source extension metadata."""
    extension = document.metadata.get("file_extension")
    language_by_extension = {
        ".md": Language.MARKDOWN,
        ".markdown": Language.MARKDOWN,
        ".html": Language.HTML,
        ".htm": Language.HTML,
    }
    language = (
        language_by_extension.get(extension.casefold())
        if isinstance(extension, str)
        else None
    )
    if language is None:
        return _recursive_splitter(config), None
    return (
        RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            add_start_index=True,
        ),
        language.value,
    )


def _validate_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidChunkingConfigurationError(f"{name} must be an integer.")


def _validate_chunking_limits(chunk_size: object, chunk_overlap: object) -> None:
    """Validate shared hard-size and overlap invariants for recursive policies."""
    _validate_integer("chunk_size", chunk_size)
    _validate_integer("chunk_overlap", chunk_overlap)
    if not isinstance(chunk_size, int) or not isinstance(chunk_overlap, int):
        return
    if chunk_size <= 0:
        raise InvalidChunkingConfigurationError("chunk_size must be greater than zero.")
    if chunk_overlap < 0:
        raise InvalidChunkingConfigurationError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise InvalidChunkingConfigurationError(
            "chunk_overlap must be smaller than chunk_size."
        )
