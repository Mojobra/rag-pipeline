"""Split extracted documents into configurable retrieval chunks.

This module owns the character-based LangChain splitting policy and preserves
the source positions required by deterministic indexing and citations.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        _validate_integer("chunk_size", self.chunk_size)
        _validate_integer("chunk_overlap", self.chunk_overlap)

        if self.chunk_size <= 0:
            raise InvalidChunkingConfigurationError(
                "chunk_size must be greater than zero."
            )
        if self.chunk_overlap < 0:
            raise InvalidChunkingConfigurationError(
                "chunk_overlap cannot be negative."
            )
        if self.chunk_overlap >= self.chunk_size:
            raise InvalidChunkingConfigurationError(
                "chunk_overlap must be smaller than chunk_size."
            )


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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks: list[Document] = []

    for document in documents:
        if not isinstance(document, Document):
            raise TypeError("documents must contain LangChain Document objects.")
        if not document.page_content.strip():
            continue

        document_chunks = splitter.split_documents([document])
        chunk_count = len(document_chunks)

        for chunk_index, chunk in enumerate(document_chunks):
            start_index = chunk.metadata.get("start_index")
            if not isinstance(start_index, int) or start_index < 0:
                raise ChunkingError("LangChain did not provide a valid start_index.")

            chunk.metadata.update(
                {
                    "chunk_index": chunk_index,
                    "chunk_count": chunk_count,
                    "end_index": start_index + len(chunk.page_content),
                    "chunk_char_count": len(chunk.page_content),
                }
            )
            chunks.append(chunk)

    return chunks


def _validate_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidChunkingConfigurationError(f"{name} must be an integer.")
