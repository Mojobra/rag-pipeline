"""Stage one bounded SharePoint drive snapshot for the existing indexer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from langchain_core.documents import Document

from rag_pipeline.application.indexing import (
    IndexingPipelineConfig,
    index_extracted_documents,
)
from rag_pipeline.exceptions import (
    SharePointInputError,
    SharePointProviderError,
    TextExtractionError,
)
from rag_pipeline.infrastructure.sharepoint import (
    SharePointFile,
    SharePointImportLimits,
)
from rag_pipeline.infrastructure.vector_store import IndexingResult
from rag_pipeline.ingestion.extraction import extract_documents


class SharePointDriveReader(Protocol):
    """Supply validated Graph file metadata and bounded bytes to the importer."""

    def list_files(self, limits: SharePointImportLimits) -> Sequence[SharePointFile]:
        """Return a complete, bounded drive snapshot or fail."""

    def download_file(self, file: SharePointFile, *, max_bytes: int) -> bytes:
        """Return at most max_bytes from one listed file."""


def index_sharepoint_drive(
    reader: SharePointDriveReader,
    *,
    config: IndexingPipelineConfig,
    limits: SharePointImportLimits,
) -> IndexingResult:
    """Import supported Graph files without changing other indexed sources.

    All remote listing, downloads, and extraction finish before model inference
    or Qdrant writes begin. Stable drive/item IDs replace chunks on rerun, but
    missing remote items are deliberately not pruned without an ACL-aware sync.
    Temporary files are removed after each extraction.
    """
    files = list(reader.list_files(limits))
    if not files:
        raise SharePointInputError("No supported files were found in this drive.")
    if len(files) > limits.max_files:
        raise SharePointInputError("SharePoint drive exceeds max_files.")

    documents: list[Document] = []
    sources: set[str] = set()
    total_bytes = 0
    for file in files:
        if not isinstance(file, SharePointFile) or file.source in sources:
            raise SharePointInputError(
                "SharePoint listing contains invalid or duplicate files."
            )
        sources.add(file.source)
        contents = reader.download_file(file, max_bytes=limits.max_file_bytes)
        if not isinstance(contents, bytes) or len(contents) > limits.max_file_bytes:
            raise SharePointInputError("SharePoint reader returned oversized content.")
        total_bytes += len(contents)
        if total_bytes > limits.max_total_bytes:
            raise SharePointInputError("SharePoint downloads exceed max_total_bytes.")
        documents.extend(_extract_remote_file(file, contents))

    return index_extracted_documents(documents, sources=sources, config=config)


def _extract_remote_file(file: SharePointFile, contents: bytes) -> list[Document]:
    """Reuse local format parsers while replacing temporary path provenance."""
    suffix = Path(file.name).suffix.lower()
    try:
        with TemporaryDirectory(prefix="rag-sharepoint-") as directory:
            staged_file = Path(directory) / f"document{suffix}"
            staged_file.write_bytes(contents)
            extracted = extract_documents(staged_file)
    except (OSError, UnicodeError, TextExtractionError) as exc:
        raise SharePointProviderError(
            "Could not stage or extract a SharePoint file."
        ) from exc

    for document in extracted:
        document.metadata.update(
            {
                "source": file.source,
                "file_name": file.name,
                "file_stem": Path(file.name).stem,
                "sharepoint_drive_id": file.drive_id,
                "sharepoint_item_id": file.item_id,
            }
        )
        if file.web_url is not None:
            document.metadata["sharepoint_web_url"] = file.web_url
    return extracted
