"""Stage a bounded Google Drive folder snapshot for the existing indexer."""

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
    GoogleDriveInputError,
    GoogleDriveProviderError,
    TextExtractionError,
)
from rag_pipeline.infrastructure.google_drive import (
    GoogleDriveFile,
    GoogleDriveImportLimits,
)
from rag_pipeline.infrastructure.vector_store import IndexingResult
from rag_pipeline.ingestion.extraction import extract_documents


class GoogleDriveFolderReader(Protocol):
    """Supply validated metadata and bounded content without coupling to HTTP."""

    def list_files(self, limits: GoogleDriveImportLimits) -> Sequence[GoogleDriveFile]:
        """Return a complete, bounded folder snapshot or fail."""

    def download_file(self, file: GoogleDriveFile, *, max_bytes: int) -> bytes:
        """Return at most max_bytes of a listed file or Docs export."""


def index_google_drive_folder(
    reader: GoogleDriveFolderReader,
    *,
    config: IndexingPipelineConfig,
    limits: GoogleDriveImportLimits,
) -> IndexingResult:
    """Stage all files before embedding and replacing their stable sources.

    Reimport replaces chunks for returned IDs, but does not remove files no
    longer visible remotely. Temporary plaintext is removed after extraction.
    """
    files = list(reader.list_files(limits))
    if not files:
        raise GoogleDriveInputError("No supported files were found in this folder.")
    if len(files) > limits.max_files:
        raise GoogleDriveInputError("Drive folder exceeds max_files.")

    documents: list[Document] = []
    sources: set[str] = set()
    total_bytes = 0
    for file in files:
        if not isinstance(file, GoogleDriveFile) or file.source in sources:
            raise GoogleDriveInputError(
                "Drive listing contains invalid or duplicate files."
            )
        sources.add(file.source)
        contents = reader.download_file(file, max_bytes=limits.max_file_bytes)
        if not isinstance(contents, bytes) or len(contents) > limits.max_file_bytes:
            raise GoogleDriveInputError("Drive reader returned oversized content.")
        total_bytes += len(contents)
        if total_bytes > limits.max_total_bytes:
            raise GoogleDriveInputError("Drive downloads exceed max_total_bytes.")
        documents.extend(_extract_remote_file(file, contents))

    return index_extracted_documents(documents, sources=sources, config=config)


def _extract_remote_file(file: GoogleDriveFile, contents: bytes) -> list[Document]:
    """Reuse local parsers while replacing temporary-path citation provenance."""
    try:
        with TemporaryDirectory(prefix="rag-google-drive-") as directory:
            staged_file = Path(directory) / f"document{file.extraction_suffix}"
            staged_file.write_bytes(contents)
            extracted = extract_documents(staged_file)
    except (OSError, UnicodeError, TextExtractionError) as exc:
        raise GoogleDriveProviderError(
            "Could not stage or extract a Google Drive file."
        ) from exc

    for document in extracted:
        document.metadata.update(
            {
                "source": file.source,
                "file_name": file.name,
                "file_stem": Path(file.name).stem,
                "google_drive_folder_id": file.root_folder_id,
                "google_drive_file_id": file.file_id,
                "google_drive_mime_type": file.mime_type,
            }
        )
    return extracted
