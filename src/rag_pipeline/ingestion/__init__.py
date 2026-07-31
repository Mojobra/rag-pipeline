"""Public document-ingestion API and feature package boundary.

Filesystem loading remains available from its established package path while
extraction, chunking, and experiments live in focused sibling modules.
"""

from rag_pipeline.exceptions import (
    IngestionError,
    IngestionPathNotFoundError,
    UnsupportedDocumentTypeError,
)
from rag_pipeline.ingestion.extraction import (
    SUPPORTED_FILE_EXTENSIONS,
    extract_documents,
)
from rag_pipeline.ingestion.loading import (
    TEXT_FILE_EXTENSIONS,
    PathInput,
    discover_files,
    load_documents,
)

__all__ = [
    "SUPPORTED_FILE_EXTENSIONS",
    "TEXT_FILE_EXTENSIONS",
    "IngestionError",
    "IngestionPathNotFoundError",
    "PathInput",
    "UnsupportedDocumentTypeError",
    "discover_files",
    "extract_documents",
    "load_documents",
]
