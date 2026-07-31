"""Compatibility exports for document extraction implementations."""

from rag_pipeline.ingestion.extraction import (
    PDF_FILE_EXTENSIONS,
    SUPPORTED_FILE_EXTENSIONS,
    TEXT_FILE_EXTENSIONS,
    WORD_FILE_EXTENSIONS,
    PathInput,
    extract_documents,
)

__all__ = [
    "PDF_FILE_EXTENSIONS",
    "SUPPORTED_FILE_EXTENSIONS",
    "TEXT_FILE_EXTENSIONS",
    "WORD_FILE_EXTENSIONS",
    "PathInput",
    "extract_documents",
]
