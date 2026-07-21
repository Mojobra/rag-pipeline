"""Discover and load supported local documents into the RAG pipeline.

The module owns deterministic filesystem discovery and delegates format-specific
text extraction while keeping directory traversal separate from parsing.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from pathlib import Path

from langchain_core.documents import Document

from rag_pipeline.exceptions import (
    IngestionError,
    IngestionPathNotFoundError,
    UnsupportedDocumentTypeError,
)
from rag_pipeline.extraction import SUPPORTED_FILE_EXTENSIONS, extract_documents


PathInput = str | Path

TEXT_FILE_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".html", ".htm"})


def discover_files(
    paths: Iterable[PathInput],
    *,
    recursive: bool = True,
    allowed_extensions: Collection[str] | None = None,
) -> list[Path]:
    """Resolve supported files from explicit paths and directory trees.

    Directory traversal is recursive by default. Results are deduplicated and
    sorted for reproducible indexing; missing paths and explicitly unsupported
    files raise ingestion errors. This function reads filesystem metadata but
    does not open document contents.
    """
    extensions = _normalize_extensions(allowed_extensions or SUPPORTED_FILE_EXTENSIONS)
    discovered: set[Path] = set()

    for raw_path in paths:
        path = Path(raw_path).expanduser()

        if not path.exists():
            raise IngestionPathNotFoundError(f"Input path does not exist: {path}")

        if path.is_file():
            _ensure_supported_file(path, extensions)
            discovered.add(path.resolve())
            continue

        if path.is_dir():
            candidates = path.rglob("*") if recursive else path.iterdir()
            discovered.update(
                candidate.resolve()
                for candidate in candidates
                if candidate.is_file() and _is_supported_file(candidate, extensions)
            )
            continue

        raise IngestionError(f"Input path is neither a file nor directory: {path}")

    return sorted(discovered, key=lambda item: str(item).lower())


def load_documents(
    paths: Iterable[PathInput],
    *,
    recursive: bool = True,
    encoding: str = "utf-8",
) -> list[Document]:
    """Extract supported local files into ordered LangChain documents.

    Files are first discovered deterministically, then opened by the extraction
    layer. A PDF may produce several page documents, so the returned count can
    exceed the number of files. The function performs filesystem I/O and lets
    stage-specific ingestion or extraction errors propagate.
    """
    documents = []

    for file_path in discover_files(paths, recursive=recursive):
        documents.extend(extract_documents(file_path, encoding=encoding))

    return documents


def _normalize_extensions(extensions: Collection[str]) -> frozenset[str]:
    return frozenset(
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    )


def _ensure_supported_file(path: Path, extensions: Collection[str]) -> None:
    if not _is_supported_file(path, extensions):
        supported = ", ".join(sorted(extensions))
        raise UnsupportedDocumentTypeError(
            f"Unsupported document type for {path}. Supported types: {supported}"
        )


def _is_supported_file(path: Path, extensions: Collection[str]) -> bool:
    return path.suffix.lower() in extensions
