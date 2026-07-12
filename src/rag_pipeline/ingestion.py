"""Local document ingestion for the RAG pipeline."""

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
    """Discover supported files from explicit file and directory inputs."""
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
    """Load local supported files into LangChain Document objects."""
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
