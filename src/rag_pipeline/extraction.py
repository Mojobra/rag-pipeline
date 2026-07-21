"""Extract supported business files into provenance-rich text documents.

Format-specific readers are normalized to LangChain ``Document`` objects so
chunking and retrieval do not depend on parser-specific response types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from rag_pipeline.exceptions import TextExtractionError, UnsupportedDocumentTypeError


PathInput = str | Path

TEXT_FILE_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".html", ".htm"})
PDF_FILE_EXTENSIONS = frozenset({".pdf"})
WORD_FILE_EXTENSIONS = frozenset({".docx"})
SUPPORTED_FILE_EXTENSIONS = (
    TEXT_FILE_EXTENSIONS | PDF_FILE_EXTENSIONS | WORD_FILE_EXTENSIONS
)


def extract_documents(path: PathInput, *, encoding: str = "utf-8") -> list[Document]:
    """Read one supported file into LangChain documents with source metadata.

    Text-like and DOCX inputs produce one document, while PDFs produce one per
    page to preserve page-level citations. The function performs filesystem and
    parser I/O. Unsupported formats use the ingestion exception hierarchy;
    PDF/DOCX parser failures are wrapped, while plain-text I/O and decoding
    errors propagate from ``Path.read_text``.
    """
    file_path = Path(path).expanduser().resolve()
    suffix = file_path.suffix.lower()

    if suffix in TEXT_FILE_EXTENSIONS:
        return [_extract_text_file(file_path, encoding=encoding)]

    if suffix in PDF_FILE_EXTENSIONS:
        return _extract_pdf(file_path)

    if suffix in WORD_FILE_EXTENSIONS:
        return _extract_docx(file_path)

    supported = ", ".join(sorted(SUPPORTED_FILE_EXTENSIONS))
    raise UnsupportedDocumentTypeError(
        f"Unsupported document type for {file_path}. Supported types: {supported}"
    )


def _extract_text_file(path: Path, *, encoding: str) -> Document:
    """Read a text-like file without interpreting its markup.

    The caller supplies the decoding to use. Markdown and HTML therefore remain
    raw text at this stage rather than being semantically cleaned. Filesystem and
    decoding errors propagate to the caller.
    """
    return Document(
        page_content=path.read_text(encoding=encoding),
        metadata=_build_metadata(path, extractor="text"),
    )


def _extract_pdf(path: Path) -> list[Document]:
    """Extract one LangChain document per PDF page using pypdf.

    Page order and total-page metadata are preserved for citations. Pages with
    no text layer yield empty content for the chunking stage to skip; encrypted,
    corrupt, or otherwise unreadable PDFs raise ``TextExtractionError``.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise TextExtractionError("PDF extraction requires pypdf.") from exc

    try:
        reader = PdfReader(str(path))
        total_pages = len(reader.pages)
        documents = []

        for page_number, page in enumerate(reader.pages):
            documents.append(
                Document(
                    page_content=page.extract_text() or "",
                    metadata={
                        **_build_metadata(path, extractor="pypdf"),
                        "page": page_number,
                        "total_pages": total_pages,
                    },
                )
            )
    except Exception as exc:
        raise TextExtractionError(f"Failed to extract PDF text from {path}") from exc

    return documents


def _extract_docx(path: Path) -> list[Document]:
    """Extract a DOCX file as one logical document using docx2txt.

    Word pagination depends on rendering settings, so this path intentionally
    records file provenance without inventing page numbers. Parser failures are
    wrapped as ``TextExtractionError``.
    """
    try:
        import docx2txt
    except ImportError as exc:
        raise TextExtractionError("DOCX extraction requires docx2txt.") from exc

    try:
        content = docx2txt.process(str(path))
    except Exception as exc:
        raise TextExtractionError(f"Failed to extract DOCX text from {path}") from exc

    return [
        Document(
            page_content=content,
            metadata=_build_metadata(path, extractor="docx2txt"),
        )
    ]


def _build_metadata(path: Path, *, extractor: str) -> dict[str, Any]:
    """Capture filesystem provenance shared by every extraction backend.

    Reading ``Path.stat`` performs filesystem I/O. The resulting scalar values
    are suitable for persistence in the Qdrant metadata payload.
    """
    stat = path.stat()
    return {
        "source": str(path),
        "file_name": path.name,
        "file_stem": path.stem,
        "file_extension": path.suffix.lower(),
        "byte_size": stat.st_size,
        "extractor": extractor,
    }
