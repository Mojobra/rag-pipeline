"""Extract supported business files into provenance-rich text documents.

Format-specific readers are normalized to LangChain ``Document`` objects so
chunking and retrieval do not depend on parser-specific response types.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from rag_pipeline.exceptions import TextExtractionError, UnsupportedDocumentTypeError

PathInput = str | Path

TEXT_FILE_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".html", ".htm"})
HTML_FILE_EXTENSIONS = frozenset({".html", ".htm"})
PDF_FILE_EXTENSIONS = frozenset({".pdf"})
WORD_FILE_EXTENSIONS = frozenset({".docx"})
SUPPORTED_FILE_EXTENSIONS = (
    TEXT_FILE_EXTENSIONS | PDF_FILE_EXTENSIONS | WORD_FILE_EXTENSIONS
)


def extract_documents(path: PathInput, *, encoding: str = "utf-8") -> list[Document]:
    """Read one supported file into LangChain documents with source metadata.

    Text-like, HTML, and DOCX inputs produce one document, while PDFs produce
    one per page for citations. HTML offsets refer to normalized visible text,
    not raw markup positions. Filesystem and decoding errors propagate for text;
    PDF/DOCX parser failures use the ingestion exception hierarchy.
    """
    file_path = Path(path).expanduser().resolve()
    suffix = file_path.suffix.lower()

    if suffix in HTML_FILE_EXTENSIONS:
        return [_extract_html_file(file_path, encoding=encoding)]

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

    The caller supplies the decoding to use. Plain text and Markdown remain
    unchanged; filesystem and decoding errors propagate to the caller.
    """
    return Document(
        page_content=path.read_text(encoding=encoding),
        metadata=_build_metadata(path, extractor="text"),
    )


def _extract_html_file(path: Path, *, encoding: str) -> Document:
    """Keep visible HTML text while discarding markup and page chrome.

    The parser preserves block and table-cell boundaries in normalized text.
    Result offsets used by chunking and citations refer to that text, not to
    byte positions in the HTML file. Reading and decoding can raise I/O errors.
    """
    parser = _VisibleHtmlParser()
    parser.feed(path.read_text(encoding=encoding))
    parser.close()
    return Document(
        page_content=parser.visible_text(),
        metadata=_build_metadata(path, extractor="html.parser"),
    )


class _VisibleHtmlParser(HTMLParser):
    """Extract readable policy text through Python's standard HTML parser.

    The parser omits non-content elements and navigation chrome while
    retaining heading, paragraph, list, and table boundaries. It does not
    attempt CSS visibility analysis or visual PDF-style layout reconstruction.
    """

    _IGNORED_TAGS = frozenset(
        {"head", "script", "style", "noscript", "nav", "template"}
    )
    _BLOCK_TAGS = frozenset(
        {
            "article",
            "aside",
            "blockquote",
            "br",
            "dd",
            "div",
            "dl",
            "dt",
            "footer",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "hr",
            "li",
            "main",
            "ol",
            "p",
            "section",
            "table",
            "tr",
            "ul",
        }
    )
    _VOID_TAGS = frozenset(
        {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._ignored_depth = 0
        self._table_cell_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track omitted subtrees and insert boundaries for visible blocks."""
        if self._ignored_depth:
            if tag not in self._VOID_TAGS:
                self._ignored_depth += 1
            return
        if tag in self._IGNORED_TAGS:
            self._ignored_depth = 1
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")
        if tag == "tr":
            self._table_cell_count = 0
        elif tag in ("th", "td"):
            if self._table_cell_count:
                self._parts.append(" | ")
            self._table_cell_count += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Process self-closing breaks without leaving ignored depth open."""
        if not self._ignored_depth and tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Close omitted subtrees or terminate a visible block."""
        if self._ignored_depth:
            self._ignored_depth -= 1
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Keep text nodes outside ignored elements for later normalization."""
        if not self._ignored_depth:
            self._parts.append(data)

    def visible_text(self) -> str:
        """Return non-empty lines with stable whitespace and decoded entities."""
        return "\n".join(
            normalized
            for line in "".join(self._parts).splitlines()
            if (normalized := " ".join(line.split()))
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
