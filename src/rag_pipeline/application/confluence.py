"""Turn a bounded Confluence space snapshot into local RAG index sources."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_core.documents import Document

from rag_pipeline.application.indexing import (
    IndexingPipelineConfig,
    index_extracted_documents,
)
from rag_pipeline.exceptions import ConfluenceInputError
from rag_pipeline.infrastructure.confluence import (
    ConfluenceImportLimits,
    ConfluencePage,
)
from rag_pipeline.infrastructure.vector_store import IndexingResult
from rag_pipeline.ingestion import html_to_visible_text


class ConfluenceSpaceReader(Protocol):
    """Supply a validated page traversal without exposing transport."""

    def list_pages(
        self, space_id: str, limits: ConfluenceImportLimits
    ) -> Sequence[ConfluencePage]:
        """Return all currently visible pages in the bounded space."""


def index_confluence_space(
    reader: ConfluenceSpaceReader,
    *,
    space_id: str,
    config: IndexingPipelineConfig,
    limits: ConfluenceImportLimits,
) -> IndexingResult:
    """Index one completed page traversal without pruning absent remote pages.

    All HTTP reads and HTML extraction finish before model inference or Qdrant
    writes. Re-listed page IDs replace their previous chunks, while pages not
    returned in this run remain until an explicit future sync policy exists.
    """
    pages = list(reader.list_pages(space_id, limits))
    if not pages:
        raise ConfluenceInputError("No current pages were found in this space.")
    if len(pages) > limits.max_pages:
        raise ConfluenceInputError("Confluence space exceeds max_pages.")

    documents: list[Document] = []
    sources: set[str] = set()
    total_bytes = 0
    for page in pages:
        if (
            not isinstance(page, ConfluencePage)
            or page.space_id != space_id
            or page.source in sources
        ):
            raise ConfluenceInputError(
                "Confluence listing contains invalid or duplicate pages."
            )
        source_bytes = len(page.body_html.encode("utf-8"))
        if source_bytes > limits.max_page_bytes:
            raise ConfluenceInputError("A Confluence page exceeds max_page_bytes.")
        total_bytes += source_bytes
        if total_bytes > limits.max_total_bytes:
            raise ConfluenceInputError("Confluence pages exceed max_total_bytes.")
        sources.add(page.source)
        visible_text = html_to_visible_text(page.body_html)
        metadata: dict[str, str | int] = {
            "source": page.source,
            "title": page.title,
            "confluence_site": page.site_host,
            "confluence_space_id": page.space_id,
            "confluence_page_id": page.page_id,
            "confluence_version": page.version,
            "extractor": "html.parser",
        }
        if page.web_url is not None:
            metadata["confluence_web_url"] = page.web_url
        documents.append(
            Document(
                page_content=f"{page.title}\n\n{visible_text}".strip(),
                metadata=metadata,
            )
        )

    return index_extracted_documents(documents, sources=sources, config=config)
