"""Compatibility exports for deterministic source citations."""

from rag_pipeline.generation.citations import (
    DEFAULT_CITATION_EXCERPT_CHARACTERS,
    Citation,
    CitationConfig,
    build_citation,
    build_citations,
    format_citation,
)

__all__ = [
    "DEFAULT_CITATION_EXCERPT_CHARACTERS",
    "Citation",
    "CitationConfig",
    "build_citation",
    "build_citations",
    "format_citation",
]
