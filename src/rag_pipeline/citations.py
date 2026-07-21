"""Build deterministic citations from retrieved answer evidence.

Citation identity and locations come from validated retrieval metadata rather
than language-model output, keeping source attribution reproducible and auditable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

from rag_pipeline.exceptions import (
    CitationInputError,
    InvalidCitationConfigurationError,
)
from rag_pipeline.retrieval import RetrievalResult


DEFAULT_CITATION_EXCERPT_CHARACTERS = 240


@dataclass(frozen=True, slots=True)
class CitationConfig:
    """Validated display limits for citation evidence excerpts.

    The setting bounds rendered previews without changing the complete source
    metadata retained by each citation record.
    """

    max_excerpt_characters: int = DEFAULT_CITATION_EXCERPT_CHARACTERS

    def __post_init__(self) -> None:
        """Validate a limit large enough to represent truncated text safely."""
        if isinstance(self.max_excerpt_characters, bool) or not isinstance(
            self.max_excerpt_characters, int
        ):
            raise InvalidCitationConfigurationError(
                "max_excerpt_characters must be an integer."
            )
        if self.max_excerpt_characters < 4:
            raise InvalidCitationConfigurationError(
                "max_excerpt_characters must be at least 4."
            )


@dataclass(frozen=True, slots=True)
class Citation:
    """Structured source reference for one evidence chunk used by generation.

    A citation carries display location, stable chunk identity, retrieval
    provenance, and a bounded excerpt so callers never need to parse model text.
    """

    number: int
    source: str
    page_number: int | None
    chunk_index: int | None
    start_index: int | None
    end_index: int | None
    chunk_id: str | None
    retrieval_rank: int
    retrieval_score: float
    excerpt: str

    @property
    def label(self) -> str:
        return f"[{self.number}]"


def build_citation(
    retrieval_result: RetrievalResult,
    *,
    number: int,
    evidence_text: str | None = None,
    config: CitationConfig | None = None,
) -> Citation:
    """Build one citation from validated retrieval provenance.

    ``evidence_text`` may provide the exact prefix that survived generation
    budgeting; when shorter than the chunk, the character end position is
    reduced accordingly. Missing or malformed provenance raises
    ``CitationInputError`` before an unsupported source can be presented.
    """
    if not isinstance(retrieval_result, RetrievalResult):
        raise TypeError("retrieval_result must be a RetrievalResult.")
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        raise CitationInputError("citation number must be a positive integer.")
    if config is not None and not isinstance(config, CitationConfig):
        raise TypeError("config must be a CitationConfig.")

    settings = config or CitationConfig()
    metadata = retrieval_result.document.metadata
    source = metadata.get("source")
    if not isinstance(source, str) or not source.strip():
        raise CitationInputError(
            f"retrieval result {retrieval_result.rank} lacks non-empty source metadata."
        )

    retrieval_rank = retrieval_result.rank
    if (
        isinstance(retrieval_rank, bool)
        or not isinstance(retrieval_rank, int)
        or retrieval_rank <= 0
    ):
        raise CitationInputError("retrieval rank must be a positive integer.")

    score = retrieval_result.score
    if isinstance(score, bool) or not isinstance(score, Real):
        raise CitationInputError("retrieval score must be a number.")
    numeric_score = float(score)
    if not isfinite(numeric_score):
        raise CitationInputError("retrieval score must be finite.")

    page_index = _optional_nonnegative_integer(metadata, "page")
    chunk_index = _optional_nonnegative_integer(metadata, "chunk_index")
    start_index = _optional_nonnegative_integer(metadata, "start_index")
    end_index = _optional_nonnegative_integer(metadata, "end_index")
    if (start_index is None) != (end_index is None):
        raise CitationInputError(
            "start_index and end_index metadata must be provided together."
        )
    if (
        start_index is not None
        and end_index is not None
        and end_index < start_index
    ):
        raise CitationInputError("end_index metadata cannot precede start_index.")

    chunk_id = metadata.get("chunk_id")
    if chunk_id is not None and (
        not isinstance(chunk_id, str) or not chunk_id.strip()
    ):
        raise CitationInputError("chunk_id metadata must be a non-empty string.")

    document_content = retrieval_result.document.page_content.strip()
    if not document_content:
        raise CitationInputError("citation evidence text cannot be empty.")
    if evidence_text is not None and not isinstance(evidence_text, str):
        raise TypeError("evidence_text must be a string.")

    evidence_content = (
        document_content if evidence_text is None else evidence_text.strip()
    )
    if not evidence_content:
        raise CitationInputError("citation evidence text cannot be empty.")
    if not document_content.startswith(evidence_content):
        raise CitationInputError(
            "citation evidence text must be a prefix of the retrieved chunk."
        )
    if (
        evidence_text is not None
        and start_index is not None
        and end_index is not None
        and len(evidence_content) < len(document_content)
    ):
        end_index = min(end_index, start_index + len(evidence_content))

    return Citation(
        number=number,
        source=source.strip(),
        page_number=None if page_index is None else page_index + 1,
        chunk_index=chunk_index,
        start_index=start_index,
        end_index=end_index,
        chunk_id=None if chunk_id is None else chunk_id.strip(),
        retrieval_rank=retrieval_rank,
        retrieval_score=numeric_score,
        excerpt=_bounded_excerpt(
            evidence_content,
            max_characters=settings.max_excerpt_characters,
        ),
    )


def build_citations(
    retrieval_results: Iterable[RetrievalResult],
    *,
    config: CitationConfig | None = None,
) -> tuple[Citation, ...]:
    """Create sequential citations in the supplied retrieval order.

    The iterable is consumed once and numbering begins at one. Validation is
    delegated to ``build_citation`` for each result.
    """
    if config is not None and not isinstance(config, CitationConfig):
        raise TypeError("config must be a CitationConfig.")

    return tuple(
        build_citation(result, number=index, config=config)
        for index, result in enumerate(retrieval_results, start=1)
    )


def format_citation(citation: Citation) -> str:
    """Render a structured citation for terminal output.

    Only available page, chunk, and character locations are included. The
    function returns text and performs no printing or other I/O itself.
    """
    if not isinstance(citation, Citation):
        raise TypeError("citation must be a Citation.")

    location = []
    if citation.page_number is not None:
        location.append(f"page {citation.page_number}")
    if citation.chunk_index is not None:
        location.append(f"chunk {citation.chunk_index + 1}")
    if citation.start_index is not None and citation.end_index is not None:
        location.append(
            f"characters {citation.start_index}-{citation.end_index}"
        )

    suffix = f" ({', '.join(location)})" if location else ""
    return f"{citation.label} {citation.source}{suffix}\n    {citation.excerpt}"


def _optional_nonnegative_integer(
    metadata: Mapping[str, Any],
    name: str,
) -> int | None:
    value = metadata.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CitationInputError(
            f"{name} metadata must be a non-negative integer."
        )
    return value


def _bounded_excerpt(content: str, *, max_characters: int) -> str:
    excerpt = " ".join(content.split())
    if len(excerpt) <= max_characters:
        return excerpt
    return f"{excerpt[: max_characters - 3].rstrip()}..."
