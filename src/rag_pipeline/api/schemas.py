"""Define bounded HTTP request and provenance-preserving response contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rag_pipeline.application.answering import (
    DEFAULT_ANSWER_SCORE_THRESHOLD,
    AnswerOutcome,
)
from rag_pipeline.generation.citations import Citation
from rag_pipeline.retrieval import RetrievalResult


class QueryRequest(BaseModel):
    """Bound one retrieval request before local model or storage I/O begins."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=20, strict=True)
    score_threshold: float | None = Field(
        default=None, ge=-1, le=1, allow_inf_nan=False, strict=True
    )


class AnswerRequest(QueryRequest):
    """Apply the CLI answer score gate unless the caller explicitly disables it."""

    score_threshold: float | None = Field(
        default=DEFAULT_ANSWER_SCORE_THRESHOLD,
        ge=-1,
        le=1,
        allow_inf_nan=False,
        strict=True,
    )


class HealthResponse(BaseModel):
    """Process liveness only; this does not assert model or collection readiness."""

    status: Literal["ok"] = "ok"


class RetrievalHit(BaseModel):
    """Expose ranked source text and selected provenance without raw metadata."""

    rank: int
    score: float
    score_kind: str
    text: str
    source: str | None
    page_number: int | None
    chunk_index: int | None
    chunk_id: str | None

    @classmethod
    def from_result(cls, result: RetrievalResult) -> RetrievalHit:
        """Convert a validated retrieval hit to stable, JSON-safe fields."""
        metadata = result.document.metadata
        page = metadata.get("page")
        chunk_index = metadata.get("chunk_index")
        source = metadata.get("source")
        chunk_id = metadata.get("chunk_id")
        return cls(
            rank=result.rank,
            score=result.score,
            score_kind=result.score_kind,
            text=result.document.page_content,
            source=source if isinstance(source, str) else None,
            page_number=(
                page + 1
                if isinstance(page, int) and not isinstance(page, bool)
                else None
            ),
            chunk_index=(
                chunk_index
                if isinstance(chunk_index, int) and not isinstance(chunk_index, bool)
                else None
            ),
            chunk_id=chunk_id if isinstance(chunk_id, str) else None,
        )


class RetrievalResponse(BaseModel):
    """Return ordered evidence; an empty list means no chunk passed selection."""

    results: list[RetrievalHit]


class CitationResponse(BaseModel):
    """Preserve deterministic citation identity and bounded evidence excerpts."""

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

    @classmethod
    def from_citation(cls, citation: Citation) -> CitationResponse:
        """Serialize a citation built outside the language model."""
        return cls(
            number=citation.number,
            source=citation.source,
            page_number=citation.page_number,
            chunk_index=citation.chunk_index,
            start_index=citation.start_index,
            end_index=citation.end_index,
            chunk_id=citation.chunk_id,
            retrieval_rank=citation.retrieval_rank,
            retrieval_score=citation.retrieval_score,
            excerpt=citation.excerpt,
        )


class AnswerResponse(BaseModel):
    """Distinguish no-evidence fallback from model invocation and abstention."""

    answer: str
    generated: bool
    model_identifier: str | None
    prompt_identifier: str | None
    citations: list[CitationResponse]

    @classmethod
    def from_outcome(cls, outcome: AnswerOutcome) -> AnswerResponse:
        """Keep citation and model metadata aligned with the generation trace."""
        generated = outcome.generated_answer
        return cls(
            answer=outcome.answer,
            generated=False if generated is None else generated.generated,
            model_identifier=None if generated is None else generated.model_identifier,
            prompt_identifier=None
            if generated is None
            else generated.prompt_identifier,
            citations=(
                []
                if generated is None
                else [
                    CitationResponse.from_citation(item) for item in generated.citations
                ]
            ),
        )
