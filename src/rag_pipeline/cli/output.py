"""Format user-facing terminal output for interactive CLI commands.

Evaluation and benchmark modules own their report formatters. This module keeps
the remaining retrieval and answer rendering pure so handlers only orchestrate
services and write the resulting text to stdout.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_pipeline.generation import GeneratedAnswer
    from rag_pipeline.retrieval import RetrievalResult


def format_retrieval_results(results: Sequence[RetrievalResult]) -> str:
    """Render ranked evidence with provenance and reranking diagnostics.

    Empty results produce the same user-facing message used by retrieval
    commands when no chunk passes filtering or score thresholds.
    """
    if not results:
        return "No chunks met the retrieval criteria."

    lines: list[str] = []
    for result in results:
        metadata = result.document.metadata
        location = f"source={metadata.get('source', '<unknown>')}"
        page = metadata.get("page")
        if isinstance(page, int) and not isinstance(page, bool):
            location += f" page={page + 1}"
        chunk_index = metadata.get("chunk_index")
        if isinstance(chunk_index, int) and not isinstance(chunk_index, bool):
            location += f" chunk={chunk_index}"

        ranking_details = f"score_kind={result.score_kind}"
        if result.retrieval_rank is not None:
            ranking_details += (
                f" retrieval_rank={result.retrieval_rank}"
                f" retrieval_score={result.retrieval_score:.4f}"
                f" retrieval_score_kind={result.retrieval_score_kind}"
                f" reranker_model={result.reranker_model}"
            )
        lines.append(
            f"{result.rank}. score={result.score:.4f} {location} "
            f"{ranking_details}"
        )
        lines.append(f"   {_content_preview(result.document.page_content)}")
    return "\n".join(lines)


def format_generated_answer(answer: GeneratedAnswer) -> str:
    """Render an answer followed by traceable citations when any are present."""
    from rag_pipeline.citations import format_citation

    lines = ["Answer:", answer.answer]
    if answer.citations:
        lines.extend(("", "Sources:"))
        lines.extend(format_citation(citation) for citation in answer.citations)
    return "\n".join(lines)


def format_abstention_answer(answer: str) -> str:
    """Render a deterministic no-context answer without a sources section."""
    return f"Answer:\n{answer}"


def _content_preview(content: str, *, max_length: int = 240) -> str:
    preview = " ".join(content.split())
    if len(preview) <= max_length:
        return preview
    return f"{preview[: max_length - 3]}..."
