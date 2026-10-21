"""Compose retrieved evidence and local generation without transport concerns."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rag_pipeline.exceptions import GenerationInputError
from rag_pipeline.generation import (
    INSUFFICIENT_CONTEXT_ANSWER,
    AnswerGenerator,
    GeneratedAnswer,
    GenerationConfig,
)
from rag_pipeline.retrieval import RetrievalResult

DEFAULT_ANSWER_SCORE_THRESHOLD = 0.2


@dataclass(frozen=True, slots=True)
class AnswerOutcome:
    """Answer text plus the generation trace, absent for no retrieved evidence."""

    answer: str
    generated_answer: GeneratedAnswer | None


def answer_from_results(
    question: str,
    results: Sequence[RetrievalResult],
    *,
    generator_factory: Callable[[], AnswerGenerator],
    config: GenerationConfig,
) -> AnswerOutcome:
    """Generate from ranked results, skipping model startup when none survive.

    The caller owns retrieval and resource lifetimes. The factory may perform
    model I/O and inference only for non-empty evidence; provider and context
    validation remain in ``AnswerGenerator``.
    """
    if not isinstance(question, str):
        raise TypeError("question must be a string.")
    if not question.strip():
        raise GenerationInputError("question cannot be empty.")
    if not results:
        return AnswerOutcome(INSUFFICIENT_CONTEXT_ANSWER, None)
    generated = generator_factory().generate(question, results, config=config)
    return AnswerOutcome(generated.answer, generated)
