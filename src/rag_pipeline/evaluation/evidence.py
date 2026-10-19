"""Diagnose whether labeled source evidence survives retrieval and prompt packing.

This evaluator observes existing answer runs. It never reruns retrieval or a
model, and treats lexical answer scores as separate from evidence availability.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from langchain_core.documents import Document

from rag_pipeline.evaluation.answers import (
    AnswerCaseMetrics,
    AnswerEvaluationDataset,
    AnswerEvaluationReport,
)
from rag_pipeline.evaluation.retrieval import (
    RelevantDocument,
    RetrievalEvaluationDataset,
)
from rag_pipeline.exceptions import AnswerEvaluationInputError
from rag_pipeline.generation import GeneratedAnswer
from rag_pipeline.retrieval import RetrievalResult

EvidenceState = Literal[
    "not_retrieved",
    "retrieved_not_packed",
    "packed_but_not_answered",
    "packed_and_answered",
]


@dataclass(frozen=True, slots=True)
class EvidenceObservation:
    """One answer run with the exact final retrieval list supplied to generation."""

    retrieval_results: tuple[RetrievalResult, ...]
    answer: GeneratedAnswer


@dataclass(frozen=True, slots=True)
class EvidenceSelectorMetrics:
    """Stage reached by one source-anchor judgment, counting duplicates once."""

    selector: RelevantDocument
    state: EvidenceState
    retrieved_ranks: tuple[int, ...]
    packed_evidence_numbers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EvidenceCaseMetrics:
    """Per-question evidence states alongside existing lexical answer scores."""

    case_id: str
    query: str
    selectors: tuple[EvidenceSelectorMetrics, ...]
    context_was_truncated: bool
    answer_exact_match: float | None
    answer_token_f1: float | None


@dataclass(frozen=True, slots=True)
class EvidenceCoverageReport:
    """Selector-level prompt coverage for paired answerable benchmark cases."""

    cases: tuple[EvidenceCaseMetrics, ...]
    selector_count: int
    not_retrieved_count: int
    retrieved_not_packed_count: int
    packed_but_not_answered_count: int
    packed_and_answered_count: int
    retrieval_coverage_rate: float
    packed_coverage_rate: float
    fully_packed_case_count: int


def supports_evidence_coverage(
    retrieval_dataset: RetrievalEvaluationDataset,
    answer_dataset: AnswerEvaluationDataset,
) -> bool:
    """Require exact source anchors and matching IDs/queries for answerable cases.

    Legacy chunk-index labels and independently authored answer sets remain
    usable by the benchmark, but cannot support this diagnostic reliably.
    """
    if retrieval_dataset.schema_version < 2:
        return False
    retrieval_cases = {case.case_id: case for case in retrieval_dataset.cases}
    answerable = [case for case in answer_dataset.cases if not case.should_abstain]
    return bool(answerable) and all(
        (matched := retrieval_cases.get(case.case_id)) is not None
        and matched.query == case.query
        and matched.relevant_documents
        and all(
            selector.content_contains is not None
            for selector in matched.relevant_documents
        )
        for case in answerable
    )


def evaluate_evidence_coverage(
    retrieval_dataset: RetrievalEvaluationDataset,
    answer_dataset: AnswerEvaluationDataset,
    answer_report: AnswerEvaluationReport,
    observations: Sequence[EvidenceObservation],
) -> EvidenceCoverageReport:
    """Classify each exact anchor using the retrieval list and packed prefixes.

    An answered state means the model did not abstain, not that its claims are
    correct. Exact match and token F1 remain separate diagnostics. Misaligned
    labels or traces fail instead of producing misleading coverage.
    """
    if not supports_evidence_coverage(retrieval_dataset, answer_dataset):
        raise AnswerEvaluationInputError(
            "evidence coverage requires paired answerable cases with source anchors."
        )
    if len(observations) != len(answer_dataset.cases) or len(
        answer_report.cases
    ) != len(observations):
        raise AnswerEvaluationInputError(
            "evidence observations must align with answer cases."
        )

    retrieval_cases = {case.case_id: case for case in retrieval_dataset.cases}
    cases: list[EvidenceCaseMetrics] = []
    for answer_case, answer_metrics, observation in zip(
        answer_dataset.cases, answer_report.cases, observations, strict=True
    ):
        if answer_metrics.case_id != answer_case.case_id or not isinstance(
            observation, EvidenceObservation
        ):
            raise AnswerEvaluationInputError(
                "evidence observations must align with answer cases."
            )
        if answer_case.should_abstain:
            continue
        retrieved, packed, texts = _validated_observation(answer_metrics, observation)
        selectors = tuple(
            _classify_selector(
                selector,
                retrieved=retrieved,
                packed=packed,
                texts=texts,
                abstained=answer_metrics.predicted_abstention,
            )
            for selector in retrieval_cases[answer_case.case_id].relevant_documents
        )
        cases.append(
            EvidenceCaseMetrics(
                case_id=answer_case.case_id,
                query=answer_case.query,
                selectors=selectors,
                context_was_truncated=observation.answer.context_was_truncated,
                answer_exact_match=answer_metrics.exact_match,
                answer_token_f1=answer_metrics.token_f1,
            )
        )

    states = [selector.state for case in cases for selector in case.selectors]
    selector_count = len(states)
    not_retrieved = states.count("not_retrieved")
    retrieved_not_packed = states.count("retrieved_not_packed")
    packed_not_answered = states.count("packed_but_not_answered")
    packed_answered = states.count("packed_and_answered")
    return EvidenceCoverageReport(
        cases=tuple(cases),
        selector_count=selector_count,
        not_retrieved_count=not_retrieved,
        retrieved_not_packed_count=retrieved_not_packed,
        packed_but_not_answered_count=packed_not_answered,
        packed_and_answered_count=packed_answered,
        retrieval_coverage_rate=(selector_count - not_retrieved) / selector_count,
        packed_coverage_rate=(packed_not_answered + packed_answered) / selector_count,
        fully_packed_case_count=sum(
            all(selector.packed_evidence_numbers for selector in case.selectors)
            for case in cases
        ),
    )


def _validated_observation(
    answer_metrics: AnswerCaseMetrics,
    observation: EvidenceObservation,
) -> tuple[tuple[RetrievalResult, ...], tuple[RetrievalResult, ...], tuple[str, ...]]:
    """Reject traces that cannot represent the scored prompt and citations."""
    if not isinstance(observation.answer, GeneratedAnswer):
        raise AnswerEvaluationInputError(
            "evidence observation must contain a GeneratedAnswer."
        )
    if answer_metrics.generated_answer != observation.answer.answer:
        raise AnswerEvaluationInputError(
            "evidence observation does not match the scored answer."
        )
    retrieved = observation.retrieval_results
    packed = observation.answer.used_context
    texts = observation.answer.packed_evidence_texts
    if len(packed) != len(texts) or any(
        not isinstance(result, RetrievalResult)
        or not isinstance(text, str)
        or not text
        or not result.document.page_content.strip().startswith(text)
        for result, text in zip(packed, texts, strict=True)
    ):
        raise AnswerEvaluationInputError(
            "packed evidence prefixes must align with used context."
        )
    if any(not isinstance(result, RetrievalResult) for result in retrieved):
        raise AnswerEvaluationInputError("retrieval trace contains an invalid result.")
    if any(not any(used is result for result in retrieved) for used in packed):
        raise AnswerEvaluationInputError(
            "packed evidence must come from the retrieval trace."
        )
    if not answer_metrics.predicted_abstention and (
        len(observation.answer.citations) != len(packed)
        or any(
            citation.number != number or citation.retrieval_rank != result.rank
            for number, (citation, result) in enumerate(
                zip(observation.answer.citations, packed, strict=True), start=1
            )
        )
    ):
        raise AnswerEvaluationInputError("citations must align with packed evidence.")
    return retrieved, packed, texts


def _classify_selector(
    selector: RelevantDocument,
    *,
    retrieved: tuple[RetrievalResult, ...],
    packed: tuple[RetrievalResult, ...],
    texts: tuple[str, ...],
    abstained: bool,
) -> EvidenceSelectorMetrics:
    """Count one anchor once, even when several ranked chunks contain it."""
    retrieved_ranks = tuple(
        result.rank for result in retrieved if selector.matches(result.document)
    )
    packed_numbers = tuple(
        number
        for number, (result, text) in enumerate(
            zip(packed, texts, strict=True), start=1
        )
        if selector.matches(
            Document(page_content=text, metadata=result.document.metadata)
        )
    )
    state: EvidenceState
    if not retrieved_ranks:
        state = "not_retrieved"
    elif not packed_numbers:
        state = "retrieved_not_packed"
    elif abstained:
        state = "packed_but_not_answered"
    else:
        state = "packed_and_answered"
    return EvidenceSelectorMetrics(selector, state, retrieved_ranks, packed_numbers)


def evidence_coverage_to_dict(report: EvidenceCoverageReport) -> dict[str, object]:
    """Serialize a diagnostic without changing legacy retrieval/answer metrics."""
    if not isinstance(report, EvidenceCoverageReport):
        raise TypeError("report must be an EvidenceCoverageReport.")
    return {
        "answerable_case_count": len(report.cases),
        "selector_count": report.selector_count,
        "metrics": {
            "not_retrieved_count": report.not_retrieved_count,
            "retrieved_not_packed_count": report.retrieved_not_packed_count,
            "packed_but_not_answered_count": report.packed_but_not_answered_count,
            "packed_and_answered_count": report.packed_and_answered_count,
            "retrieval_coverage_rate": report.retrieval_coverage_rate,
            "packed_coverage_rate": report.packed_coverage_rate,
            "fully_packed_case_count": report.fully_packed_case_count,
        },
        "cases": [
            {
                "id": case.case_id,
                "query": case.query,
                "context_was_truncated": case.context_was_truncated,
                "answer_exact_match": case.answer_exact_match,
                "answer_token_f1": case.answer_token_f1,
                "selectors": [
                    {
                        "metadata": dict(selector.selector.metadata),
                        "content_contains": selector.selector.content_contains,
                        "state": selector.state,
                        "retrieved_ranks": list(selector.retrieved_ranks),
                        "packed_evidence_numbers": list(
                            selector.packed_evidence_numbers
                        ),
                    }
                    for selector in case.selectors
                ],
            }
            for case in report.cases
        ],
    }
