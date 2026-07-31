"""Evaluate generated answers against explicit reference and abstention labels.

The module loads a versioned JSON dataset, calculates deterministic lexical
answer-quality and abstention metrics, checks citation-state behavior, and
formats reports without owning retrieval or model-provider construction.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from rag_pipeline.exceptions import (
    AnswerEvaluationInputError,
    InvalidAnswerEvaluationDatasetError,
)
from rag_pipeline.generation import GeneratedAnswer
from rag_pipeline.generation.prompting import INSUFFICIENT_CONTEXT_ANSWER

ANSWER_EVALUATION_SCHEMA_VERSION = 1

AnswerFunction = Callable[[str], GeneratedAnswer]

_DATASET_FIELDS = frozenset({"schema_version", "name", "cases"})
_CASE_FIELDS = frozenset({"id", "query", "should_abstain", "reference_answers"})
_WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class AnswerEvaluationCase:
    """One question with explicit answerability and accepted reference answers.

    Answerable cases require at least one reference; abstention cases require
    none. Multiple references let deterministic scoring accept known equivalent
    phrasings without delegating judgment to another language model.
    """

    case_id: str
    query: str
    should_abstain: bool
    reference_answers: tuple[str, ...]

    def __post_init__(self) -> None:
        """Normalize labels and reject contradictory or duplicate expectations."""
        normalized_case_id = _validate_non_empty_string("case id", self.case_id)
        normalized_query = _validate_non_empty_string("query", self.query)
        if not isinstance(self.should_abstain, bool):
            raise InvalidAnswerEvaluationDatasetError(
                "should_abstain must be a JSON boolean."
            )
        if isinstance(self.reference_answers, (str, bytes)):
            raise InvalidAnswerEvaluationDatasetError(
                "reference_answers must be a list of strings."
            )
        try:
            raw_references = tuple(self.reference_answers)
        except TypeError as exc:
            raise InvalidAnswerEvaluationDatasetError(
                "reference_answers must be a list of strings."
            ) from exc

        normalized_references = []
        seen_references: set[str] = set()
        for reference in raw_references:
            normalized_reference = _validate_non_empty_string(
                "reference answer",
                reference,
            )
            canonical_reference = _normalize_answer(normalized_reference)
            if not canonical_reference:
                raise InvalidAnswerEvaluationDatasetError(
                    f"case {normalized_case_id!r} contains a reference answer "
                    "without letters or numbers."
                )
            if canonical_reference in seen_references:
                raise InvalidAnswerEvaluationDatasetError(
                    f"case {normalized_case_id!r} contains a duplicate "
                    "reference answer after normalization."
                )
            if normalized_reference == INSUFFICIENT_CONTEXT_ANSWER:
                raise InvalidAnswerEvaluationDatasetError(
                    f"case {normalized_case_id!r} cannot use the configured "
                    "abstention response as a reference answer."
                )
            seen_references.add(canonical_reference)
            normalized_references.append(normalized_reference)

        if self.should_abstain and normalized_references:
            raise InvalidAnswerEvaluationDatasetError(
                f"case {normalized_case_id!r} expects abstention and cannot "
                "contain reference answers."
            )
        if not self.should_abstain and not normalized_references:
            raise InvalidAnswerEvaluationDatasetError(
                f"answerable case {normalized_case_id!r} requires at least one "
                "reference answer."
            )

        object.__setattr__(self, "case_id", normalized_case_id)
        object.__setattr__(self, "query", normalized_query)
        object.__setattr__(
            self,
            "reference_answers",
            tuple(normalized_references),
        )


@dataclass(frozen=True, slots=True)
class AnswerEvaluationDataset:
    """Versioned collection of answerable and abstention evaluation cases.

    A stable dataset name and unique case IDs make saved reports traceable.
    Schema validation happens before any retrieval or generation work begins.
    """

    name: str
    cases: tuple[AnswerEvaluationCase, ...]
    schema_version: int = ANSWER_EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate schema compatibility, dataset identity, and unique case IDs."""
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != ANSWER_EVALUATION_SCHEMA_VERSION
        ):
            raise InvalidAnswerEvaluationDatasetError(
                "unsupported answer evaluation schema_version; expected "
                f"{ANSWER_EVALUATION_SCHEMA_VERSION}."
            )
        normalized_name = _validate_non_empty_string("dataset name", self.name)
        try:
            cases = tuple(self.cases)
        except TypeError as exc:
            raise InvalidAnswerEvaluationDatasetError("cases must be a list.") from exc
        if not cases:
            raise InvalidAnswerEvaluationDatasetError(
                "answer evaluation datasets must contain at least one case."
            )

        seen_case_ids: set[str] = set()
        for case in cases:
            if not isinstance(case, AnswerEvaluationCase):
                raise InvalidAnswerEvaluationDatasetError(
                    "cases must contain AnswerEvaluationCase objects."
                )
            if case.case_id in seen_case_ids:
                raise InvalidAnswerEvaluationDatasetError(
                    f"duplicate answer evaluation case id {case.case_id!r}."
                )
            seen_case_ids.add(case.case_id)

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "cases", cases)


@dataclass(frozen=True, slots=True)
class AnswerCaseMetrics:
    """Reference, abstention, and citation diagnostics for one generated answer."""

    case_id: str
    query: str
    reference_answers: tuple[str, ...]
    generated_answer: str
    expected_abstention: bool
    predicted_abstention: bool
    abstention_correct: bool
    exact_match: float | None
    token_f1: float | None
    citation_count: int
    citation_behavior_correct: bool
    used_context_count: int
    context_was_truncated: bool
    generated: bool
    model_identifier: str
    prompt_identifier: str


@dataclass(frozen=True, slots=True)
class AnswerAggregateMetrics:
    """Macro answer scores and abstention-classification diagnostics.

    Optional values are undefined when a dataset lacks the required class or
    the model never predicts abstention. JSON reports represent them as null.
    """

    answerable_count: int
    expected_abstention_count: int
    predicted_abstention_count: int
    exact_match_rate: float | None
    mean_token_f1: float | None
    abstention_accuracy: float
    abstention_precision: float | None
    abstention_recall: float | None
    answerable_response_rate: float | None
    citation_behavior_rate: float


@dataclass(frozen=True, slots=True)
class AnswerEvaluationReport:
    """Complete per-case and aggregate answer evaluation for one dataset."""

    dataset_name: str
    schema_version: int
    cases: tuple[AnswerCaseMetrics, ...]
    aggregate: AnswerAggregateMetrics


def load_answer_evaluation_dataset(path: str | Path) -> AnswerEvaluationDataset:
    """Load and strictly validate a UTF-8 JSON answer evaluation dataset.

    The function performs filesystem I/O. Read, decoding, JSON syntax, unknown
    fields, and schema failures are normalized to
    ``InvalidAnswerEvaluationDatasetError`` with the resolved path attached.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path.")
    resolved_path = Path(path).expanduser().resolve()

    try:
        raw_data = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidAnswerEvaluationDatasetError(
            f"failed to load answer evaluation dataset {resolved_path}: {exc}"
        ) from exc

    try:
        return _parse_dataset(raw_data)
    except InvalidAnswerEvaluationDatasetError as exc:
        raise InvalidAnswerEvaluationDatasetError(
            f"invalid answer evaluation dataset {resolved_path}: {exc}"
        ) from exc


def evaluate_answers(
    dataset: AnswerEvaluationDataset,
    answer: AnswerFunction,
) -> AnswerEvaluationReport:
    """Generate and score every labeled answer case in dataset order.

    ``answer`` may perform retrieval, reranking, and model inference. Provider
    failures propagate unchanged; malformed return values fail the run rather
    than producing plausible metrics from an invalid prediction contract.
    """
    if not isinstance(dataset, AnswerEvaluationDataset):
        raise TypeError("dataset must be an AnswerEvaluationDataset.")
    if not callable(answer):
        raise TypeError("answer must be callable.")

    case_metrics = tuple(
        _evaluate_case(case, answer(case.query)) for case in dataset.cases
    )
    answerable_cases = tuple(
        case for case in case_metrics if not case.expected_abstention
    )
    expected_abstentions = tuple(
        case for case in case_metrics if case.expected_abstention
    )
    predicted_abstentions = tuple(
        case for case in case_metrics if case.predicted_abstention
    )

    true_abstentions = sum(
        case.expected_abstention and case.predicted_abstention for case in case_metrics
    )
    answered_answerable_cases = sum(
        not case.predicted_abstention for case in answerable_cases
    )
    aggregate = AnswerAggregateMetrics(
        answerable_count=len(answerable_cases),
        expected_abstention_count=len(expected_abstentions),
        predicted_abstention_count=len(predicted_abstentions),
        exact_match_rate=_mean_optional(
            tuple(case.exact_match for case in answerable_cases)
        ),
        mean_token_f1=_mean_optional(tuple(case.token_f1 for case in answerable_cases)),
        abstention_accuracy=(
            sum(case.abstention_correct for case in case_metrics) / len(case_metrics)
        ),
        abstention_precision=_safe_ratio(
            true_abstentions,
            len(predicted_abstentions),
        ),
        abstention_recall=_safe_ratio(
            true_abstentions,
            len(expected_abstentions),
        ),
        answerable_response_rate=_safe_ratio(
            answered_answerable_cases,
            len(answerable_cases),
        ),
        citation_behavior_rate=(
            sum(case.citation_behavior_correct for case in case_metrics)
            / len(case_metrics)
        ),
    )
    return AnswerEvaluationReport(
        dataset_name=dataset.name,
        schema_version=dataset.schema_version,
        cases=case_metrics,
        aggregate=aggregate,
    )


def answer_evaluation_to_dict(report: AnswerEvaluationReport) -> dict[str, object]:
    """Serialize an answer report to stable JSON-compatible primitives."""
    if not isinstance(report, AnswerEvaluationReport):
        raise TypeError("report must be an AnswerEvaluationReport.")
    aggregate = report.aggregate
    return {
        "dataset_name": report.dataset_name,
        "schema_version": report.schema_version,
        "case_count": len(report.cases),
        "metrics": {
            "answerable_count": aggregate.answerable_count,
            "expected_abstention_count": aggregate.expected_abstention_count,
            "predicted_abstention_count": aggregate.predicted_abstention_count,
            "exact_match_rate": aggregate.exact_match_rate,
            "mean_token_f1": aggregate.mean_token_f1,
            "abstention_accuracy": aggregate.abstention_accuracy,
            "abstention_precision": aggregate.abstention_precision,
            "abstention_recall": aggregate.abstention_recall,
            "answerable_response_rate": aggregate.answerable_response_rate,
            "citation_behavior_rate": aggregate.citation_behavior_rate,
        },
        "cases": [
            {
                "id": case.case_id,
                "query": case.query,
                "reference_answers": list(case.reference_answers),
                "generated_answer": case.generated_answer,
                "expected_abstention": case.expected_abstention,
                "predicted_abstention": case.predicted_abstention,
                "abstention_correct": case.abstention_correct,
                "exact_match": case.exact_match,
                "token_f1": case.token_f1,
                "citation_count": case.citation_count,
                "citation_behavior_correct": case.citation_behavior_correct,
                "used_context_count": case.used_context_count,
                "context_was_truncated": case.context_was_truncated,
                "generated": case.generated,
                "model_identifier": case.model_identifier,
                "prompt_identifier": case.prompt_identifier,
            }
            for case in report.cases
        ],
    }


def format_answer_evaluation_table(report: AnswerEvaluationReport) -> str:
    """Render per-case answer metrics and aggregate diagnostics as a table."""
    if not isinstance(report, AnswerEvaluationReport):
        raise TypeError("report must be an AnswerEvaluationReport.")

    headers = (
        "Case",
        "Expected",
        "Predicted",
        "EM",
        "F1",
        "Citations",
        "Citation state",
    )
    rows = [
        (
            case.case_id,
            _answer_state(case.expected_abstention),
            _answer_state(case.predicted_abstention),
            _format_optional_metric(case.exact_match),
            _format_optional_metric(case.token_f1),
            str(case.citation_count),
            "OK" if case.citation_behavior_correct else "FAIL",
        )
        for case in report.cases
    ]
    rows.append(
        (
            "MACRO",
            "-",
            "-",
            _format_optional_metric(report.aggregate.exact_match_rate),
            _format_optional_metric(report.aggregate.mean_token_f1),
            "-",
            f"{report.aggregate.citation_behavior_rate:.3f}",
        )
    )
    widths = tuple(
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    )

    def format_row(values: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(width) if index == 0 else value.rjust(width)
            for index, (value, width) in enumerate(zip(values, widths, strict=True))
        )

    aggregate = report.aggregate
    abstention_summary = (
        "Abstention: "
        f"accuracy={aggregate.abstention_accuracy:.3f}, "
        f"precision={_format_optional_metric(aggregate.abstention_precision)}, "
        f"recall={_format_optional_metric(aggregate.abstention_recall)}, "
        "answerable_response_rate="
        f"{_format_optional_metric(aggregate.answerable_response_rate)}"
    )
    separator = "  ".join("-" * width for width in widths)
    return "\n".join(
        (
            f"Answer evaluation: {report.dataset_name}",
            (
                f"Cases: {len(report.cases)} | "
                f"Answerable: {aggregate.answerable_count} | "
                "Expected abstentions: "
                f"{aggregate.expected_abstention_count}"
            ),
            abstention_summary,
            "",
            format_row(headers),
            separator,
            *(format_row(row) for row in rows),
        )
    )


def _parse_dataset(raw_data: object) -> AnswerEvaluationDataset:
    """Convert untrusted decoded JSON into validated answer-evaluation objects."""
    dataset_object = _require_object(raw_data, context="dataset")
    _validate_object_fields(
        dataset_object,
        expected=_DATASET_FIELDS,
        context="dataset",
    )
    raw_cases = dataset_object["cases"]
    if not isinstance(raw_cases, list):
        raise InvalidAnswerEvaluationDatasetError("cases must be a list.")

    cases: list[AnswerEvaluationCase] = []
    for index, raw_case in enumerate(raw_cases):
        case_object = _require_object(raw_case, context=f"cases[{index}]")
        _validate_object_fields(
            case_object,
            expected=_CASE_FIELDS,
            context=f"cases[{index}]",
        )
        raw_references = case_object["reference_answers"]
        if not isinstance(raw_references, list):
            raise InvalidAnswerEvaluationDatasetError(
                f"cases[{index}].reference_answers must be a list."
            )
        cases.append(
            AnswerEvaluationCase(
                case_id=cast(str, case_object["id"]),
                query=cast(str, case_object["query"]),
                should_abstain=cast(bool, case_object["should_abstain"]),
                reference_answers=cast(tuple[str, ...], tuple(raw_references)),
            )
        )

    return AnswerEvaluationDataset(
        name=cast(str, dataset_object["name"]),
        cases=tuple(cases),
        schema_version=cast(int, dataset_object["schema_version"]),
    )


def _evaluate_case(
    case: AnswerEvaluationCase,
    prediction: GeneratedAnswer,
) -> AnswerCaseMetrics:
    """Validate one generation result and calculate case-level diagnostics."""
    if not isinstance(prediction, GeneratedAnswer):
        raise AnswerEvaluationInputError(
            f"answer callback for case {case.case_id!r} must return a GeneratedAnswer."
        )
    generated_answer = _validate_prediction_string(
        prediction.answer,
        case_id=case.case_id,
        field="answer",
    )
    model_identifier = _validate_prediction_string(
        prediction.model_identifier,
        case_id=case.case_id,
        field="model_identifier",
    )
    prompt_identifier = _validate_prediction_string(
        prediction.prompt_identifier,
        case_id=case.case_id,
        field="prompt_identifier",
    )
    if not isinstance(prediction.citations, tuple):
        raise AnswerEvaluationInputError(
            f"answer callback for case {case.case_id!r} returned invalid citations."
        )
    if not isinstance(prediction.used_context, tuple):
        raise AnswerEvaluationInputError(
            f"answer callback for case {case.case_id!r} returned invalid context."
        )

    predicted_abstention = generated_answer == INSUFFICIENT_CONTEXT_ANSWER
    if case.should_abstain:
        exact_match = None
        token_f1 = None
    elif predicted_abstention:
        exact_match = 0.0
        token_f1 = 0.0
    else:
        exact_match = max(
            _exact_match(generated_answer, reference)
            for reference in case.reference_answers
        )
        token_f1 = max(
            _token_f1(generated_answer, reference)
            for reference in case.reference_answers
        )

    citation_count = len(prediction.citations)
    citation_behavior_correct = (
        citation_count == 0 if predicted_abstention else citation_count > 0
    )
    return AnswerCaseMetrics(
        case_id=case.case_id,
        query=case.query,
        reference_answers=case.reference_answers,
        generated_answer=generated_answer,
        expected_abstention=case.should_abstain,
        predicted_abstention=predicted_abstention,
        abstention_correct=case.should_abstain == predicted_abstention,
        exact_match=exact_match,
        token_f1=token_f1,
        citation_count=citation_count,
        citation_behavior_correct=citation_behavior_correct,
        used_context_count=len(prediction.used_context),
        context_was_truncated=prediction.context_was_truncated,
        generated=prediction.generated,
        model_identifier=model_identifier,
        prompt_identifier=prompt_identifier,
    )


def _exact_match(candidate: str, reference: str) -> float:
    """Return case-insensitive, punctuation-insensitive normalized exact match."""
    return float(_normalize_answer(candidate) == _normalize_answer(reference))


def _token_f1(candidate: str, reference: str) -> float:
    """Calculate multiset token F1 after language-agnostic normalization."""
    candidate_tokens = _answer_tokens(candidate)
    reference_tokens = _answer_tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return float(candidate_tokens == reference_tokens)

    common_count = sum((Counter(candidate_tokens) & Counter(reference_tokens)).values())
    if common_count == 0:
        return 0.0
    precision = common_count / len(candidate_tokens)
    recall = common_count / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize_answer(value: str) -> str:
    """Normalize Unicode, case, punctuation, and whitespace for lexical scoring."""
    return " ".join(_answer_tokens(value))


def _answer_tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(_WORD_PATTERN.findall(normalized))


def _require_object(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise InvalidAnswerEvaluationDatasetError(f"{context} must be a JSON object.")
    return value


def _validate_object_fields(
    value: dict[str, object],
    *,
    expected: frozenset[str],
    context: str,
) -> None:
    """Reject missing and unknown schema fields so label typos fail early."""
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise InvalidAnswerEvaluationDatasetError(
            f"{context} is missing required field(s): {', '.join(missing)}."
        )
    if unknown:
        raise InvalidAnswerEvaluationDatasetError(
            f"{context} contains unknown field(s): {', '.join(unknown)}."
        )


def _validate_non_empty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidAnswerEvaluationDatasetError(f"{name} must be a non-empty string.")
    return value.strip()


def _validate_prediction_string(
    value: object,
    *,
    case_id: str,
    field: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AnswerEvaluationInputError(
            f"answer callback for case {case_id!r} returned an invalid {field}."
        )
    return value.strip()


def _mean_optional(values: tuple[float | None, ...]) -> float | None:
    numeric_values = tuple(value for value in values if value is not None)
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _format_optional_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _answer_state(should_abstain: bool) -> str:
    return "abstain" if should_abstain else "answer"
