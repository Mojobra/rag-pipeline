"""Measure ranked retrieval quality against explicit relevance judgments.

The module loads a versioned JSON dataset, matches returned LangChain document
metadata to binary relevance selectors, calculates standard top-k metrics, and
formats deterministic reports without invoking answer generation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

from langchain_core.documents import Document

from rag_pipeline.exceptions import (
    InvalidRetrievalEvaluationDatasetError,
    RetrievalEvaluationInputError,
)
from rag_pipeline.retrieval import RetrievalResult


RETRIEVAL_EVALUATION_SCHEMA_VERSION = 1

MetadataScalar: TypeAlias = str | int | float | bool | None
RetrievalFunction: TypeAlias = Callable[[str], Sequence[RetrievalResult]]

_MISSING = object()
_DATASET_FIELDS = frozenset({"schema_version", "name", "cases"})
_CASE_FIELDS = frozenset({"id", "query", "relevant"})


@dataclass(frozen=True, slots=True)
class RelevantDocument:
    """Exact metadata selector representing relevant evidence for one query.

    Every key-value pair must occur on a returned LangChain document for the
    selector to match. Selectors can use current fields such as ``file_name``,
    ``page``, or ``chunk_id`` and later adopt stable business document IDs.
    """

    metadata: Mapping[str, MetadataScalar]

    def __post_init__(self) -> None:
        """Validate and freeze a non-empty map of JSON scalar metadata values."""
        if not isinstance(self.metadata, Mapping):
            raise InvalidRetrievalEvaluationDatasetError(
                "relevant selectors must be JSON objects."
            )
        if not self.metadata:
            raise InvalidRetrievalEvaluationDatasetError(
                "relevant selectors cannot be empty."
            )

        normalized_metadata: dict[str, MetadataScalar] = {}
        for key, value in self.metadata.items():
            if not isinstance(key, str) or not key.strip():
                raise InvalidRetrievalEvaluationDatasetError(
                    "relevant selector keys must be non-empty strings."
                )
            normalized_key = key.strip()
            if normalized_key in normalized_metadata:
                raise InvalidRetrievalEvaluationDatasetError(
                    f"duplicate relevant selector key {normalized_key!r}."
                )
            _validate_metadata_scalar(value, key=normalized_key)
            normalized_metadata[normalized_key] = value

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(normalized_metadata),
        )

    def matches(self, document: Document) -> bool:
        """Return whether a document contains every selector value exactly.

        Value types must also match, preventing Python's equality rules from
        treating metadata such as boolean ``true`` as integer ``1``.
        """
        if not isinstance(document, Document):
            raise TypeError("document must be a LangChain Document object.")

        for key, expected_value in self.metadata.items():
            actual_value = document.metadata.get(key, _MISSING)
            if actual_value is _MISSING or type(actual_value) is not type(
                expected_value
            ):
                return False
            if actual_value != expected_value:
                return False
        return True


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCase:
    """One query and its binary metadata-based relevance judgments.

    Case IDs identify regressions in reports, while the query is passed unchanged
    to the configured retriever. At least one unique relevance selector is
    required so recall has a defined denominator.
    """

    case_id: str
    query: str
    relevant_documents: tuple[RelevantDocument, ...]

    def __post_init__(self) -> None:
        """Normalize text fields and reject missing or duplicate judgments."""
        normalized_case_id = _validate_non_empty_string("case id", self.case_id)
        normalized_query = _validate_non_empty_string("query", self.query)
        try:
            relevant_documents = tuple(self.relevant_documents)
        except TypeError as exc:
            raise InvalidRetrievalEvaluationDatasetError(
                "relevant must be a list of metadata selectors."
            ) from exc
        if not relevant_documents:
            raise InvalidRetrievalEvaluationDatasetError(
                f"case {normalized_case_id!r} must contain relevant selectors."
            )

        canonical_selectors: set[str] = set()
        for relevant_document in relevant_documents:
            if not isinstance(relevant_document, RelevantDocument):
                raise InvalidRetrievalEvaluationDatasetError(
                    "relevant must contain RelevantDocument objects."
                )
            canonical_selector = json.dumps(
                dict(relevant_document.metadata),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            if canonical_selector in canonical_selectors:
                raise InvalidRetrievalEvaluationDatasetError(
                    f"case {normalized_case_id!r} contains a duplicate "
                    "relevant selector."
                )
            canonical_selectors.add(canonical_selector)

        object.__setattr__(self, "case_id", normalized_case_id)
        object.__setattr__(self, "query", normalized_query)
        object.__setattr__(self, "relevant_documents", relevant_documents)


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationDataset:
    """Versioned collection of labeled queries evaluated as one snapshot.

    Dataset names make persisted reports identifiable. Unique case IDs and a
    fixed schema version keep comparisons reproducible across retrieval changes.
    """

    name: str
    cases: tuple[RetrievalEvaluationCase, ...]
    schema_version: int = RETRIEVAL_EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate schema compatibility, dataset identity, and unique cases."""
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != RETRIEVAL_EVALUATION_SCHEMA_VERSION
        ):
            raise InvalidRetrievalEvaluationDatasetError(
                "unsupported retrieval evaluation schema_version; expected "
                f"{RETRIEVAL_EVALUATION_SCHEMA_VERSION}."
            )
        normalized_name = _validate_non_empty_string("dataset name", self.name)
        try:
            cases = tuple(self.cases)
        except TypeError as exc:
            raise InvalidRetrievalEvaluationDatasetError(
                "cases must be a list."
            ) from exc
        if not cases:
            raise InvalidRetrievalEvaluationDatasetError(
                "retrieval evaluation datasets must contain at least one case."
            )

        seen_case_ids: set[str] = set()
        for case in cases:
            if not isinstance(case, RetrievalEvaluationCase):
                raise InvalidRetrievalEvaluationDatasetError(
                    "cases must contain RetrievalEvaluationCase objects."
                )
            if case.case_id in seen_case_ids:
                raise InvalidRetrievalEvaluationDatasetError(
                    f"duplicate retrieval evaluation case id {case.case_id!r}."
                )
            seen_case_ids.add(case.case_id)

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "cases", cases)


@dataclass(frozen=True, slots=True)
class RetrievalCaseMetrics:
    """Top-k retrieval metrics and diagnostics for one labeled query.

    Precision counts empty result slots as misses by using ``top_k`` as its
    denominator. Recall counts distinct relevance selectors matched by at least
    one returned document.
    """

    case_id: str
    query: str
    retrieved_count: int
    relevant_count: int
    matched_relevant_count: int
    relevant_retrieved_count: int
    first_relevant_rank: int | None
    hit_at_k: float
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank_at_k: float


@dataclass(frozen=True, slots=True)
class RetrievalAggregateMetrics:
    """Macro-averaged quality metrics giving every query equal weight."""

    hit_rate_at_k: float
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank_at_k: float


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    """Complete per-case and aggregate result for one dataset and cutoff."""

    dataset_name: str
    schema_version: int
    top_k: int
    cases: tuple[RetrievalCaseMetrics, ...]
    aggregate: RetrievalAggregateMetrics


def load_retrieval_evaluation_dataset(
    path: str | Path,
) -> RetrievalEvaluationDataset:
    """Load and strictly validate a UTF-8 JSON retrieval dataset.

    The function performs filesystem I/O. Read, decoding, JSON syntax, unknown
    fields, and schema errors are normalized to
    ``InvalidRetrievalEvaluationDatasetError`` with the dataset path attached.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or pathlib.Path.")
    resolved_path = Path(path).expanduser().resolve()

    try:
        raw_data = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidRetrievalEvaluationDatasetError(
            f"failed to load retrieval evaluation dataset {resolved_path}: {exc}"
        ) from exc

    try:
        return _parse_dataset(raw_data)
    except InvalidRetrievalEvaluationDatasetError as exc:
        raise InvalidRetrievalEvaluationDatasetError(
            f"invalid retrieval evaluation dataset {resolved_path}: {exc}"
        ) from exc


def evaluate_retrieval(
    dataset: RetrievalEvaluationDataset,
    retrieve: RetrievalFunction,
    *,
    top_k: int,
) -> RetrievalEvaluationReport:
    """Run every labeled query and calculate deterministic binary top-k metrics.

    ``retrieve`` may perform model inference and vector-store I/O. Its ordered
    results are evaluated up to ``top_k`` without calling generation. Provider
    failures propagate; malformed result sequences raise an evaluation input
    error so invalid rankings cannot silently produce credible-looking scores.
    """
    if not isinstance(dataset, RetrievalEvaluationDataset):
        raise TypeError("dataset must be a RetrievalEvaluationDataset.")
    if not callable(retrieve):
        raise TypeError("retrieve must be callable.")
    _validate_top_k(top_k)

    case_metrics = tuple(
        _evaluate_case(case, retrieve(case.query), top_k=top_k)
        for case in dataset.cases
    )
    case_count = len(case_metrics)
    aggregate = RetrievalAggregateMetrics(
        hit_rate_at_k=sum(case.hit_at_k for case in case_metrics) / case_count,
        mean_precision_at_k=(
            sum(case.precision_at_k for case in case_metrics) / case_count
        ),
        mean_recall_at_k=(
            sum(case.recall_at_k for case in case_metrics) / case_count
        ),
        mean_reciprocal_rank_at_k=(
            sum(case.reciprocal_rank_at_k for case in case_metrics) / case_count
        ),
    )
    return RetrievalEvaluationReport(
        dataset_name=dataset.name,
        schema_version=dataset.schema_version,
        top_k=top_k,
        cases=case_metrics,
        aggregate=aggregate,
    )


def retrieval_evaluation_to_dict(
    report: RetrievalEvaluationReport,
) -> dict[str, object]:
    """Serialize a report to stable JSON-compatible fields for automation."""
    if not isinstance(report, RetrievalEvaluationReport):
        raise TypeError("report must be a RetrievalEvaluationReport.")
    return {
        "dataset_name": report.dataset_name,
        "schema_version": report.schema_version,
        "top_k": report.top_k,
        "case_count": len(report.cases),
        "metrics": {
            "hit_rate_at_k": report.aggregate.hit_rate_at_k,
            "mean_precision_at_k": report.aggregate.mean_precision_at_k,
            "mean_recall_at_k": report.aggregate.mean_recall_at_k,
            "mean_reciprocal_rank_at_k": (
                report.aggregate.mean_reciprocal_rank_at_k
            ),
        },
        "cases": [
            {
                "id": case.case_id,
                "query": case.query,
                "retrieved_count": case.retrieved_count,
                "relevant_count": case.relevant_count,
                "matched_relevant_count": case.matched_relevant_count,
                "relevant_retrieved_count": case.relevant_retrieved_count,
                "first_relevant_rank": case.first_relevant_rank,
                "hit_at_k": case.hit_at_k,
                "precision_at_k": case.precision_at_k,
                "recall_at_k": case.recall_at_k,
                "reciprocal_rank_at_k": case.reciprocal_rank_at_k,
            }
            for case in report.cases
        ],
    }


def format_retrieval_evaluation_table(report: RetrievalEvaluationReport) -> str:
    """Render per-query metrics and macro averages as an aligned table."""
    if not isinstance(report, RetrievalEvaluationReport):
        raise TypeError("report must be a RetrievalEvaluationReport.")

    cutoff = report.top_k
    headers = (
        "Case",
        "Returned",
        "Matched",
        f"Hit@{cutoff}",
        f"P@{cutoff}",
        f"R@{cutoff}",
        f"RR@{cutoff}",
        "First rank",
    )
    rows = [
        (
            case.case_id,
            str(case.retrieved_count),
            f"{case.matched_relevant_count}/{case.relevant_count}",
            f"{case.hit_at_k:.3f}",
            f"{case.precision_at_k:.3f}",
            f"{case.recall_at_k:.3f}",
            f"{case.reciprocal_rank_at_k:.3f}",
            "-" if case.first_relevant_rank is None else str(case.first_relevant_rank),
        )
        for case in report.cases
    ]
    rows.append(
        (
            "MACRO",
            "-",
            "-",
            f"{report.aggregate.hit_rate_at_k:.3f}",
            f"{report.aggregate.mean_precision_at_k:.3f}",
            f"{report.aggregate.mean_recall_at_k:.3f}",
            f"{report.aggregate.mean_reciprocal_rank_at_k:.3f}",
            "-",
        )
    )
    widths = tuple(
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    )

    def format_row(values: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(width) if index == 0 else value.rjust(width)
            for index, (value, width) in enumerate(
                zip(values, widths, strict=True)
            )
        )

    separator = "  ".join("-" * width for width in widths)
    return "\n".join(
        (
            f"Retrieval evaluation: {report.dataset_name}",
            f"Cases: {len(report.cases)} | Cutoff: {report.top_k}",
            "",
            format_row(headers),
            separator,
            *(format_row(row) for row in rows),
        )
    )


def _parse_dataset(raw_data: object) -> RetrievalEvaluationDataset:
    """Convert untrusted decoded JSON into validated domain objects."""
    dataset_object = _require_object(raw_data, context="dataset")
    _validate_object_fields(
        dataset_object,
        expected=_DATASET_FIELDS,
        context="dataset",
    )
    raw_cases = dataset_object["cases"]
    if not isinstance(raw_cases, list):
        raise InvalidRetrievalEvaluationDatasetError("cases must be a list.")

    cases = []
    for index, raw_case in enumerate(raw_cases):
        case_object = _require_object(raw_case, context=f"cases[{index}]")
        _validate_object_fields(
            case_object,
            expected=_CASE_FIELDS,
            context=f"cases[{index}]",
        )
        raw_relevant = case_object["relevant"]
        if not isinstance(raw_relevant, list):
            raise InvalidRetrievalEvaluationDatasetError(
                f"cases[{index}].relevant must be a list."
            )
        cases.append(
            RetrievalEvaluationCase(
                case_id=case_object["id"],
                query=case_object["query"],
                relevant_documents=tuple(
                    RelevantDocument(
                        _require_object(
                            selector,
                            context=f"cases[{index}].relevant[{selector_index}]",
                        )
                    )
                    for selector_index, selector in enumerate(raw_relevant)
                ),
            )
        )

    return RetrievalEvaluationDataset(
        name=dataset_object["name"],
        cases=tuple(cases),
        schema_version=dataset_object["schema_version"],
    )


def _evaluate_case(
    case: RetrievalEvaluationCase,
    raw_results: Sequence[RetrievalResult],
    *,
    top_k: int,
) -> RetrievalCaseMetrics:
    """Validate one ranking and calculate metrics at the configured cutoff."""
    if isinstance(raw_results, (str, bytes)) or not isinstance(
        raw_results, Sequence
    ):
        raise RetrievalEvaluationInputError(
            f"retriever output for case {case.case_id!r} must be a sequence."
        )

    results = tuple(raw_results[:top_k])
    for expected_rank, result in enumerate(results, start=1):
        if not isinstance(result, RetrievalResult):
            raise RetrievalEvaluationInputError(
                f"retriever output for case {case.case_id!r} contains an "
                "invalid result."
            )
        if result.rank != expected_rank:
            raise RetrievalEvaluationInputError(
                f"retriever output for case {case.case_id!r} must have "
                "contiguous one-based ranks."
            )

    relevance_flags = tuple(
        any(
            relevant_document.matches(result.document)
            for relevant_document in case.relevant_documents
        )
        for result in results
    )
    matched_relevant_count = sum(
        any(
            relevant_document.matches(result.document)
            for result in results
        )
        for relevant_document in case.relevant_documents
    )
    relevant_retrieved_count = sum(relevance_flags)
    first_relevant_rank = next(
        (index for index, relevant in enumerate(relevance_flags, start=1) if relevant),
        None,
    )

    return RetrievalCaseMetrics(
        case_id=case.case_id,
        query=case.query,
        retrieved_count=len(results),
        relevant_count=len(case.relevant_documents),
        matched_relevant_count=matched_relevant_count,
        relevant_retrieved_count=relevant_retrieved_count,
        first_relevant_rank=first_relevant_rank,
        hit_at_k=1.0 if first_relevant_rank is not None else 0.0,
        precision_at_k=relevant_retrieved_count / top_k,
        recall_at_k=matched_relevant_count / len(case.relevant_documents),
        reciprocal_rank_at_k=(
            0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
        ),
    )


def _require_object(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise InvalidRetrievalEvaluationDatasetError(
            f"{context} must be a JSON object."
        )
    return value


def _validate_object_fields(
    value: Mapping[str, object],
    *,
    expected: frozenset[str],
    context: str,
) -> None:
    """Reject missing and unknown schema fields so dataset typos fail early."""
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise InvalidRetrievalEvaluationDatasetError(
            f"{context} is missing required field(s): {', '.join(missing)}."
        )
    if unknown:
        raise InvalidRetrievalEvaluationDatasetError(
            f"{context} contains unknown field(s): {', '.join(unknown)}."
        )


def _validate_non_empty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidRetrievalEvaluationDatasetError(
            f"{name} must be a non-empty string."
        )
    return value.strip()


def _validate_metadata_scalar(value: object, *, key: str) -> None:
    if isinstance(value, float) and not isfinite(value):
        raise InvalidRetrievalEvaluationDatasetError(
            f"relevant selector value for {key!r} must be finite."
        )
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    raise InvalidRetrievalEvaluationDatasetError(
        f"relevant selector value for {key!r} must be a JSON scalar."
    )


def _validate_top_k(top_k: object) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise RetrievalEvaluationInputError(
            "top_k must be a positive integer."
        )
