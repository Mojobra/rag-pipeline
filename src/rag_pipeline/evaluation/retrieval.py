"""Measure ranked retrieval quality against explicit relevance judgments.

The module loads a versioned JSON dataset, matches returned LangChain document
metadata to binary relevance selectors, calculates standard top-k metrics, and
formats deterministic reports without invoking answer generation.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias, cast

from langchain_core.documents import Document

from rag_pipeline.exceptions import (
    InvalidRetrievalEvaluationDatasetError,
    RetrievalEvaluationInputError,
)
from rag_pipeline.retrieval import RetrievalResult

# Preserve the original public constructor default for callers using v1 labels.
RETRIEVAL_EVALUATION_SCHEMA_VERSION = 1
LATEST_RETRIEVAL_EVALUATION_SCHEMA_VERSION = 3
SUPPORTED_RETRIEVAL_EVALUATION_SCHEMA_VERSIONS = frozenset({1, 2, 3})

MetadataScalar: TypeAlias = str | int | float | bool | None
RetrievalFunction: TypeAlias = Callable[[str], Sequence[RetrievalResult]]

_MISSING = object()
_DATASET_FIELDS = frozenset({"schema_version", "name", "cases"})
_CASE_FIELDS = frozenset({"id", "query", "relevant"})
_CONTENT_SELECTOR_FIELDS = frozenset({"metadata", "content_contains"})


@dataclass(frozen=True, slots=True)
class RelevantDocument:
    """Metadata and optional source-text selector for one relevant evidence span.

    Every key-value pair must occur on a returned LangChain document for the
    selector to match. Schema-v2 datasets additionally require an exact content
    anchor, keeping judgments valid when an experiment changes chunk boundaries.
    """

    metadata: Mapping[str, MetadataScalar]
    content_contains: str | None = None

    def __post_init__(self) -> None:
        """Validate and freeze metadata plus an optional exact content anchor."""
        if not isinstance(self.metadata, Mapping):
            raise InvalidRetrievalEvaluationDatasetError(
                "relevant selectors must be JSON objects."
            )
        if not self.metadata and self.content_contains is None:
            raise InvalidRetrievalEvaluationDatasetError(
                "relevant selectors require metadata or a content anchor."
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
        if self.content_contains is not None:
            content_anchor = _validate_non_empty_string(
                "content_contains",
                self.content_contains,
            )
            object.__setattr__(self, "content_contains", content_anchor)

    def matches(self, document: Document) -> bool:
        """Return whether metadata and source content satisfy this judgment.

        Value types must also match, preventing Python's equality rules from
        treating metadata such as boolean ``true`` as integer ``1``. Content
        anchors use deterministic, case-sensitive substring matching.
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
        return (
            self.content_contains is None
            or self.content_contains in document.page_content
        )


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCase:
    """One query and its binary metadata-based relevance judgments.

    Case IDs identify regressions in reports, while the query is passed unchanged
    to the configured retriever. Empty relevance requires explicit opt-in and
    denotes a negative case only in a containing schema-v3 dataset.
    """

    case_id: str
    query: str
    relevant_documents: tuple[RelevantDocument, ...]
    allow_empty_relevance: bool = False

    def __post_init__(self) -> None:
        """Normalize text fields and reject malformed or duplicate judgments."""
        normalized_case_id = _validate_non_empty_string("case id", self.case_id)
        normalized_query = _validate_non_empty_string("query", self.query)
        try:
            relevant_documents = tuple(self.relevant_documents)
        except TypeError as exc:
            raise InvalidRetrievalEvaluationDatasetError(
                "relevant must be a list of metadata selectors."
            ) from exc
        if not isinstance(self.allow_empty_relevance, bool):
            raise InvalidRetrievalEvaluationDatasetError(
                "allow_empty_relevance must be a boolean."
            )
        if not relevant_documents and not self.allow_empty_relevance:
            raise InvalidRetrievalEvaluationDatasetError(
                f"case {normalized_case_id!r} must contain relevant selectors."
            )
        if relevant_documents and self.allow_empty_relevance:
            raise InvalidRetrievalEvaluationDatasetError(
                "allow_empty_relevance is only valid for negative cases."
            )
        canonical_selectors: set[str] = set()
        for relevant_document in relevant_documents:
            if not isinstance(relevant_document, RelevantDocument):
                raise InvalidRetrievalEvaluationDatasetError(
                    "relevant must contain RelevantDocument objects."
                )
            canonical_selector = json.dumps(
                {
                    "metadata": dict(relevant_document.metadata),
                    "content_contains": relevant_document.content_contains,
                },
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
    fixed schema version keep comparisons reproducible; schema v1 remains the
    programmatic default for backward compatibility.
    """

    name: str
    cases: tuple[RetrievalEvaluationCase, ...]
    schema_version: int = RETRIEVAL_EVALUATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate schema compatibility, dataset identity, and unique cases."""
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version not in SUPPORTED_RETRIEVAL_EVALUATION_SCHEMA_VERSIONS
        ):
            raise InvalidRetrievalEvaluationDatasetError(
                "unsupported retrieval evaluation schema_version; expected 1, 2, or 3."
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
        positive_case_count = 0
        for case in cases:
            if not isinstance(case, RetrievalEvaluationCase):
                raise InvalidRetrievalEvaluationDatasetError(
                    "cases must contain RetrievalEvaluationCase objects."
                )
            if case.case_id in seen_case_ids:
                raise InvalidRetrievalEvaluationDatasetError(
                    f"duplicate retrieval evaluation case id {case.case_id!r}."
                )
            if not case.relevant_documents and self.schema_version != 3:
                raise InvalidRetrievalEvaluationDatasetError(
                    f"case {case.case_id!r} must contain relevant selectors."
                )
            if case.relevant_documents:
                positive_case_count += 1
            for relevant_document in case.relevant_documents:
                has_content_anchor = relevant_document.content_contains is not None
                if self.schema_version == 1 and has_content_anchor:
                    raise InvalidRetrievalEvaluationDatasetError(
                        "schema_version 1 does not support content anchors."
                    )
                if self.schema_version in (2, 3) and not has_content_anchor:
                    raise InvalidRetrievalEvaluationDatasetError(
                        f"schema_version {self.schema_version} requires content anchors."
                    )
                if self.schema_version in (2, 3) and not relevant_document.metadata:
                    raise InvalidRetrievalEvaluationDatasetError(
                        f"schema_version {self.schema_version} requires non-empty "
                        "metadata selectors."
                    )
            seen_case_ids.add(case.case_id)

        if not positive_case_count:
            raise InvalidRetrievalEvaluationDatasetError(
                "retrieval evaluation datasets require at least one positive case "
                "to calculate Hit, Precision, Recall, and MRR."
            )

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "cases", cases)


@dataclass(frozen=True, slots=True)
class RetrievalCaseMetrics:
    """Top-k retrieval metrics and diagnostics for one labeled query.

    Precision counts empty result slots as misses by using ``top_k`` as its
    denominator. Recall counts distinct relevance selectors matched by at least
    one returned document. Positive metrics are not assigned to negative cases.
    """

    case_id: str
    query: str
    retrieved_count: int
    relevant_count: int
    matched_relevant_count: int
    relevant_retrieved_count: int
    first_relevant_rank: int | None
    hit_at_k: float | None
    precision_at_k: float | None
    recall_at_k: float | None
    reciprocal_rank_at_k: float | None
    matched_selector_indices: tuple[int, ...] = ()
    missing_selector_indices: tuple[int, ...] = ()
    redundant_relevant_ranks: tuple[int, ...] = ()
    unwanted_ranks: tuple[int, ...] = ()
    no_hit_at_k: float | None = None
    false_positive_at_k: float | None = None


@dataclass(frozen=True, slots=True)
class RetrievalAggregateMetrics:
    """Positive-case macro metrics and separate negative-case rates."""

    hit_rate_at_k: float
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank_at_k: float
    positive_case_count: int = 0
    negative_case_count: int = 0
    negative_no_hit_rate_at_k: float | None = None
    negative_false_positive_rate_at_k: float | None = None


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

    return score_retrieval_results(
        dataset,
        tuple(retrieve(case.query) for case in dataset.cases),
        top_k=top_k,
    )


def score_retrieval_results(
    dataset: RetrievalEvaluationDataset,
    results_by_case: Sequence[Sequence[RetrievalResult]],
    *,
    top_k: int,
) -> RetrievalEvaluationReport:
    """Score precomputed rankings in dataset order without repeating retrieval.

    This separates provider I/O from metric computation when an experiment
    evaluates the same first-stage pool at several cutoffs. It rejects missing
    or extra case rankings and delegates rank validation to the case scorer.
    """
    if not isinstance(dataset, RetrievalEvaluationDataset):
        raise TypeError("dataset must be a RetrievalEvaluationDataset.")
    _validate_top_k(top_k)
    if isinstance(results_by_case, (str, bytes)) or not isinstance(
        results_by_case, Sequence
    ):
        raise RetrievalEvaluationInputError("results_by_case must be a sequence.")
    if len(results_by_case) != len(dataset.cases):
        raise RetrievalEvaluationInputError(
            "results_by_case must have one ranking per dataset case."
        )

    case_metrics = tuple(
        _evaluate_case(case, results, top_k=top_k)
        for case, results in zip(dataset.cases, results_by_case, strict=True)
    )
    positive_cases = tuple(case for case in case_metrics if case.relevant_count > 0)
    negative_cases = tuple(case for case in case_metrics if case.relevant_count == 0)
    positive_count = len(positive_cases)
    negative_count = len(negative_cases)
    aggregate = RetrievalAggregateMetrics(
        hit_rate_at_k=sum(case.hit_at_k or 0.0 for case in positive_cases)
        / positive_count,
        mean_precision_at_k=(
            sum(case.precision_at_k or 0.0 for case in positive_cases) / positive_count
        ),
        mean_recall_at_k=(
            sum(case.recall_at_k or 0.0 for case in positive_cases) / positive_count
        ),
        mean_reciprocal_rank_at_k=(
            sum(case.reciprocal_rank_at_k or 0.0 for case in positive_cases)
            / positive_count
        ),
        positive_case_count=positive_count,
        negative_case_count=negative_count,
        negative_no_hit_rate_at_k=(
            None
            if not negative_count
            else sum(case.no_hit_at_k or 0.0 for case in negative_cases)
            / negative_count
        ),
        negative_false_positive_rate_at_k=(
            None
            if not negative_count
            else sum(case.false_positive_at_k or 0.0 for case in negative_cases)
            / negative_count
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
    serialized: dict[str, object] = {
        "dataset_name": report.dataset_name,
        "schema_version": report.schema_version,
        "top_k": report.top_k,
        "case_count": len(report.cases),
        "metrics": {
            "hit_rate_at_k": report.aggregate.hit_rate_at_k,
            "mean_precision_at_k": report.aggregate.mean_precision_at_k,
            "mean_recall_at_k": report.aggregate.mean_recall_at_k,
            "mean_reciprocal_rank_at_k": (report.aggregate.mean_reciprocal_rank_at_k),
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
    if report.schema_version == 3:
        metrics = cast(dict[str, object], serialized["metrics"])
        metrics.update(
            {
                "positive_case_count": report.aggregate.positive_case_count,
                "negative_case_count": report.aggregate.negative_case_count,
                "negative_no_hit_rate_at_k": (
                    report.aggregate.negative_no_hit_rate_at_k
                ),
                "negative_false_positive_rate_at_k": (
                    report.aggregate.negative_false_positive_rate_at_k
                ),
            }
        )
        cases = cast(list[dict[str, object]], serialized["cases"])
        for case_data, case in zip(cases, report.cases, strict=True):
            case_data.update(
                {
                    "case_type": "negative" if case.relevant_count == 0 else "positive",
                    "matched_selector_indices": list(case.matched_selector_indices),
                    "missing_selector_indices": list(case.missing_selector_indices),
                    "redundant_relevant_ranks": list(case.redundant_relevant_ranks),
                    "unwanted_ranks": list(case.unwanted_ranks),
                    "no_hit_at_k": case.no_hit_at_k,
                    "false_positive_at_k": case.false_positive_at_k,
                }
            )
    return serialized


def format_retrieval_evaluation_table(report: RetrievalEvaluationReport) -> str:
    """Render per-query metrics and macro averages as an aligned table."""
    if not isinstance(report, RetrievalEvaluationReport):
        raise TypeError("report must be a RetrievalEvaluationReport.")

    if report.schema_version == 3:
        return _format_v3_retrieval_table(report)

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
            for index, (value, width) in enumerate(zip(values, widths, strict=True))
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


def _format_v3_retrieval_table(report: RetrievalEvaluationReport) -> str:
    """Show positive evidence coverage and negative no-hit behavior separately."""
    cutoff = report.top_k
    headers = (
        "Case",
        "Type",
        "Returned",
        "Evidence",
        "Dup",
        "Unwanted",
        f"Hit@{cutoff}",
        f"P@{cutoff}",
        f"R@{cutoff}",
        f"RR@{cutoff}",
        "No-hit",
    )
    rows = [
        (
            case.case_id,
            "NEG" if case.relevant_count == 0 else "POS",
            str(case.retrieved_count),
            f"{case.matched_relevant_count}/{case.relevant_count}",
            str(len(case.redundant_relevant_ranks)),
            str(len(case.unwanted_ranks)),
            *_format_optional_metrics(case),
            _format_optional_metric(case.no_hit_at_k),
        )
        for case in report.cases
    ]
    rows.append(
        (
            "MACRO",
            "-",
            "-",
            "-",
            "-",
            "-",
            f"{report.aggregate.hit_rate_at_k:.3f}",
            f"{report.aggregate.mean_precision_at_k:.3f}",
            f"{report.aggregate.mean_recall_at_k:.3f}",
            f"{report.aggregate.mean_reciprocal_rank_at_k:.3f}",
            _format_optional_metric(report.aggregate.negative_no_hit_rate_at_k),
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

    separator = "  ".join("-" * width for width in widths)
    return "\n".join(
        (
            f"Retrieval evaluation: {report.dataset_name}",
            f"Positive: {report.aggregate.positive_case_count} | "
            f"Negative: {report.aggregate.negative_case_count} | Cutoff: {cutoff}",
            "Negative false-positive rate: "
            + _format_optional_metric(
                report.aggregate.negative_false_positive_rate_at_k
            ),
            "",
            format_row(headers),
            separator,
            *(format_row(row) for row in rows),
        )
    )


def _format_optional_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _format_optional_metrics(case: RetrievalCaseMetrics) -> tuple[str, ...]:
    return (
        _format_optional_metric(case.hit_at_k),
        _format_optional_metric(case.precision_at_k),
        _format_optional_metric(case.recall_at_k),
        _format_optional_metric(case.reciprocal_rank_at_k),
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
    schema_version = dataset_object["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version not in SUPPORTED_RETRIEVAL_EVALUATION_SCHEMA_VERSIONS
    ):
        raise InvalidRetrievalEvaluationDatasetError(
            "unsupported retrieval evaluation schema_version; expected 1, 2, or 3."
        )

    cases: list[RetrievalEvaluationCase] = []
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
                case_id=cast(str, case_object["id"]),
                query=cast(str, case_object["query"]),
                relevant_documents=tuple(
                    _parse_relevant_selector(
                        selector,
                        schema_version=schema_version,
                        context=f"cases[{index}].relevant[{selector_index}]",
                    )
                    for selector_index, selector in enumerate(raw_relevant)
                ),
                allow_empty_relevance=(schema_version == 3 and not raw_relevant),
            )
        )

    return RetrievalEvaluationDataset(
        name=cast(str, dataset_object["name"]),
        cases=tuple(cases),
        schema_version=schema_version,
    )


def _parse_relevant_selector(
    raw_selector: object,
    *,
    schema_version: int,
    context: str,
) -> RelevantDocument:
    """Parse a legacy metadata selector or a v2 metadata-plus-anchor selector."""
    selector = _require_object(raw_selector, context=context)
    if schema_version == 1:
        return RelevantDocument(cast(dict[str, MetadataScalar], selector))

    _validate_object_fields(
        selector,
        expected=_CONTENT_SELECTOR_FIELDS,
        context=context,
    )
    metadata = _require_object(selector["metadata"], context=f"{context}.metadata")
    if not metadata:
        raise InvalidRetrievalEvaluationDatasetError(
            f"{context}.metadata cannot be empty."
        )
    return RelevantDocument(
        cast(dict[str, MetadataScalar], metadata),
        content_contains=cast(str, selector["content_contains"]),
    )


def _evaluate_case(
    case: RetrievalEvaluationCase,
    raw_results: Sequence[RetrievalResult],
    *,
    top_k: int,
) -> RetrievalCaseMetrics:
    """Validate one ranking and calculate metrics at the configured cutoff."""
    if isinstance(raw_results, (str, bytes)) or not isinstance(raw_results, Sequence):
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

    selectors_by_rank = tuple(
        tuple(
            index
            for index, selector in enumerate(case.relevant_documents)
            if selector.matches(result.document)
        )
        for result in results
    )
    relevance_flags = tuple(bool(indices) for indices in selectors_by_rank)
    matched_selector_indices = tuple(
        sorted({index for indices in selectors_by_rank for index in indices})
    )
    matched_relevant_count = len(matched_selector_indices)
    relevant_retrieved_count = sum(relevance_flags)
    first_relevant_rank = next(
        (index for index, relevant in enumerate(relevance_flags, start=1) if relevant),
        None,
    )

    seen_selectors: set[int] = set()
    redundant_ranks: list[int] = []
    for rank, indices in enumerate(selectors_by_rank, start=1):
        if indices and set(indices).issubset(seen_selectors):
            redundant_ranks.append(rank)
        seen_selectors.update(indices)

    is_negative = not case.relevant_documents
    return RetrievalCaseMetrics(
        case_id=case.case_id,
        query=case.query,
        retrieved_count=len(results),
        relevant_count=len(case.relevant_documents),
        matched_relevant_count=matched_relevant_count,
        relevant_retrieved_count=relevant_retrieved_count,
        first_relevant_rank=first_relevant_rank,
        hit_at_k=(
            None if is_negative else 1.0 if first_relevant_rank is not None else 0.0
        ),
        precision_at_k=(None if is_negative else relevant_retrieved_count / top_k),
        recall_at_k=(
            None
            if is_negative
            else matched_relevant_count / len(case.relevant_documents)
        ),
        reciprocal_rank_at_k=(
            None
            if is_negative
            else 0.0
            if first_relevant_rank is None
            else 1.0 / first_relevant_rank
        ),
        matched_selector_indices=matched_selector_indices,
        missing_selector_indices=tuple(
            index
            for index in range(len(case.relevant_documents))
            if index not in matched_selector_indices
        ),
        redundant_relevant_ranks=tuple(redundant_ranks),
        unwanted_ranks=tuple(
            rank
            for rank, indices in enumerate(selectors_by_rank, start=1)
            if not indices
        ),
        no_hit_at_k=(None if not is_negative else 1.0 if not results else 0.0),
        false_positive_at_k=(None if not is_negative else 0.0 if not results else 1.0),
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
        raise RetrievalEvaluationInputError("top_k must be a positive integer.")
