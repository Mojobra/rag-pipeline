"""Define the stable metric registry shared by benchmark gates and comparisons.

Each allowlisted metric has one artifact path, category, preferred direction,
and unit. Central extraction prevents threshold evaluation and comparison
reporting from interpreting saved values differently.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Literal

from rag_pipeline.exceptions import InvalidBenchmarkArtifactError


@dataclass(frozen=True, slots=True)
class _MetricSpec:
    """Map one public metric name to its artifact field and interpretation."""

    name: str
    source_path: tuple[str, ...]
    category: Literal["quality", "latency", "runtime", "storage"]
    direction: Literal["higher", "lower"]
    unit: Literal["ratio", "seconds", "bytes"]


_METRIC_SPECS = (
    _MetricSpec(
        "retrieval.hit_rate_at_k",
        ("results", "retrieval", "metrics", "hit_rate_at_k"),
        "quality",
        "higher",
        "ratio",
    ),
    _MetricSpec(
        "retrieval.mean_precision_at_k",
        ("results", "retrieval", "metrics", "mean_precision_at_k"),
        "quality",
        "higher",
        "ratio",
    ),
    _MetricSpec(
        "retrieval.mean_recall_at_k",
        ("results", "retrieval", "metrics", "mean_recall_at_k"),
        "quality",
        "higher",
        "ratio",
    ),
    _MetricSpec(
        "retrieval.mean_reciprocal_rank_at_k",
        (
            "results",
            "retrieval",
            "metrics",
            "mean_reciprocal_rank_at_k",
        ),
        "quality",
        "higher",
        "ratio",
    ),
    _MetricSpec(
        "answer.exact_match_rate",
        ("results", "answer", "metrics", "exact_match_rate"),
        "quality",
        "higher",
        "ratio",
    ),
    _MetricSpec(
        "answer.mean_token_f1",
        ("results", "answer", "metrics", "mean_token_f1"),
        "quality",
        "higher",
        "ratio",
    ),
    _MetricSpec(
        "answer.abstention_accuracy",
        ("results", "answer", "metrics", "abstention_accuracy"),
        "quality",
        "higher",
        "ratio",
    ),
    _MetricSpec(
        "answer.abstention_precision",
        ("results", "answer", "metrics", "abstention_precision"),
        "quality",
        "higher",
        "ratio",
    ),
    _MetricSpec(
        "answer.abstention_recall",
        ("results", "answer", "metrics", "abstention_recall"),
        "quality",
        "higher",
        "ratio",
    ),
    _MetricSpec(
        "answer.answerable_response_rate",
        ("results", "answer", "metrics", "answerable_response_rate"),
        "quality",
        "higher",
        "ratio",
    ),
    _MetricSpec(
        "answer.citation_behavior_rate",
        ("results", "answer", "metrics", "citation_behavior_rate"),
        "quality",
        "higher",
        "ratio",
    ),
    _MetricSpec(
        "latency.retrieval.mean_seconds",
        ("timings", "retrieval", "mean_seconds"),
        "latency",
        "lower",
        "seconds",
    ),
    _MetricSpec(
        "latency.retrieval.p95_seconds",
        ("timings", "retrieval", "p95_seconds"),
        "latency",
        "lower",
        "seconds",
    ),
    _MetricSpec(
        "latency.answer.mean_seconds",
        ("timings", "answer", "mean_seconds"),
        "latency",
        "lower",
        "seconds",
    ),
    _MetricSpec(
        "latency.answer.p95_seconds",
        ("timings", "answer", "p95_seconds"),
        "latency",
        "lower",
        "seconds",
    ),
    _MetricSpec(
        "runtime.total_seconds",
        ("timings", "total_seconds"),
        "runtime",
        "lower",
        "seconds",
    ),
    _MetricSpec(
        "index.storage_bytes",
        ("index", "storage_bytes"),
        "storage",
        "lower",
        "bytes",
    ),
)
_METRIC_SPECS_BY_NAME = {spec.name: spec for spec in _METRIC_SPECS}


def _metric_values(
    artifact: Mapping[str, object],
) -> dict[str, float | None]:
    """Read and domain-validate every allowlisted metric from an artifact."""
    values: dict[str, float | None] = {}
    for spec in _METRIC_SPECS:
        raw_value = _path_value(artifact, spec.source_path)
        if raw_value is None:
            if spec.category != "quality":
                raise InvalidBenchmarkArtifactError(f"{spec.name} cannot be null.")
            values[spec.name] = None
            continue
        value = _finite_number(raw_value, context=spec.name)
        if spec.unit == "ratio" and not 0.0 <= value <= 1.0:
            raise InvalidBenchmarkArtifactError(f"{spec.name} must be between 0 and 1.")
        if spec.unit in ("seconds", "bytes") and value < 0:
            raise InvalidBenchmarkArtifactError(f"{spec.name} cannot be negative.")
        values[spec.name] = value
    return values


def _path_value(
    value: Mapping[str, object],
    path: tuple[str, ...],
) -> object:
    """Read one required metric path from a nested artifact mapping."""
    current: object = value
    traversed: list[str] = []
    for part in path:
        traversed.append(part)
        if not isinstance(current, Mapping) or part not in current:
            raise InvalidBenchmarkArtifactError(
                f"missing required field {'.'.join(traversed)!r}."
            )
        current = current[part]
    return current


def _finite_number(value: object, *, context: str) -> float:
    """Normalize a finite artifact number while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidBenchmarkArtifactError(f"{context} must be a number.")
    normalized = float(value)
    if not isfinite(normalized):
        raise InvalidBenchmarkArtifactError(f"{context} must be finite.")
    return normalized


def _format_metric(value: float | None, unit: str) -> str:
    """Format an absolute metric value for comparison tables."""
    if value is None:
        return "-"
    if unit == "bytes":
        return str(int(value))
    return f"{value:.4f}"


def _format_signed_metric(value: float | None, unit: str) -> str:
    """Format a signed metric delta for comparison tables."""
    if value is None:
        return "-"
    if unit == "bytes":
        return f"{value:+.0f}"
    return f"{value:+.4f}"


def _format_percentage(value: float | None) -> str:
    """Format an optional relative change as a signed percentage."""
    return "-" if value is None else f"{value:+.1%}"
