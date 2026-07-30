"""Validate benchmark artifacts, apply gates, and compare compatible runs.

This module owns the stable JSON-facing metric registry. It treats saved
artifacts and threshold files as untrusted input, verifies comparison
provenance, and distinguishes portable quality deltas from hardware-sensitive
operational measurements.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import cast

from rag_pipeline.benchmark_metrics import (
    _METRIC_SPECS,
    _format_metric,
    _format_percentage,
    _format_signed_metric,
    _metric_values,
    _MetricSpec,
)
from rag_pipeline.benchmark_thresholds import (
    BENCHMARK_THRESHOLD_SCHEMA_VERSION,
    BenchmarkThreshold,
    BenchmarkThresholdApplicability,
    BenchmarkThresholdProfile,
    ThresholdCheckResult,
    ThresholdGateResult,
    _evaluate_threshold,
    load_benchmark_threshold_profile,
    threshold_gate_to_dict,
    validate_benchmark_threshold_applicability,
)
from rag_pipeline.exceptions import (
    BenchmarkComparisonError,
    InvalidBenchmarkArtifactError,
)

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "BENCHMARK_THRESHOLD_SCHEMA_VERSION",
    "BenchmarkComparison",
    "BenchmarkMetricDelta",
    "BenchmarkThreshold",
    "BenchmarkThresholdApplicability",
    "BenchmarkThresholdProfile",
    "ThresholdCheckResult",
    "ThresholdGateResult",
    "benchmark_comparison_to_dict",
    "compare_benchmark_artifacts",
    "evaluate_benchmark_thresholds",
    "format_benchmark_comparison_table",
    "load_benchmark_artifact",
    "load_benchmark_threshold_profile",
    "threshold_gate_to_dict",
    "validate_benchmark_threshold_applicability",
]

BENCHMARK_SCHEMA_VERSION = 1

_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "run",
        "provenance",
        "configuration",
        "index",
        "timings",
        "results",
        "threshold_gate",
        "reproducibility_warnings",
    }
)


@dataclass(frozen=True, slots=True)
class BenchmarkMetricDelta:
    """Baseline and candidate values plus their signed change."""

    metric: str
    category: str
    direction: str
    unit: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    relative_change: float | None
    comparable: bool


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """Metric deltas for artifacts with compatible evaluation ground truth."""

    baseline_name: str
    candidate_name: str
    changed_configuration_sections: tuple[str, ...]
    latency_comparable: bool
    latency_warning: str | None
    metrics: tuple[BenchmarkMetricDelta, ...]


def evaluate_benchmark_thresholds(
    artifact: Mapping[str, object],
    profile: BenchmarkThresholdProfile,
) -> ThresholdGateResult:
    """Apply every inclusive metric bound to a complete report artifact."""
    if not isinstance(profile, BenchmarkThresholdProfile):
        raise TypeError("profile must be a BenchmarkThresholdProfile.")
    validated_artifact = _copy_and_validate_artifact(
        artifact,
        context="benchmark",
    )
    validate_benchmark_threshold_applicability(
        profile,
        corpus_sha256=_artifact_digest(
            validated_artifact,
            ("provenance", "corpus", "sha256"),
            context="corpus sha256",
        ),
        retrieval_dataset_sha256=_artifact_digest(
            validated_artifact,
            ("provenance", "datasets", "retrieval", "sha256"),
            context="retrieval dataset sha256",
        ),
        answer_dataset_sha256=_artifact_digest(
            validated_artifact,
            ("provenance", "datasets", "answer", "sha256"),
            context="answer dataset sha256",
        ),
        top_k=_artifact_top_k(validated_artifact),
    )
    metric_values = _metric_values(validated_artifact)
    results = tuple(
        _evaluate_threshold(check, metric_values[check.metric])
        for check in profile.checks
    )
    return ThresholdGateResult(
        profile_name=profile.name,
        profile_sha256=profile.sha256,
        applicability_verified=True,
        passed=all(result.passed for result in results),
        checks=results,
    )


def load_benchmark_artifact(path: str | Path) -> dict[str, object]:
    """Load and validate stable fields needed for gates and comparisons."""
    resolved_path, _, raw_data = _load_json_file(
        path,
        error_type=InvalidBenchmarkArtifactError,
        description="benchmark artifact",
    )
    try:
        artifact = _require_object(
            raw_data,
            context="benchmark artifact",
            error_type=InvalidBenchmarkArtifactError,
        )
        _validate_artifact(artifact)
    except InvalidBenchmarkArtifactError as exc:
        raise InvalidBenchmarkArtifactError(
            f"invalid benchmark artifact {resolved_path}: {exc}"
        ) from exc
    return artifact


def compare_benchmark_artifacts(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> BenchmarkComparison:
    """Compare metrics after proving corpus, labels, and cutoff compatibility.

    Pipeline settings may differ because they are normal experiment variables.
    Quality stays comparable across machines; operational metrics are marked
    diagnostic when software, platform, or configured devices differ.
    """
    baseline_artifact = _copy_and_validate_artifact(
        baseline,
        context="baseline",
    )
    candidate_artifact = _copy_and_validate_artifact(
        candidate,
        context="candidate",
    )
    _validate_comparison_provenance(
        baseline_artifact,
        candidate_artifact,
    )

    latency_comparable = _environment_signature(
        baseline_artifact
    ) == _environment_signature(candidate_artifact)
    warning = (
        None
        if latency_comparable
        else (
            "Runtime environment or inference devices differ; latency, total "
            "runtime, and storage deltas are diagnostic only."
        )
    )
    baseline_values = _metric_values(baseline_artifact)
    candidate_values = _metric_values(candidate_artifact)
    metrics = tuple(
        _build_metric_delta(
            spec,
            baseline_values[spec.name],
            candidate_values[spec.name],
            operationally_comparable=latency_comparable,
        )
        for spec in _METRIC_SPECS
    )
    return BenchmarkComparison(
        baseline_name=_run_name(baseline_artifact),
        candidate_name=_run_name(candidate_artifact),
        changed_configuration_sections=_changed_configuration_sections(
            baseline_artifact,
            candidate_artifact,
        ),
        latency_comparable=latency_comparable,
        latency_warning=warning,
        metrics=metrics,
    )


def benchmark_comparison_to_dict(
    comparison: BenchmarkComparison,
) -> dict[str, object]:
    """Serialize a comparison for scripts and stored diagnostics."""
    if not isinstance(comparison, BenchmarkComparison):
        raise TypeError("comparison must be a BenchmarkComparison.")
    return {
        "baseline_name": comparison.baseline_name,
        "candidate_name": comparison.candidate_name,
        "ground_truth_compatible": True,
        "changed_configuration_sections": list(
            comparison.changed_configuration_sections
        ),
        "latency_comparable": comparison.latency_comparable,
        "latency_warning": comparison.latency_warning,
        "metrics": [
            {
                "metric": metric.metric,
                "category": metric.category,
                "direction": metric.direction,
                "unit": metric.unit,
                "baseline": metric.baseline,
                "candidate": metric.candidate,
                "delta": metric.delta,
                "relative_change": metric.relative_change,
                "comparable": metric.comparable,
            }
            for metric in comparison.metrics
        ],
    }


def format_benchmark_comparison_table(
    comparison: BenchmarkComparison,
) -> str:
    """Render metric deltas and comparability state as a compact table."""
    if not isinstance(comparison, BenchmarkComparison):
        raise TypeError("comparison must be a BenchmarkComparison.")
    headers = (
        "Metric",
        "Baseline",
        "Candidate",
        "Delta",
        "Change",
        "Better",
        "Comparable",
    )
    rows = [
        (
            metric.metric,
            _format_metric(metric.baseline, metric.unit),
            _format_metric(metric.candidate, metric.unit),
            _format_signed_metric(metric.delta, metric.unit),
            _format_percentage(metric.relative_change),
            "higher" if metric.direction == "higher" else "lower",
            "yes" if metric.comparable else "no",
        )
        for metric in comparison.metrics
    ]
    widths = tuple(
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    )

    def format_row(values: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(width) if index == 0 else value.rjust(width)
            for index, (value, width) in enumerate(zip(values, widths, strict=True))
        )

    lines = [
        (
            f"Benchmark comparison: {comparison.baseline_name} -> "
            f"{comparison.candidate_name}"
        ),
        "Ground truth: compatible",
        (
            "Configuration sections changed: "
            + (
                ", ".join(comparison.changed_configuration_sections)
                if comparison.changed_configuration_sections
                else "none"
            )
        ),
        (
            "Operational metrics: comparable"
            if comparison.latency_comparable
            else "Operational metrics: diagnostic only"
        ),
    ]
    if comparison.latency_warning is not None:
        lines.append(f"Warning: {comparison.latency_warning}")
    lines.extend(
        (
            "",
            format_row(headers),
            "  ".join("-" * width for width in widths),
            *(format_row(row) for row in rows),
        )
    )
    return "\n".join(lines)


def _build_metric_delta(
    spec: _MetricSpec,
    baseline: float | None,
    candidate: float | None,
    *,
    operationally_comparable: bool,
) -> BenchmarkMetricDelta:
    delta: float | None = None
    if baseline is not None and candidate is not None:
        delta = candidate - baseline
    relative_change = None
    if delta is not None and baseline is not None and baseline != 0:
        relative_change = delta / abs(baseline)
    return BenchmarkMetricDelta(
        metric=spec.name,
        category=spec.category,
        direction=spec.direction,
        unit=spec.unit,
        baseline=baseline,
        candidate=candidate,
        delta=delta,
        relative_change=relative_change,
        comparable=(
            delta is not None
            and (spec.category == "quality" or operationally_comparable)
        ),
    )


def _validate_artifact(artifact: Mapping[str, object]) -> None:
    """Validate schema identity and every field used by gates or comparisons."""
    _validate_fields(
        artifact,
        expected=_ARTIFACT_FIELDS,
        context="benchmark artifact",
        error_type=InvalidBenchmarkArtifactError,
    )
    schema_version = artifact["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != BENCHMARK_SCHEMA_VERSION
    ):
        raise InvalidBenchmarkArtifactError(
            f"unsupported schema_version; expected {BENCHMARK_SCHEMA_VERSION}."
        )
    _run_name(artifact)
    _artifact_digest(
        artifact,
        ("provenance", "corpus", "sha256"),
        context="corpus sha256",
    )
    _artifact_digest(
        artifact,
        ("provenance", "datasets", "retrieval", "sha256"),
        context="retrieval dataset sha256",
    )
    _artifact_digest(
        artifact,
        ("provenance", "datasets", "answer", "sha256"),
        context="answer dataset sha256",
    )
    _artifact_top_k(artifact)
    _metric_values(artifact)
    _environment_signature(artifact)


def _copy_and_validate_artifact(
    artifact: Mapping[str, object],
    *,
    context: str,
) -> dict[str, object]:
    """Create an isolated JSON-compatible copy and validate its contracts."""
    if not isinstance(artifact, Mapping):
        raise TypeError(f"{context} artifact must be a mapping.")
    try:
        copied = cast(
            dict[str, object],
            json.loads(json.dumps(artifact, ensure_ascii=True, allow_nan=False)),
        )
    except (TypeError, ValueError) as exc:
        raise InvalidBenchmarkArtifactError(
            f"{context} artifact is not JSON-compatible: {exc}"
        ) from exc
    try:
        _validate_artifact(copied)
    except InvalidBenchmarkArtifactError as exc:
        raise InvalidBenchmarkArtifactError(
            f"invalid {context} benchmark artifact: {exc}"
        ) from exc
    return copied


def _validate_comparison_provenance(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> None:
    """Require identical ground truth while allowing experiment settings to vary."""
    compatibility_fields = (
        ("corpus fingerprint", ("provenance", "corpus", "sha256")),
        (
            "retrieval dataset fingerprint",
            ("provenance", "datasets", "retrieval", "sha256"),
        ),
        (
            "answer dataset fingerprint",
            ("provenance", "datasets", "answer", "sha256"),
        ),
        ("retrieval top_k", ("configuration", "retrieval", "top_k")),
    )
    mismatches = [
        label
        for label, path in compatibility_fields
        if _path_value(
            baseline,
            path,
            error_type=InvalidBenchmarkArtifactError,
        )
        != _path_value(
            candidate,
            path,
            error_type=InvalidBenchmarkArtifactError,
        )
    ]
    if mismatches:
        raise BenchmarkComparisonError(
            "benchmark artifacts are not comparable because these ground-truth "
            f"contracts differ: {', '.join(mismatches)}."
        )


def _changed_configuration_sections(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> tuple[str, ...]:
    """Return sorted top-level configuration sections with unequal values."""
    baseline_config = _path_value(
        baseline,
        ("configuration",),
        error_type=InvalidBenchmarkArtifactError,
    )
    candidate_config = _path_value(
        candidate,
        ("configuration",),
        error_type=InvalidBenchmarkArtifactError,
    )
    if not isinstance(baseline_config, Mapping) or not isinstance(
        candidate_config,
        Mapping,
    ):
        raise InvalidBenchmarkArtifactError("configuration must be a JSON object.")
    sections = set(baseline_config) | set(candidate_config)
    return tuple(
        sorted(
            section
            for section in sections
            if baseline_config.get(section) != candidate_config.get(section)
        )
    )


def _environment_signature(artifact: Mapping[str, object]) -> str:
    """Build the canonical software, hardware, and device comparison identity."""
    signature = {
        "platform": _path_value(
            artifact,
            ("provenance", "environment", "platform"),
            error_type=InvalidBenchmarkArtifactError,
        ),
        "python": _path_value(
            artifact,
            ("provenance", "environment", "python"),
            error_type=InvalidBenchmarkArtifactError,
        ),
        "packages": _path_value(
            artifact,
            ("provenance", "environment", "packages"),
            error_type=InvalidBenchmarkArtifactError,
        ),
        "accelerator": _path_value(
            artifact,
            ("provenance", "environment", "accelerator"),
            error_type=InvalidBenchmarkArtifactError,
        ),
        "devices": {
            "embedding": _path_value(
                artifact,
                ("configuration", "embedding", "device"),
                error_type=InvalidBenchmarkArtifactError,
            ),
            "generation": _path_value(
                artifact,
                ("configuration", "generation", "device"),
                error_type=InvalidBenchmarkArtifactError,
            ),
            "reranking": _optional_path_value(
                artifact,
                ("configuration", "reranking", "device"),
            ),
        },
    }
    try:
        return json.dumps(
            signature,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise InvalidBenchmarkArtifactError(
            f"runtime environment is not JSON-compatible: {exc}"
        ) from exc


def _run_name(artifact: Mapping[str, object]) -> str:
    return _non_empty_string(
        _path_value(
            artifact,
            ("run", "name"),
            error_type=InvalidBenchmarkArtifactError,
        ),
        context="run.name",
        error_type=InvalidBenchmarkArtifactError,
    )


def _artifact_top_k(artifact: Mapping[str, object]) -> int:
    value = _path_value(
        artifact,
        ("configuration", "retrieval", "top_k"),
        error_type=InvalidBenchmarkArtifactError,
    )
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidBenchmarkArtifactError(
            "configuration.retrieval.top_k must be a positive integer."
        )
    return value


def _artifact_digest(
    artifact: Mapping[str, object],
    path: tuple[str, ...],
    *,
    context: str,
) -> str:
    return _sha256_digest(
        _path_value(
            artifact,
            path,
            error_type=InvalidBenchmarkArtifactError,
        ),
        context=context,
        error_type=InvalidBenchmarkArtifactError,
    )


def _load_json_file(
    path: str | Path,
    *,
    error_type: type[ValueError],
    description: str,
) -> tuple[Path, bytes, object]:
    if not isinstance(path, (str, Path)):
        raise TypeError(f"{description} path must be a string or pathlib.Path.")
    resolved_path = Path(path).expanduser().resolve()
    try:
        content = resolved_path.read_bytes()
        raw_data = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise error_type(
            f"failed to load {description} {resolved_path}: {exc}"
        ) from exc
    return resolved_path, content, raw_data


def _path_value(
    value: Mapping[str, object],
    path: tuple[str, ...],
    *,
    error_type: type[ValueError],
) -> object:
    current: object = value
    traversed = []
    for part in path:
        traversed.append(part)
        if not isinstance(current, Mapping) or part not in current:
            raise error_type(f"missing required field {'.'.join(traversed)!r}.")
        current = current[part]
    return current


def _optional_path_value(
    value: Mapping[str, object],
    path: tuple[str, ...],
) -> object:
    current: object = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _require_object(
    value: object,
    *,
    context: str,
    error_type: type[ValueError],
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise error_type(f"{context} must be a JSON object.")
    return value


def _validate_fields(
    value: Mapping[str, object],
    *,
    expected: frozenset[str],
    context: str,
    error_type: type[ValueError],
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise error_type(
            f"{context} is missing required field(s): {', '.join(missing)}."
        )
    if unknown:
        raise error_type(f"{context} contains unknown field(s): {', '.join(unknown)}.")


def _non_empty_string(
    value: object,
    *,
    context: str,
    error_type: type[ValueError],
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{context} must be a non-empty string.")
    return value.strip()


def _sha256_digest(
    value: object,
    *,
    context: str,
    error_type: type[ValueError],
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise error_type(f"{context} must be a lowercase SHA-256 digest.")
    return value


def _finite_number(
    value: object,
    *,
    context: str,
    error_type: type[ValueError],
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error_type(f"{context} must be a number.")
    normalized = float(value)
    if not isfinite(normalized):
        raise error_type(f"{context} must be finite.")
    return normalized
