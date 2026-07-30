"""Parse benchmark threshold policy and evaluate allowlisted metric bounds.

Threshold profiles are strict, versioned, and bound to immutable corpus and
dataset fingerprints plus the final retrieval cutoff. Metric extraction is
shared with artifact comparison so gates and reports interpret values equally.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Literal, TypeAlias, cast

from rag_pipeline.benchmark_metrics import (
    _METRIC_SPECS_BY_NAME,
)
from rag_pipeline.exceptions import InvalidBenchmarkThresholdsError

BENCHMARK_THRESHOLD_SCHEMA_VERSION = 1
ThresholdOperator: TypeAlias = Literal["minimum", "maximum"]

_PROFILE_FIELDS = frozenset({"schema_version", "name", "applies_to", "checks"})
_APPLICABILITY_FIELDS = frozenset(
    {
        "corpus_sha256",
        "retrieval_dataset_sha256",
        "answer_dataset_sha256",
        "top_k",
    }
)
_CHECK_FIELDS = frozenset({"metric", "operator", "value"})


@dataclass(frozen=True, slots=True)
class BenchmarkThreshold:
    """One inclusive lower or upper bound on an allowlisted benchmark metric."""

    metric: str
    operator: ThresholdOperator
    value: float

    def __post_init__(self) -> None:
        """Reject unknown metrics, operators, and impossible numeric bounds."""
        if not isinstance(self.metric, str) or self.metric not in _METRIC_SPECS_BY_NAME:
            supported = ", ".join(sorted(_METRIC_SPECS_BY_NAME))
            raise InvalidBenchmarkThresholdsError(
                f"unsupported benchmark metric {self.metric!r}; "
                f"supported metrics: {supported}."
            )
        if self.operator not in ("minimum", "maximum"):
            raise InvalidBenchmarkThresholdsError(
                "threshold operator must be 'minimum' or 'maximum'."
            )
        normalized_value = _finite_number(
            self.value,
            context="threshold value",
            error_type=InvalidBenchmarkThresholdsError,
        )
        metric_spec = _METRIC_SPECS_BY_NAME[self.metric]
        if metric_spec.unit == "ratio" and not 0.0 <= normalized_value <= 1.0:
            raise InvalidBenchmarkThresholdsError(
                f"threshold for {self.metric!r} must be between 0 and 1."
            )
        if metric_spec.unit in ("seconds", "bytes") and normalized_value < 0:
            raise InvalidBenchmarkThresholdsError(
                f"threshold for {self.metric!r} cannot be negative."
            )
        object.__setattr__(self, "value", normalized_value)


@dataclass(frozen=True, slots=True)
class BenchmarkThresholdApplicability:
    """Ground-truth fingerprints and cutoff a threshold profile governs."""

    corpus_sha256: str
    retrieval_dataset_sha256: str
    answer_dataset_sha256: str
    top_k: int

    def __post_init__(self) -> None:
        """Validate immutable input identities and the positive final cutoff."""
        for field_name in (
            "corpus_sha256",
            "retrieval_dataset_sha256",
            "answer_dataset_sha256",
        ):
            _sha256_digest(
                getattr(self, field_name),
                context=field_name,
                error_type=InvalidBenchmarkThresholdsError,
            )
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k <= 0
        ):
            raise InvalidBenchmarkThresholdsError(
                "threshold applicability top_k must be a positive integer."
            )


@dataclass(frozen=True, slots=True)
class BenchmarkThresholdProfile:
    """Named, versioned collection of benchmark regression checks.

    A profile loaded from disk records its SHA-256 digest in the resulting
    artifact. Undefined optional metrics fail configured checks rather than
    disappearing from the gate.
    """

    name: str
    applies_to: BenchmarkThresholdApplicability
    checks: tuple[BenchmarkThreshold, ...]
    schema_version: int = BENCHMARK_THRESHOLD_SCHEMA_VERSION
    sha256: str | None = None

    def __post_init__(self) -> None:
        """Validate profile identity, schema version, and unique checks."""
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != BENCHMARK_THRESHOLD_SCHEMA_VERSION
        ):
            raise InvalidBenchmarkThresholdsError(
                "unsupported benchmark threshold schema_version; expected "
                f"{BENCHMARK_THRESHOLD_SCHEMA_VERSION}."
            )
        normalized_name = _non_empty_string(
            self.name,
            context="threshold profile name",
            error_type=InvalidBenchmarkThresholdsError,
        )
        if not isinstance(
            self.applies_to,
            BenchmarkThresholdApplicability,
        ):
            raise InvalidBenchmarkThresholdsError(
                "applies_to must be a BenchmarkThresholdApplicability."
            )
        try:
            checks = tuple(self.checks)
        except TypeError as exc:
            raise InvalidBenchmarkThresholdsError(
                "threshold checks must be a list."
            ) from exc
        if not checks:
            raise InvalidBenchmarkThresholdsError(
                "threshold profiles must contain at least one check."
            )

        seen_checks: set[tuple[str, str]] = set()
        for check in checks:
            if not isinstance(check, BenchmarkThreshold):
                raise InvalidBenchmarkThresholdsError(
                    "threshold checks must contain BenchmarkThreshold objects."
                )
            key = (check.metric, check.operator)
            if key in seen_checks:
                raise InvalidBenchmarkThresholdsError(
                    f"duplicate {check.operator} check for {check.metric!r}."
                )
            seen_checks.add(key)
        if self.sha256 is not None:
            _sha256_digest(
                self.sha256,
                context="threshold profile sha256",
                error_type=InvalidBenchmarkThresholdsError,
            )
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "checks", checks)


@dataclass(frozen=True, slots=True)
class ThresholdCheckResult:
    """Observed metric value and outcome for one configured threshold."""

    metric: str
    operator: ThresholdOperator
    threshold: float
    observed: float | None
    passed: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ThresholdGateResult:
    """Aggregate pass/fail result for a named threshold profile."""

    profile_name: str
    profile_sha256: str | None
    applicability_verified: bool
    passed: bool
    checks: tuple[ThresholdCheckResult, ...]


def load_benchmark_threshold_profile(
    path: str | Path,
) -> BenchmarkThresholdProfile:
    """Load and strictly validate a versioned UTF-8 threshold profile.

    The function performs filesystem I/O and records a source digest so the
    benchmark artifact identifies the exact gate definition that was applied.
    """
    resolved_path, content, raw_data = _load_json_file(
        path,
        error_type=InvalidBenchmarkThresholdsError,
        description="benchmark threshold profile",
    )
    try:
        profile = _require_object(
            raw_data,
            context="threshold profile",
            error_type=InvalidBenchmarkThresholdsError,
        )
        _validate_fields(
            profile,
            expected=_PROFILE_FIELDS,
            context="threshold profile",
            error_type=InvalidBenchmarkThresholdsError,
        )
        raw_checks = profile["checks"]
        if not isinstance(raw_checks, list):
            raise InvalidBenchmarkThresholdsError("checks must be a list.")
        applies_to = _parse_threshold_applicability(profile["applies_to"])
        checks = tuple(
            _parse_threshold_check(raw_check, index=index)
            for index, raw_check in enumerate(raw_checks)
        )
        return BenchmarkThresholdProfile(
            name=cast(str, profile["name"]),
            applies_to=applies_to,
            checks=checks,
            schema_version=cast(int, profile["schema_version"]),
            sha256=sha256(content).hexdigest(),
        )
    except (InvalidBenchmarkThresholdsError, TypeError) as exc:
        message = (
            str(exc)
            if isinstance(exc, InvalidBenchmarkThresholdsError)
            else f"invalid field type: {exc}"
        )
        raise InvalidBenchmarkThresholdsError(
            f"invalid benchmark threshold profile {resolved_path}: {message}"
        ) from exc


def validate_benchmark_threshold_applicability(
    profile: BenchmarkThresholdProfile,
    *,
    corpus_sha256: str,
    retrieval_dataset_sha256: str,
    answer_dataset_sha256: str,
    top_k: int,
) -> None:
    """Reject a threshold profile intended for different ground truth.

    This can run after input hashing but before model initialization, avoiding
    an expensive run against an unrelated corpus, label snapshot, or cutoff.
    """
    if not isinstance(profile, BenchmarkThresholdProfile):
        raise TypeError("profile must be a BenchmarkThresholdProfile.")
    actual = BenchmarkThresholdApplicability(
        corpus_sha256=corpus_sha256,
        retrieval_dataset_sha256=retrieval_dataset_sha256,
        answer_dataset_sha256=answer_dataset_sha256,
        top_k=top_k,
    )
    expected = profile.applies_to
    mismatches = [
        field_name
        for field_name in (
            "corpus_sha256",
            "retrieval_dataset_sha256",
            "answer_dataset_sha256",
            "top_k",
        )
        if getattr(actual, field_name) != getattr(expected, field_name)
    ]
    if mismatches:
        raise InvalidBenchmarkThresholdsError(
            f"threshold profile {profile.name!r} does not apply to this "
            f"benchmark; mismatched field(s): {', '.join(mismatches)}."
        )


def threshold_gate_to_dict(gate: ThresholdGateResult) -> dict[str, object]:
    """Serialize a gate result to stable JSON-compatible primitives."""
    if not isinstance(gate, ThresholdGateResult):
        raise TypeError("gate must be a ThresholdGateResult.")
    return {
        "profile_name": gate.profile_name,
        "profile_sha256": gate.profile_sha256,
        "applicability_verified": gate.applicability_verified,
        "passed": gate.passed,
        "checks": [
            {
                "metric": check.metric,
                "operator": check.operator,
                "threshold": check.threshold,
                "observed": check.observed,
                "passed": check.passed,
                "reason": check.reason,
            }
            for check in gate.checks
        ],
    }


def _parse_threshold_applicability(
    raw_value: object,
) -> BenchmarkThresholdApplicability:
    """Parse the immutable ground-truth identity from one profile."""
    applicability = _require_object(
        raw_value,
        context="applies_to",
        error_type=InvalidBenchmarkThresholdsError,
    )
    _validate_fields(
        applicability,
        expected=_APPLICABILITY_FIELDS,
        context="applies_to",
        error_type=InvalidBenchmarkThresholdsError,
    )
    return BenchmarkThresholdApplicability(
        corpus_sha256=cast(str, applicability["corpus_sha256"]),
        retrieval_dataset_sha256=cast(
            str,
            applicability["retrieval_dataset_sha256"],
        ),
        answer_dataset_sha256=cast(str, applicability["answer_dataset_sha256"]),
        top_k=cast(int, applicability["top_k"]),
    )


def _parse_threshold_check(
    raw_check: object,
    *,
    index: int,
) -> BenchmarkThreshold:
    """Parse one allowlisted metric bound from a profile check list."""
    check = _require_object(
        raw_check,
        context=f"checks[{index}]",
        error_type=InvalidBenchmarkThresholdsError,
    )
    _validate_fields(
        check,
        expected=_CHECK_FIELDS,
        context=f"checks[{index}]",
        error_type=InvalidBenchmarkThresholdsError,
    )
    return BenchmarkThreshold(
        metric=cast(str, check["metric"]),
        operator=cast(ThresholdOperator, check["operator"]),
        value=cast(float, check["value"]),
    )


def _evaluate_threshold(
    threshold: BenchmarkThreshold,
    observed: float | None,
) -> ThresholdCheckResult:
    """Evaluate one inclusive bound, failing when its metric is undefined."""
    if observed is None:
        return ThresholdCheckResult(
            metric=threshold.metric,
            operator=threshold.operator,
            threshold=threshold.value,
            observed=None,
            passed=False,
            reason="metric is undefined for this dataset",
        )
    passed = (
        observed >= threshold.value
        if threshold.operator == "minimum"
        else observed <= threshold.value
    )
    return ThresholdCheckResult(
        metric=threshold.metric,
        operator=threshold.operator,
        threshold=threshold.value,
        observed=observed,
        passed=passed,
    )


def _load_json_file(
    path: str | Path,
    *,
    error_type: type[ValueError],
    description: str,
) -> tuple[Path, bytes, object]:
    """Read and decode one UTF-8 JSON file with a caller-specific error type."""
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


def _require_object(
    value: object,
    *,
    context: str,
    error_type: type[ValueError],
) -> dict[str, object]:
    """Require one decoded JSON object."""
    if not isinstance(value, dict):
        raise error_type(f"{context} must be a JSON object.")
    return cast(dict[str, object], value)


def _validate_fields(
    value: Mapping[str, object],
    *,
    expected: frozenset[str],
    context: str,
    error_type: type[ValueError],
) -> None:
    """Reject missing and unknown JSON fields at a strict schema boundary."""
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
    """Normalize a required non-empty JSON string."""
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{context} must be a non-empty string.")
    return value.strip()


def _sha256_digest(
    value: object,
    *,
    context: str,
    error_type: type[ValueError],
) -> str:
    """Validate a canonical lowercase SHA-256 digest."""
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
    """Normalize a finite JSON number while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error_type(f"{context} must be a number.")
    normalized = float(value)
    if not isfinite(normalized):
        raise error_type(f"{context} must be finite.")
    return normalized
