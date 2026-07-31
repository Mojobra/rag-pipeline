"""Measure deterministic benchmark stages and summarize per-case latency.

The helpers use an injected monotonic clock, reject malformed or negative
durations, and serialize timing data without owning any RAG provider lifecycle.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from math import ceil, floor, isfinite

from rag_pipeline.exceptions import BenchmarkInputError

Clock = Callable[[], float]


@dataclass(frozen=True, slots=True)
class CaseTiming:
    """Wall-clock duration associated with one evaluation case."""

    case_id: str
    seconds: float


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """Aggregate and per-case latency for one measured evaluation pass.

    Percentiles use linear interpolation. They support like-for-like regression
    checks, but one pass is not a statistically rigorous service load test.
    """

    total_seconds: float
    mean_seconds: float
    p50_seconds: float
    p95_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    cases: tuple[CaseTiming, ...]


class StageRecorder:
    """Record named, non-overlapping benchmark stages with an injected clock."""

    def __init__(self, clock: Clock) -> None:
        """Create an empty recorder without reading the clock."""
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self._clock = clock
        self._durations: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Measure one named block and reject duplicate stage names."""
        if name in self._durations:
            raise RuntimeError(f"benchmark stage {name!r} was measured twice.")
        started = self._clock()
        try:
            yield
        finally:
            self._durations[name] = elapsed_seconds(
                started,
                self._clock(),
                context=f"benchmark stage {name!r}",
            )

    @property
    def durations(self) -> dict[str, float]:
        """Return a copy so callers cannot mutate recorded measurements."""
        return dict(self._durations)


def summarize_case_timings(
    case_ids: Sequence[str],
    durations: Sequence[float],
) -> TimingSummary:
    """Validate ordered case durations and calculate descriptive latency data."""
    if isinstance(case_ids, (str, bytes)) or isinstance(
        durations,
        (str, bytes),
    ):
        raise BenchmarkInputError("case_ids and durations must be sequences.")
    case_id_values = tuple(case_ids)
    duration_values = tuple(durations)
    if not case_id_values:
        raise BenchmarkInputError("at least one case timing is required.")
    if len(case_id_values) != len(duration_values):
        raise BenchmarkInputError(
            "case_ids and durations must contain the same number of values."
        )

    cases = tuple(
        (
            CaseTiming(
                case_id=_non_empty_string(
                    case_id,
                    context=f"case_ids[{index}]",
                ),
                seconds=_duration(duration, context=f"durations[{index}]"),
            )
        )
        for index, (case_id, duration) in enumerate(
            zip(case_id_values, duration_values, strict=True)
        )
    )

    seconds = tuple(case.seconds for case in cases)
    total_seconds = sum(seconds)
    return TimingSummary(
        total_seconds=total_seconds,
        mean_seconds=total_seconds / len(seconds),
        p50_seconds=_percentile(seconds, 0.50),
        p95_seconds=_percentile(seconds, 0.95),
        minimum_seconds=min(seconds),
        maximum_seconds=max(seconds),
        cases=cases,
    )


def timing_to_dict(summary: TimingSummary) -> dict[str, object]:
    """Serialize one timing summary to stable JSON-compatible primitives."""
    if not isinstance(summary, TimingSummary):
        raise TypeError("summary must be a TimingSummary.")
    return {
        "total_seconds": summary.total_seconds,
        "mean_seconds": summary.mean_seconds,
        "p50_seconds": summary.p50_seconds,
        "p95_seconds": summary.p95_seconds,
        "minimum_seconds": summary.minimum_seconds,
        "maximum_seconds": summary.maximum_seconds,
        "cases": [
            {"id": case.case_id, "seconds": case.seconds} for case in summary.cases
        ],
    }


def elapsed_seconds(started: float, finished: float, *, context: str) -> float:
    """Validate two clock readings and return a non-negative duration."""
    if isinstance(started, bool) or not isinstance(started, (int, float)):
        raise BenchmarkInputError(f"{context} start time must be numeric.")
    if isinstance(finished, bool) or not isinstance(finished, (int, float)):
        raise BenchmarkInputError(f"{context} finish time must be numeric.")
    elapsed = float(finished) - float(started)
    if not isfinite(elapsed) or elapsed < 0:
        raise BenchmarkInputError(
            f"{context} clock must produce finite monotonic values."
        )
    return elapsed


def _duration(value: object, *, context: str) -> float:
    """Normalize one finite, non-negative duration value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkInputError(f"{context} must be numeric seconds.")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0:
        raise BenchmarkInputError(f"{context} must be finite non-negative seconds.")
    return normalized


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Calculate a linearly interpolated percentile for ordered observations."""
    if not values:
        raise BenchmarkInputError("percentile values cannot be empty.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower_index = floor(position)
    upper_index = ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def _non_empty_string(value: object, *, context: str) -> str:
    """Normalize a required case identifier with benchmark input errors."""
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkInputError(f"{context} must be a non-empty string.")
    return value.strip()
