"""Represent, serialize, format, and atomically persist benchmark reports.

The runner supplies an in-memory :class:`BenchmarkReport`; this module owns the
stable schema-v1 JSON boundary and local artifact-writing guarantees. It does
not initialize models, run evaluation, or mutate vector collections.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from rag_pipeline.answer_evaluation import (
    AnswerEvaluationReport,
    answer_evaluation_to_dict,
)
from rag_pipeline.benchmark_artifacts import BENCHMARK_SCHEMA_VERSION
from rag_pipeline.benchmark_provenance import (
    CorpusFingerprint,
    DatasetFingerprint,
    corpus_fingerprint_to_dict,
    dataset_fingerprint_to_dict,
)
from rag_pipeline.benchmark_thresholds import (
    ThresholdGateResult,
    threshold_gate_to_dict,
)
from rag_pipeline.benchmark_timing import TimingSummary, timing_to_dict
from rag_pipeline.exceptions import InvalidBenchmarkConfigurationError
from rag_pipeline.retrieval_evaluation import (
    RetrievalEvaluationReport,
    retrieval_evaluation_to_dict,
)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Complete in-memory benchmark result before JSON serialization."""

    name: str
    started_at: str
    finished_at: str
    total_seconds: float
    corpus: CorpusFingerprint
    retrieval_dataset: DatasetFingerprint
    answer_dataset: DatasetFingerprint
    source_revision: Mapping[str, object]
    environment: Mapping[str, object]
    configuration: Mapping[str, object]
    index: Mapping[str, object]
    stage_seconds: Mapping[str, float]
    retrieval_timing: TimingSummary
    answer_timing: TimingSummary
    retrieval_report: RetrievalEvaluationReport
    answer_report: AnswerEvaluationReport
    threshold_gate: ThresholdGateResult | None
    reproducibility_warnings: tuple[str, ...]


def validate_benchmark_output_path(
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate a JSON artifact destination without modifying the filesystem.

    Existing files are rejected unless overwrite is explicitly enabled. Callers
    can run this before model work; the writer repeats the check before I/O.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("benchmark output path must be a string or pathlib.Path.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    resolved_path = Path(path).expanduser().resolve()
    if resolved_path.suffix.lower() != ".json":
        raise InvalidBenchmarkConfigurationError(
            "benchmark output path must use a .json extension."
        )
    if resolved_path.exists():
        if resolved_path.is_dir():
            raise InvalidBenchmarkConfigurationError(
                f"benchmark output path is a directory: {resolved_path}"
            )
        if not overwrite:
            raise InvalidBenchmarkConfigurationError(
                f"benchmark output already exists: {resolved_path}; "
                "choose another path or enable overwrite."
            )
    return resolved_path


def benchmark_report_to_dict(report: BenchmarkReport) -> dict[str, object]:
    """Serialize a report to the stable schema-v1 artifact contract."""
    if not isinstance(report, BenchmarkReport):
        raise TypeError("report must be a BenchmarkReport.")
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "run": {
            "name": report.name,
            "started_at": report.started_at,
            "finished_at": report.finished_at,
        },
        "provenance": {
            "source": dict(report.source_revision),
            "corpus": corpus_fingerprint_to_dict(report.corpus),
            "datasets": {
                "retrieval": dataset_fingerprint_to_dict(report.retrieval_dataset),
                "answer": dataset_fingerprint_to_dict(report.answer_dataset),
            },
            "environment": _json_copy(report.environment),
        },
        "configuration": _json_copy(report.configuration),
        "index": _json_copy(report.index),
        "timings": {
            "total_seconds": report.total_seconds,
            "stages": dict(report.stage_seconds),
            "retrieval": timing_to_dict(report.retrieval_timing),
            "answer": timing_to_dict(report.answer_timing),
        },
        "results": {
            "retrieval": retrieval_evaluation_to_dict(report.retrieval_report),
            "answer": answer_evaluation_to_dict(report.answer_report),
        },
        "threshold_gate": (
            None
            if report.threshold_gate is None
            else threshold_gate_to_dict(report.threshold_gate)
        ),
        "reproducibility_warnings": list(report.reproducibility_warnings),
    }


def write_benchmark_report(
    report: BenchmarkReport,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically persist a benchmark artifact as UTF-8 JSON.

    Parent directories are created. Existing artifacts are preserved unless
    overwrite is explicitly enabled. The function returns the resolved path
    after the temporary file has been flushed and replaced successfully.
    """
    resolved_path = validate_benchmark_output_path(path, overwrite=overwrite)
    try:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InvalidBenchmarkConfigurationError(
            f"failed to create benchmark output directory {resolved_path.parent}: {exc}"
        ) from exc
    payload = json.dumps(
        benchmark_report_to_dict(report),
        ensure_ascii=True,
        indent=2,
        allow_nan=False,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{resolved_path.name}.",
            suffix=".tmp",
            dir=resolved_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_file.write(f"{payload}\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        if resolved_path.exists() and not overwrite:
            raise InvalidBenchmarkConfigurationError(
                f"benchmark output already exists: {resolved_path}; "
                "choose another path or enable overwrite."
            )
        os.replace(temporary_path, resolved_path)
        temporary_path = None
    except OSError as exc:
        raise InvalidBenchmarkConfigurationError(
            f"failed to write benchmark artifact {resolved_path}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
    return resolved_path


def format_benchmark_summary(report: BenchmarkReport) -> str:
    """Render the highest-signal quality, latency, and gate outcomes."""
    if not isinstance(report, BenchmarkReport):
        raise TypeError("report must be a BenchmarkReport.")
    retrieval = report.retrieval_report.aggregate
    answer = report.answer_report.aggregate
    gate_status = (
        "not configured"
        if report.threshold_gate is None
        else ("PASS" if report.threshold_gate.passed else "FAIL")
    )
    return "\n".join(
        (
            f"Benchmark: {report.name}",
            (
                f"Corpus: {report.corpus.file_count} file(s), "
                f"{report.index['chunk_count']} chunk(s)"
            ),
            (
                f"Retrieval: Hit@{report.retrieval_report.top_k}="
                f"{retrieval.hit_rate_at_k:.3f}, "
                f"Recall@{report.retrieval_report.top_k}="
                f"{retrieval.mean_recall_at_k:.3f}, "
                f"p95={report.retrieval_timing.p95_seconds:.4f}s"
            ),
            (
                f"Answer: EM={_optional_ratio(answer.exact_match_rate)}, "
                f"F1={_optional_ratio(answer.mean_token_f1)}, "
                f"abstention_accuracy={answer.abstention_accuracy:.3f}, "
                f"p95={report.answer_timing.p95_seconds:.4f}s"
            ),
            f"Total runtime: {report.total_seconds:.4f}s",
            f"Threshold gate: {gate_status}",
        )
    )


def _json_copy(value: Mapping[str, object]) -> dict[str, object]:
    """Copy JSON-compatible mappings and reject non-finite values."""
    copied = json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))
    return cast(dict[str, object], copied)


def _optional_ratio(value: float | None) -> str:
    """Format an optional ratio for the concise terminal summary."""
    return "-" if value is None else f"{value:.3f}"
