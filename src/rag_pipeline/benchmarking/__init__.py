"""Public benchmark runner API and feature package boundary."""

from rag_pipeline.benchmarking.runner import (
    BENCHMARK_COLLECTION_NAME,
    BenchmarkConfig,
    BenchmarkReport,
    CaseTiming,
    TimingSummary,
    benchmark_report_to_dict,
    format_benchmark_summary,
    run_benchmark,
    summarize_case_timings,
    validate_benchmark_output_path,
    write_benchmark_report,
)

__all__ = [
    "BENCHMARK_COLLECTION_NAME",
    "BenchmarkConfig",
    "BenchmarkReport",
    "CaseTiming",
    "TimingSummary",
    "benchmark_report_to_dict",
    "format_benchmark_summary",
    "run_benchmark",
    "summarize_case_timings",
    "validate_benchmark_output_path",
    "write_benchmark_report",
]
