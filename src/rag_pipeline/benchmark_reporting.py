"""Compatibility exports for benchmark report construction and persistence."""

from rag_pipeline.benchmarking.reporting import (
    BenchmarkReport,
    benchmark_report_to_dict,
    format_benchmark_summary,
    validate_benchmark_output_path,
    write_benchmark_report,
)

__all__ = [
    "BenchmarkReport",
    "benchmark_report_to_dict",
    "format_benchmark_summary",
    "validate_benchmark_output_path",
    "write_benchmark_report",
]
