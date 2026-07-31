"""Compatibility exports for ingestion chunking experiments."""

from rag_pipeline.ingestion.experiments import (
    DEFAULT_CHUNKING_CANDIDATES,
    ChunkingExperimentReport,
    ChunkingExperimentResult,
    ChunkingMetrics,
    chunking_experiment_to_dict,
    format_chunking_experiment_table,
    parse_chunking_candidate,
    run_chunking_experiment,
)

__all__ = [
    "DEFAULT_CHUNKING_CANDIDATES",
    "ChunkingExperimentReport",
    "ChunkingExperimentResult",
    "ChunkingMetrics",
    "chunking_experiment_to_dict",
    "format_chunking_experiment_table",
    "parse_chunking_candidate",
    "run_chunking_experiment",
]
