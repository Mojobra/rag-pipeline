"""Compatibility exports for deterministic retrieval evaluation."""

from rag_pipeline.evaluation.retrieval import (
    LATEST_RETRIEVAL_EVALUATION_SCHEMA_VERSION,
    RETRIEVAL_EVALUATION_SCHEMA_VERSION,
    SUPPORTED_RETRIEVAL_EVALUATION_SCHEMA_VERSIONS,
    RelevantDocument,
    RetrievalAggregateMetrics,
    RetrievalCaseMetrics,
    RetrievalEvaluationCase,
    RetrievalEvaluationDataset,
    RetrievalEvaluationReport,
    evaluate_retrieval,
    format_retrieval_evaluation_table,
    load_retrieval_evaluation_dataset,
    retrieval_evaluation_to_dict,
)

__all__ = [
    "LATEST_RETRIEVAL_EVALUATION_SCHEMA_VERSION",
    "RETRIEVAL_EVALUATION_SCHEMA_VERSION",
    "SUPPORTED_RETRIEVAL_EVALUATION_SCHEMA_VERSIONS",
    "RelevantDocument",
    "RetrievalAggregateMetrics",
    "RetrievalCaseMetrics",
    "RetrievalEvaluationCase",
    "RetrievalEvaluationDataset",
    "RetrievalEvaluationReport",
    "evaluate_retrieval",
    "format_retrieval_evaluation_table",
    "load_retrieval_evaluation_dataset",
    "retrieval_evaluation_to_dict",
]
