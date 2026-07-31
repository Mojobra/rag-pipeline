"""Compatibility exports for deterministic retrieval evaluation."""

from rag_pipeline.evaluation.retrieval import (
    RETRIEVAL_EVALUATION_SCHEMA_VERSION,
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
    "RETRIEVAL_EVALUATION_SCHEMA_VERSION",
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
