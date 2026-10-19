"""Transport-neutral use cases for composing the local RAG domain services.

The application package coordinates provider lifecycles and multi-stage
workflows. It has no argparse, HTTP, or output-rendering dependencies, so the
CLI and API can share the same behavior.
"""

from rag_pipeline.application.answering import AnswerOutcome, answer_from_results
from rag_pipeline.application.indexing import (
    EmbeddingPreview,
    IndexingPipelineConfig,
    index_local_documents,
    preview_local_embeddings,
)
from rag_pipeline.application.retrieval import (
    RetrievalPipeline,
    RetrievalPipelineConfig,
    open_local_retrieval_pipeline,
)

__all__ = [
    "AnswerOutcome",
    "EmbeddingPreview",
    "IndexingPipelineConfig",
    "RetrievalPipeline",
    "RetrievalPipelineConfig",
    "answer_from_results",
    "index_local_documents",
    "open_local_retrieval_pipeline",
    "preview_local_embeddings",
]
