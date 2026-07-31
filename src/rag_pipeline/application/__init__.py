"""Transport-neutral use cases for composing the local RAG domain services.

The application package coordinates provider lifecycles and multi-stage
workflows. It deliberately has no argparse or terminal-output dependencies, so
the same use cases can support the CLI today and an HTTP adapter later.
"""

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
    "EmbeddingPreview",
    "IndexingPipelineConfig",
    "RetrievalPipeline",
    "RetrievalPipelineConfig",
    "index_local_documents",
    "open_local_retrieval_pipeline",
    "preview_local_embeddings",
]
