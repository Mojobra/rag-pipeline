"""Translate parsed CLI values into validated pipeline configuration objects.

This module is the boundary between argparse's dynamic ``Namespace`` and the
typed configuration contracts used by the pipeline. Builders validate settings
but deliberately avoid model initialization, downloads, filesystem access, and
vector-store I/O.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_pipeline.benchmarking import BenchmarkConfig
    from rag_pipeline.chunking import ChunkingConfig
    from rag_pipeline.embeddings import LocalEmbeddingConfig
    from rag_pipeline.generation import GenerationConfig, LocalGenerationConfig
    from rag_pipeline.reranking import LocalRerankerConfig, RerankingConfig
    from rag_pipeline.retrieval import RetrievalConfig
    from rag_pipeline.sparse_embeddings import LocalSparseEmbeddingConfig
    from rag_pipeline.vector_store import SearchMode, VectorStoreConfig


@dataclass(frozen=True, slots=True)
class IndexCommandConfig:
    """Validated settings needed to build or update a local collection.

    The object keeps indexing handlers focused on orchestration while retaining
    separate dense, optional sparse, chunking, and Qdrant contracts.
    """

    chunking: ChunkingConfig
    embedding: LocalEmbeddingConfig
    vector_store: VectorStoreConfig
    sparse_embedding: LocalSparseEmbeddingConfig | None


@dataclass(frozen=True, slots=True)
class RetrievalRuntimeConfig:
    """Validated service settings shared by retrieval-based CLI commands.

    Interactive retrieval, evaluations, and answer generation use this same
    contract so common flags cannot be interpreted differently. Provider
    resources are still initialized lazily by each command handler.
    """

    embedding: LocalEmbeddingConfig
    vector_store: VectorStoreConfig
    sparse_embedding: LocalSparseEmbeddingConfig | None
    retrieval: RetrievalConfig
    local_reranker: LocalRerankerConfig | None
    reranking: RerankingConfig | None


def build_chunking_config(args: argparse.Namespace) -> ChunkingConfig:
    """Build validated character chunking settings from parsed CLI fields."""
    from rag_pipeline.chunking import ChunkingConfig

    return ChunkingConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )


def build_embedding_config(args: argparse.Namespace) -> LocalEmbeddingConfig:
    """Build the dense model contract shared by indexing and querying."""
    from rag_pipeline.embeddings import LocalEmbeddingConfig

    return LocalEmbeddingConfig(
        model_name=args.model,
        model_revision=args.model_revision,
        device=args.device,
        batch_size=args.batch_size,
    )


def build_generation_configs(
    args: argparse.Namespace,
) -> tuple[LocalGenerationConfig, GenerationConfig]:
    """Build local model and prompt-budget settings for answer generation."""
    from rag_pipeline.generation import GenerationConfig, LocalGenerationConfig

    return (
        LocalGenerationConfig(
            model_name=args.generation_model,
            model_revision=args.generation_model_revision,
            device=args.generation_device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        ),
        GenerationConfig(
            max_context_characters=args.max_context_characters,
            max_input_tokens=args.max_input_tokens,
        ),
    )


def build_index_command_config(
    args: argparse.Namespace,
) -> IndexCommandConfig:
    """Assemble validated chunking, embedding, sparse, and Qdrant settings.

    The builder performs no document loading or inference. Sparse settings are
    created only for hybrid collections, matching the indexing command's
    provider lifecycle.
    """
    from rag_pipeline.vector_store import (
        SearchMode,
        VectorStoreConfig,
    )

    chunking_config = build_chunking_config(args)
    embedding_config = build_embedding_config(args)
    search_mode = SearchMode(args.search_mode)
    vector_store_config = VectorStoreConfig(
        path=args.store_path,
        collection_name=args.collection_name,
        write_batch_size=args.write_batch_size,
        search_mode=search_mode,
    )
    sparse_embedding_config = _build_sparse_embedding_config(
        args,
        search_mode=search_mode,
    )
    return IndexCommandConfig(
        chunking=chunking_config,
        embedding=embedding_config,
        vector_store=vector_store_config,
        sparse_embedding=sparse_embedding_config,
    )


def build_benchmark_config(args: argparse.Namespace) -> BenchmarkConfig:
    """Translate CLI fields into the isolated benchmark orchestration contract.

    Every stage is validated here, but the function performs no filesystem
    access, model initialization, inference, or vector-store writes.
    """
    from rag_pipeline.benchmarking import BenchmarkConfig
    from rag_pipeline.retrieval import RetrievalConfig, parse_metadata_filter
    from rag_pipeline.vector_store import SearchMode

    embedding_config = build_embedding_config(args)
    search_mode = SearchMode(args.search_mode)
    sparse_embedding_config = _build_sparse_embedding_config(
        args,
        search_mode=search_mode,
    )
    (
        local_reranker_config,
        reranking_config,
        retrieval_top_k,
    ) = _build_reranking_configs(args)
    retrieval_config = RetrievalConfig(
        top_k=retrieval_top_k,
        score_threshold=args.score_threshold,
        metadata_filters=tuple(
            parse_metadata_filter(value)
            for value in (args.metadata_filters or ())
        ),
    )
    chunking_config = build_chunking_config(args)
    local_generation_config, generation_config = build_generation_configs(args)
    return BenchmarkConfig(
        name=args.name,
        chunking=chunking_config,
        embedding=embedding_config,
        search_mode=search_mode,
        sparse_embedding=sparse_embedding_config,
        write_batch_size=args.write_batch_size,
        retrieval=retrieval_config,
        local_reranker=local_reranker_config,
        reranking=reranking_config,
        local_generation=local_generation_config,
        generation=generation_config,
        work_directory=args.work_dir,
    )


def build_retrieval_runtime_config(
    args: argparse.Namespace,
) -> RetrievalRuntimeConfig:
    """Translate shared CLI fields into one validated retrieval contract.

    Dense and optional sparse models, Qdrant location, result limits, metadata
    filters, and optional reranking are validated without performing provider
    initialization, downloads, vector-store I/O, or inference.
    """
    from rag_pipeline.retrieval import RetrievalConfig, parse_metadata_filter
    from rag_pipeline.vector_store import SearchMode, VectorStoreConfig

    embedding_config = build_embedding_config(args)
    search_mode = SearchMode(args.search_mode)
    vector_store_config = VectorStoreConfig(
        path=args.store_path,
        collection_name=args.collection_name,
        search_mode=search_mode,
    )
    sparse_embedding_config = _build_sparse_embedding_config(
        args,
        search_mode=search_mode,
    )
    (
        local_reranker_config,
        reranking_config,
        retrieval_top_k,
    ) = _build_reranking_configs(args)
    retrieval_config = RetrievalConfig(
        top_k=retrieval_top_k,
        score_threshold=args.score_threshold,
        metadata_filters=tuple(
            parse_metadata_filter(value)
            for value in (args.metadata_filters or ())
        ),
    )
    return RetrievalRuntimeConfig(
        embedding=embedding_config,
        vector_store=vector_store_config,
        sparse_embedding=sparse_embedding_config,
        retrieval=retrieval_config,
        local_reranker=local_reranker_config,
        reranking=reranking_config,
    )


def _build_sparse_embedding_config(
    args: argparse.Namespace,
    *,
    search_mode: SearchMode,
) -> LocalSparseEmbeddingConfig | None:
    """Build sparse provider settings only when the selected mode is hybrid."""
    from rag_pipeline.sparse_embeddings import LocalSparseEmbeddingConfig
    from rag_pipeline.vector_store import SearchMode

    if search_mode != SearchMode.HYBRID:
        return None
    return LocalSparseEmbeddingConfig(
        model_name=args.sparse_model,
        cache_dir=args.sparse_cache_dir,
        batch_size=args.sparse_batch_size,
        threads=args.sparse_threads,
    )


def _build_reranking_configs(
    args: argparse.Namespace,
) -> tuple[
    LocalRerankerConfig | None,
    RerankingConfig | None,
    int,
]:
    """Build optional reranking settings and the first-stage result width.

    Disabled reranking preserves ``top_k`` as the retrieval width. Enabled
    reranking validates that ``candidate_k`` can satisfy the final result count.
    """
    from rag_pipeline.exceptions import InvalidRerankingConfigurationError
    from rag_pipeline.reranking import LocalRerankerConfig, RerankingConfig

    if not args.rerank:
        return None, None, args.top_k

    reranking_config = RerankingConfig(top_n=args.top_k)
    if args.candidate_k < reranking_config.top_n:
        raise InvalidRerankingConfigurationError(
            "candidate_k must be greater than or equal to top_k."
        )
    local_config = LocalRerankerConfig(
        model_name=args.reranker_model,
        model_revision=args.reranker_model_revision,
        device=args.reranker_device,
        cache_dir=args.reranker_cache_dir,
        batch_size=args.reranker_batch_size,
        max_length=args.reranker_max_length,
    )
    return local_config, reranking_config, args.candidate_k
