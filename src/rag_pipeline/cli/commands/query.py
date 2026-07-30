"""Register and execute interactive retrieval and answer commands."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from rag_pipeline.cli.options import (
    DEFAULT_ANSWER_SCORE_THRESHOLD,
    add_embedding_arguments,
    add_generation_arguments,
    add_hybrid_search_arguments,
    add_reranking_arguments,
    add_retrieval_arguments,
    add_vector_store_location_arguments,
)

if TYPE_CHECKING:
    from rag_pipeline.reranking import LocalRerankerConfig, RerankingConfig
    from rag_pipeline.retrieval import RetrievalResult


def register_retrieve_command(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ranked evidence inspection command."""
    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Inspect ranked chunks from an existing Qdrant collection.",
        description=(
            "Embed a query, search a compatible Qdrant collection, and print ranked "
            "evidence. Optional filtering, hybrid search, and reranking apply before "
            "output; no answer model is loaded."
        ),
    )
    retrieve_parser.add_argument(
        "query",
        help=(
            "Text to embed and match against indexed chunks. Specific wording and "
            "keywords can materially change dense and hybrid retrieval results."
        ),
    )
    add_embedding_arguments(retrieve_parser)
    add_vector_store_location_arguments(retrieve_parser)
    add_hybrid_search_arguments(retrieve_parser)
    add_retrieval_arguments(retrieve_parser)
    add_reranking_arguments(retrieve_parser)
    retrieve_parser.set_defaults(_handler=run_retrieve)


def register_answer_command(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the grounded answer generation command."""
    answer_parser = subparsers.add_parser(
        "answer",
        help="Generate a cited local answer from retrieved evidence.",
        description=(
            "Retrieve eligible chunks from Qdrant, optionally rerank them, and pass "
            "bounded evidence to a local generation model. Returns a deterministic "
            "abstention when no chunk passes retrieval criteria."
        ),
    )
    answer_parser.add_argument(
        "query",
        help=(
            "Question used for both retrieval and grounded generation. Clear, "
            "specific wording generally yields more focused evidence and answers."
        ),
    )
    add_embedding_arguments(answer_parser)
    add_vector_store_location_arguments(answer_parser)
    add_hybrid_search_arguments(answer_parser)
    add_retrieval_arguments(
        answer_parser,
        default_score_threshold=DEFAULT_ANSWER_SCORE_THRESHOLD,
    )
    add_reranking_arguments(answer_parser)
    add_generation_arguments(answer_parser)
    answer_parser.set_defaults(_handler=run_answer)


def run_retrieve(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Retrieve, optionally rerank, and print evidence from local Qdrant."""
    from rag_pipeline.cli.config import build_retrieval_runtime_config
    from rag_pipeline.cli.output import format_retrieval_results
    from rag_pipeline.embeddings import (
        InvalidEmbeddingConfigurationError,
        create_local_embedding_service,
    )
    from rag_pipeline.exceptions import (
        InvalidRerankingConfigurationError,
        InvalidRetrievalConfigurationError,
        InvalidVectorStoreConfigurationError,
    )
    from rag_pipeline.retrieval import RetrieverService
    from rag_pipeline.sparse_embeddings import (
        create_local_sparse_embedding_service,
    )
    from rag_pipeline.vector_store import LocalVectorStore

    try:
        runtime_config = build_retrieval_runtime_config(args)
    except (
        InvalidEmbeddingConfigurationError,
        InvalidVectorStoreConfigurationError,
        InvalidRetrievalConfigurationError,
        InvalidRerankingConfigurationError,
    ) as exc:
        parser.error(str(exc))

    embedding_service = create_local_embedding_service(
        runtime_config.embedding
    )
    sparse_embedding_service = (
        create_local_sparse_embedding_service(runtime_config.sparse_embedding)
        if runtime_config.sparse_embedding is not None
        else None
    )
    with LocalVectorStore(runtime_config.vector_store) as vector_store:
        results = RetrieverService(
            embedding_service,
            vector_store,
            sparse_embedding_service,
        ).retrieve(args.query, config=runtime_config.retrieval)

    results = _rerank_results(
        args.query,
        results,
        local_config=runtime_config.local_reranker,
        config=runtime_config.reranking,
    )
    print(format_retrieval_results(results))
    return 0


def run_answer(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Retrieve evidence and generate a grounded, cited local answer.

    Retrieval and optional reranking complete before the generation model is
    initialized. No eligible evidence therefore produces the deterministic
    abstention without paying generation model startup or inference cost.
    """
    from rag_pipeline.cli.config import (
        build_generation_configs,
        build_retrieval_runtime_config,
    )
    from rag_pipeline.cli.output import (
        format_abstention_answer,
        format_generated_answer,
    )
    from rag_pipeline.embeddings import (
        InvalidEmbeddingConfigurationError,
        create_local_embedding_service,
    )
    from rag_pipeline.exceptions import (
        InvalidGenerationConfigurationError,
        InvalidRerankingConfigurationError,
        InvalidRetrievalConfigurationError,
        InvalidVectorStoreConfigurationError,
    )
    from rag_pipeline.generation import (
        INSUFFICIENT_CONTEXT_ANSWER,
        create_local_answer_generator,
    )
    from rag_pipeline.retrieval import RetrieverService
    from rag_pipeline.sparse_embeddings import (
        create_local_sparse_embedding_service,
    )
    from rag_pipeline.vector_store import LocalVectorStore

    try:
        runtime_config = build_retrieval_runtime_config(args)
        local_generation_config, generation_config = build_generation_configs(
            args
        )
    except (
        InvalidEmbeddingConfigurationError,
        InvalidVectorStoreConfigurationError,
        InvalidRetrievalConfigurationError,
        InvalidRerankingConfigurationError,
        InvalidGenerationConfigurationError,
    ) as exc:
        parser.error(str(exc))

    embedding_service = create_local_embedding_service(
        runtime_config.embedding
    )
    sparse_embedding_service = (
        create_local_sparse_embedding_service(runtime_config.sparse_embedding)
        if runtime_config.sparse_embedding is not None
        else None
    )
    with LocalVectorStore(runtime_config.vector_store) as vector_store:
        retrieval_results = RetrieverService(
            embedding_service,
            vector_store,
            sparse_embedding_service,
        ).retrieve(args.query, config=runtime_config.retrieval)

    retrieval_results = _rerank_results(
        args.query,
        retrieval_results,
        local_config=runtime_config.local_reranker,
        config=runtime_config.reranking,
    )

    if not retrieval_results:
        print(format_abstention_answer(INSUFFICIENT_CONTEXT_ANSWER))
        return 0

    answer_generator = create_local_answer_generator(local_generation_config)
    generated_answer = answer_generator.generate(
        args.query,
        retrieval_results,
        config=generation_config,
    )
    print(format_generated_answer(generated_answer))
    return 0


def _rerank_results(
    query: str,
    results: list[RetrievalResult],
    *,
    local_config: LocalRerankerConfig | None,
    config: RerankingConfig | None,
) -> list[RetrievalResult]:
    """Optionally initialize the local reranker and reorder retrieved results.

    Empty input or disabled reranking returns the original list unchanged.
    Otherwise this may download/cache a model and performs cross-encoder
    inference through the existing reranking service.
    """
    if not results or config is None:
        return results
    if local_config is None:
        raise RuntimeError("Reranking config has no local model configuration.")

    from rag_pipeline.reranking import create_local_reranker_service

    reranker = create_local_reranker_service(local_config)
    return reranker.rerank(query, results, config=config)
