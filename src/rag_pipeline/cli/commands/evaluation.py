"""Register and execute retrieval and answer evaluation CLI commands."""

from __future__ import annotations

import argparse
import json
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
    from rag_pipeline.generation import GeneratedAnswer
    from rag_pipeline.retrieval import RetrievalResult


def register_evaluation_commands(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register labeled retrieval and answer evaluation handlers."""
    retrieval_parser = subparsers.add_parser(
        "evaluate-retrieval",
        help="Measure retrieval quality on a labeled query dataset.",
        description=(
            "Run every query in a versioned JSON dataset through the configured "
            "Qdrant retrieval path and report binary top-k metrics. Optional "
            "filters, hybrid search, and reranking are included; generation is "
            "never invoked and the index is not modified."
        ),
    )
    retrieval_parser.add_argument(
        "dataset",
        help=(
            "UTF-8 JSON file containing schema_version, name, and labeled query "
            "cases with exact metadata selectors under relevant. Change it to "
            "evaluate a different corpus or relevance-judgment snapshot."
        ),
    )
    add_embedding_arguments(retrieval_parser)
    add_vector_store_location_arguments(retrieval_parser)
    add_hybrid_search_arguments(retrieval_parser)
    add_retrieval_arguments(retrieval_parser)
    add_reranking_arguments(retrieval_parser)
    retrieval_parser.add_argument(
        "--output-format",
        choices=("table", "json"),
        default="table",
        help=(
            "Render the same per-case and macro metrics as an aligned terminal "
            "table or structured JSON. Use JSON when saving or comparing runs "
            "programmatically (default: table)."
        ),
    )
    retrieval_parser.set_defaults(_handler=run_evaluate_retrieval)

    answer_parser = subparsers.add_parser(
        "evaluate-answer",
        help="Measure answer quality, abstention, and citation behavior.",
        description=(
            "Run each case in a versioned JSON dataset through retrieval and "
            "grounded generation, then report deterministic reference, "
            "abstention, and citation-state metrics. The command reads an "
            "existing collection and never modifies the index."
        ),
    )
    answer_parser.add_argument(
        "dataset",
        help=(
            "UTF-8 JSON file containing schema_version, name, and labeled cases "
            "with should_abstain plus reference_answers. Change it to evaluate "
            "a different corpus, question set, or accepted-answer snapshot."
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
    answer_parser.add_argument(
        "--output-format",
        choices=("table", "json"),
        default="table",
        help=(
            "Render identical case and aggregate metrics as an aligned table or "
            "structured JSON. Use JSON for saved comparisons and later benchmark "
            "automation (default: table)."
        ),
    )
    answer_parser.set_defaults(_handler=run_evaluate_answer)


def run_evaluate_retrieval(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Evaluate all labeled queries while reusing one retrieval service stack."""
    from rag_pipeline.cli.config import build_retrieval_runtime_config
    from rag_pipeline.embeddings import (
        InvalidEmbeddingConfigurationError,
        create_local_embedding_service,
    )
    from rag_pipeline.exceptions import (
        InvalidRerankingConfigurationError,
        InvalidRetrievalConfigurationError,
        InvalidRetrievalEvaluationDatasetError,
        InvalidVectorStoreConfigurationError,
    )
    from rag_pipeline.reranking import create_local_reranker_service
    from rag_pipeline.retrieval import RetrieverService
    from rag_pipeline.retrieval_evaluation import (
        evaluate_retrieval,
        format_retrieval_evaluation_table,
        load_retrieval_evaluation_dataset,
        retrieval_evaluation_to_dict,
    )
    from rag_pipeline.sparse_embeddings import (
        create_local_sparse_embedding_service,
    )
    from rag_pipeline.vector_store import LocalVectorStore

    try:
        dataset = load_retrieval_evaluation_dataset(args.dataset)
        runtime_config = build_retrieval_runtime_config(args)
    except (
        InvalidEmbeddingConfigurationError,
        InvalidVectorStoreConfigurationError,
        InvalidRetrievalConfigurationError,
        InvalidRerankingConfigurationError,
        InvalidRetrievalEvaluationDatasetError,
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
    reranker = (
        create_local_reranker_service(runtime_config.local_reranker)
        if runtime_config.local_reranker is not None
        else None
    )

    with LocalVectorStore(runtime_config.vector_store) as vector_store:
        retriever = RetrieverService(
            embedding_service,
            vector_store,
            sparse_embedding_service,
        )

        def retrieve_for_evaluation(query: str) -> list[RetrievalResult]:
            """Run one query through the shared retriever and reranker."""
            results = retriever.retrieve(
                query,
                config=runtime_config.retrieval,
            )
            if reranker is None:
                return results
            if runtime_config.reranking is None:
                raise RuntimeError(
                    "Reranker service has no result-limit configuration."
                )
            return reranker.rerank(
                query,
                results,
                config=runtime_config.reranking,
            )

        report = evaluate_retrieval(
            dataset,
            retrieve_for_evaluation,
            top_k=args.top_k,
        )

    if args.output_format == "json":
        print(json.dumps(retrieval_evaluation_to_dict(report), indent=2))
    else:
        print(format_retrieval_evaluation_table(report))
    return 0


def run_evaluate_answer(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Score labeled answers while reusing retrieval and model resources.

    The generation model remains lazy until the first case reaches generation,
    preserving abstention-only behavior and avoiding unnecessary initialization.
    """
    from rag_pipeline.answer_evaluation import (
        answer_evaluation_to_dict,
        evaluate_answers,
        format_answer_evaluation_table,
        load_answer_evaluation_dataset,
    )
    from rag_pipeline.cli.config import (
        build_generation_configs,
        build_retrieval_runtime_config,
    )
    from rag_pipeline.embeddings import (
        InvalidEmbeddingConfigurationError,
        create_local_embedding_service,
    )
    from rag_pipeline.exceptions import (
        InvalidAnswerEvaluationDatasetError,
        InvalidGenerationConfigurationError,
        InvalidRerankingConfigurationError,
        InvalidRetrievalConfigurationError,
        InvalidVectorStoreConfigurationError,
    )
    from rag_pipeline.generation import create_local_answer_generator
    from rag_pipeline.reranking import create_local_reranker_service
    from rag_pipeline.retrieval import RetrieverService
    from rag_pipeline.sparse_embeddings import (
        create_local_sparse_embedding_service,
    )
    from rag_pipeline.vector_store import LocalVectorStore

    try:
        dataset = load_answer_evaluation_dataset(args.dataset)
        runtime_config = build_retrieval_runtime_config(args)
        local_generation_config, generation_config = build_generation_configs(
            args
        )
    except (
        InvalidAnswerEvaluationDatasetError,
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
    reranker = (
        create_local_reranker_service(runtime_config.local_reranker)
        if runtime_config.local_reranker is not None
        else None
    )
    answer_generator = None

    with LocalVectorStore(runtime_config.vector_store) as vector_store:
        retriever = RetrieverService(
            embedding_service,
            vector_store,
            sparse_embedding_service,
        )

        def generate_for_evaluation(query: str) -> GeneratedAnswer:
            """Run one case through shared retrieval and generation services."""
            nonlocal answer_generator
            results = retriever.retrieve(
                query,
                config=runtime_config.retrieval,
            )
            if reranker is not None:
                if runtime_config.reranking is None:
                    raise RuntimeError(
                        "Reranker service has no result-limit configuration."
                    )
                results = reranker.rerank(
                    query,
                    results,
                    config=runtime_config.reranking,
                )
            if answer_generator is None:
                answer_generator = create_local_answer_generator(
                    local_generation_config
                )
            return answer_generator.generate(
                query,
                results,
                config=generation_config,
            )

        report = evaluate_answers(dataset, generate_for_evaluation)

    if args.output_format == "json":
        print(json.dumps(answer_evaluation_to_dict(report), indent=2))
    else:
        print(format_answer_evaluation_table(report))
    return 0
