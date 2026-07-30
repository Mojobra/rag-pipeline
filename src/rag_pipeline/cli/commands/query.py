"""Register and execute interactive retrieval and answer commands."""

from __future__ import annotations

import argparse

from rag_pipeline.cli.options import (
    DEFAULT_ANSWER_SCORE_THRESHOLD,
    add_embedding_arguments,
    add_generation_arguments,
    add_hybrid_search_arguments,
    add_reranking_arguments,
    add_retrieval_arguments,
    add_vector_store_location_arguments,
)


def register_retrieve_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
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
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
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
    from rag_pipeline.application.retrieval import (
        open_local_retrieval_pipeline,
    )
    from rag_pipeline.cli.config import build_retrieval_runtime_config
    from rag_pipeline.cli.output import format_retrieval_results
    from rag_pipeline.exceptions import (
        InvalidEmbeddingConfigurationError,
        InvalidPipelineConfigurationError,
        InvalidRerankingConfigurationError,
        InvalidRetrievalConfigurationError,
        InvalidVectorStoreConfigurationError,
    )

    try:
        runtime_config = build_retrieval_runtime_config(args)
    except (
        InvalidEmbeddingConfigurationError,
        InvalidPipelineConfigurationError,
        InvalidVectorStoreConfigurationError,
        InvalidRetrievalConfigurationError,
        InvalidRerankingConfigurationError,
    ) as exc:
        parser.error(str(exc))

    with open_local_retrieval_pipeline(runtime_config) as pipeline:
        results = pipeline.retrieve(args.query)

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
    from rag_pipeline.application.retrieval import (
        open_local_retrieval_pipeline,
    )
    from rag_pipeline.cli.config import (
        build_generation_configs,
        build_retrieval_runtime_config,
    )
    from rag_pipeline.cli.output import (
        format_abstention_answer,
        format_generated_answer,
    )
    from rag_pipeline.exceptions import (
        InvalidEmbeddingConfigurationError,
        InvalidGenerationConfigurationError,
        InvalidPipelineConfigurationError,
        InvalidRerankingConfigurationError,
        InvalidRetrievalConfigurationError,
        InvalidVectorStoreConfigurationError,
    )
    from rag_pipeline.generation import create_local_answer_generator
    from rag_pipeline.prompting import INSUFFICIENT_CONTEXT_ANSWER

    try:
        runtime_config = build_retrieval_runtime_config(args)
        local_generation_config, generation_config = build_generation_configs(args)
    except (
        InvalidEmbeddingConfigurationError,
        InvalidPipelineConfigurationError,
        InvalidVectorStoreConfigurationError,
        InvalidRetrievalConfigurationError,
        InvalidRerankingConfigurationError,
        InvalidGenerationConfigurationError,
    ) as exc:
        parser.error(str(exc))

    with open_local_retrieval_pipeline(runtime_config) as pipeline:
        retrieval_results = pipeline.retrieve(args.query)

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
