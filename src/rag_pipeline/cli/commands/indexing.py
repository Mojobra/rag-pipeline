"""Register and execute embedding preview and Qdrant indexing commands."""

from __future__ import annotations

import argparse

from rag_pipeline.cli.options import (
    add_chunking_arguments,
    add_document_input_arguments,
    add_embedding_arguments,
    add_hybrid_search_arguments,
    add_vector_store_location_arguments,
)


def register_indexing_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register embedding diagnostics and persistent indexing handlers."""
    embed_parser = subparsers.add_parser(
        "embed",
        help="Inspect local dense embedding output without indexing.",
        description=(
            "Extract, chunk, and embed local documents, then report vector count "
            "and dimension. Models may be downloaded, but no Qdrant data is written."
        ),
    )
    add_document_input_arguments(embed_parser)
    add_chunking_arguments(embed_parser)
    add_embedding_arguments(embed_parser)
    embed_parser.set_defaults(_handler=run_embed)

    index_parser = subparsers.add_parser(
        "index",
        help="Build or update a persistent local Qdrant collection.",
        description=(
            "Extract, chunk, and embed documents, then upsert deterministic points "
            "into local Qdrant. Model and search-mode settings become part of the "
            "collection compatibility contract."
        ),
    )
    add_document_input_arguments(index_parser)
    add_chunking_arguments(index_parser)
    add_embedding_arguments(index_parser)
    add_vector_store_location_arguments(index_parser)
    add_hybrid_search_arguments(index_parser)
    index_parser.add_argument(
        "--write-batch-size",
        type=int,
        default=64,
        help=(
            "Number of chunk vectors sent in each synchronous Qdrant upsert. "
            "Larger batches reduce write calls but use more memory; reduce after "
            "memory-related write failures (default: 64)."
        ),
    )
    index_parser.set_defaults(_handler=run_index)


def run_embed(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Embed extracted chunks and report the provider's vector dimension.

    The command may initialize or download the selected model, but it performs
    no vector-store writes.
    """
    from rag_pipeline.application.indexing import preview_local_embeddings
    from rag_pipeline.cli.config import (
        build_chunking_config,
        build_embedding_config,
    )
    from rag_pipeline.exceptions import (
        InvalidChunkingConfigurationError,
        InvalidEmbeddingConfigurationError,
    )

    try:
        chunking_config = build_chunking_config(args)
        embedding_config = build_embedding_config(args)
    except (
        InvalidChunkingConfigurationError,
        InvalidEmbeddingConfigurationError,
    ) as exc:
        parser.error(str(exc))

    preview = preview_local_embeddings(
        args.paths,
        recursive=args.recursive,
        chunking=chunking_config,
        embedding=embedding_config,
    )

    if preview.dimension is None:
        print("Embedded 0 chunk(s); no vectors were created.")
        return 0

    print(
        f"Embedded {preview.chunk_count} chunk(s) into "
        f"{preview.dimension}-dimensional vectors using "
        f"{preview.model_identifier}."
    )
    return 0


def run_index(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Build dense or hybrid vectors and upsert them into local Qdrant.

    The handler performs document and model I/O and mutates the selected
    collection. Configuration validation occurs before those side effects.
    """
    from rag_pipeline.application.indexing import index_local_documents
    from rag_pipeline.cli.config import build_index_command_config
    from rag_pipeline.exceptions import (
        InvalidChunkingConfigurationError,
        InvalidEmbeddingConfigurationError,
        InvalidPipelineConfigurationError,
        InvalidVectorStoreConfigurationError,
    )

    try:
        config = build_index_command_config(args)
    except (
        InvalidChunkingConfigurationError,
        InvalidEmbeddingConfigurationError,
        InvalidPipelineConfigurationError,
        InvalidVectorStoreConfigurationError,
    ) as exc:
        parser.error(str(exc))

    result = index_local_documents(
        args.paths,
        recursive=args.recursive,
        config=config,
    )

    print(
        f"Indexed {result.indexed_count} chunk(s) into "
        f"{result.collection_name!r}; collection now contains "
        f"{result.total_count} point(s)."
    )
    return 0
