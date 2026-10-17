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
        help="Replace indexed chunks for supplied sources in local Qdrant.",
        description=(
            "Extract, chunk, and embed documents, then replace all indexed chunks "
            "for each supplied source in local Qdrant. Other sources remain "
            "untouched unless --prune-missing is selected. Model and search-mode "
            "settings must match the collection contract."
        ),
    )
    add_document_input_arguments(index_parser)
    add_chunking_arguments(index_parser)
    add_embedding_arguments(index_parser)
    add_vector_store_location_arguments(index_parser)
    add_hybrid_search_arguments(index_parser)
    index_parser.add_argument(
        "--prune-missing",
        action="store_true",
        help=(
            "Also remove indexed sources no longer found under the supplied "
            "directory. Requires exactly one directory and recursive scanning; "
            "review the scope because points for missing files are deleted "
            "from the collection (not the source files)."
        ),
    )
    index_parser.add_argument(
        "--write-batch-size",
        type=int,
        default=64,
        help=(
            "Maximum points per synchronous Qdrant upsert or stale-point "
            "deletion; also sets source-scan page size. Larger batches reduce "
            "calls but use more memory (default: 64)."
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
    """Replace source chunks in local Qdrant and optionally prune a directory.

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

    try:
        result = index_local_documents(
            args.paths,
            recursive=args.recursive,
            config=config,
            prune_missing=args.prune_missing,
        )
    except InvalidPipelineConfigurationError as exc:
        parser.error(str(exc))

    print(
        f"Indexed {result.indexed_count} chunk(s) into "
        f"{result.collection_name!r}; collection now contains "
        f"{result.total_count} point(s)."
    )
    if result.removed_count:
        print(f"Removed {result.removed_count} obsolete point(s).")
    return 0
