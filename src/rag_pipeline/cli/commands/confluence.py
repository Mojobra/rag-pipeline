"""Expose bounded, explicitly acknowledged Confluence Cloud page indexing."""

from __future__ import annotations

import argparse

from rag_pipeline.cli.options import (
    add_chunking_arguments,
    add_embedding_arguments,
    add_hybrid_search_arguments,
    add_vector_store_location_arguments,
)


def register_confluence_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register one manual space import using the local indexing settings."""
    parser = subparsers.add_parser(
        "index-confluence",
        help="Import current pages from one Confluence Cloud space.",
        description=(
            "Read current pages from one Confluence Cloud space, then replace "
            "their sources in local Qdrant. Missing remote pages are not deleted "
            "and Confluence ACLs are not enforced on retrieval. Credentials "
            "come only from environment variables."
        ),
    )
    parser.add_argument(
        "space_id",
        help=(
            "Numeric Confluence space ID to scan. The API account must have "
            "view access; this ID scopes listing but not query-time permissions."
        ),
    )
    parser.add_argument(
        "--acknowledge-no-acl",
        action="store_true",
        help=(
            "Required confirmation that indexed Confluence pages have no "
            "per-user access control in this pipeline. Use a trusted collection; "
            "this flag does not add authorization."
        ),
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help=(
            "Maximum current pages accepted before indexing. Raise for larger "
            "spaces at the cost of more API work and RAM (default: 100)."
        ),
    )
    parser.add_argument(
        "--max-page-mib",
        type=int,
        default=2,
        help=(
            "Maximum UTF-8 storage-HTML size per page in MiB. Larger pages "
            "use more memory and may need more processing (default: 2)."
        ),
    )
    parser.add_argument(
        "--max-total-mib",
        type=int,
        default=20,
        help=(
            "Maximum combined storage-HTML size in MiB. The extracted "
            "snapshot remains in RAM until indexing (default: 20)."
        ),
    )
    add_chunking_arguments(parser)
    add_embedding_arguments(parser)
    add_vector_store_location_arguments(
        parser, default_collection_name="confluence_documents"
    )
    add_hybrid_search_arguments(parser)
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=64,
        help=(
            "Maximum points per synchronous Qdrant upsert or stale-point "
            "deletion; also sets source-scan page size. Larger batches reduce "
            "calls but use more memory (default: 64)."
        ),
    )
    parser.set_defaults(_handler=run_index_confluence)


def run_index_confluence(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Validate risk and settings, then fetch and index one cloud space.

    This reads an API token from the environment, sends Confluence requests,
    may initialize local models, and mutates only the selected Qdrant sources.
    """
    from rag_pipeline.application.confluence import index_confluence_space
    from rag_pipeline.cli.config import build_index_command_config
    from rag_pipeline.exceptions import (
        ConfluenceConfigurationError,
        ConfluenceInputError,
        ConfluenceProviderError,
        InvalidChunkingConfigurationError,
        InvalidEmbeddingConfigurationError,
        InvalidPipelineConfigurationError,
        InvalidVectorStoreConfigurationError,
    )
    from rag_pipeline.infrastructure.confluence import (
        ConfluenceCloudClient,
        ConfluenceImportLimits,
    )

    if not args.acknowledge_no_acl:
        parser.error(
            "index-confluence requires --acknowledge-no-acl because Confluence "
            "permissions are not enforced on retrieval."
        )
    try:
        config = build_index_command_config(args)
        limits = ConfluenceImportLimits(
            max_pages=args.max_pages,
            max_page_bytes=args.max_page_mib * 1024 * 1024,
            max_total_bytes=args.max_total_mib * 1024 * 1024,
        )
        with ConfluenceCloudClient.from_environment() as reader:
            result = index_confluence_space(
                reader, space_id=args.space_id, config=config, limits=limits
            )
    except (
        ConfluenceConfigurationError,
        ConfluenceInputError,
        ConfluenceProviderError,
        InvalidChunkingConfigurationError,
        InvalidEmbeddingConfigurationError,
        InvalidPipelineConfigurationError,
        InvalidVectorStoreConfigurationError,
    ) as exc:
        parser.error(str(exc))

    print(
        f"Indexed {result.indexed_count} Confluence chunk(s) into "
        f"{result.collection_name!r}; collection now contains "
        f"{result.total_count} point(s)."
    )
    if result.removed_count:
        print(f"Removed {result.removed_count} obsolete point(s).")
    return 0
