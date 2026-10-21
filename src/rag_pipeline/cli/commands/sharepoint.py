"""Expose bounded, explicitly acknowledged SharePoint drive indexing in CLI."""

from __future__ import annotations

import argparse

from rag_pipeline.cli.options import (
    add_chunking_arguments,
    add_embedding_arguments,
    add_hybrid_search_arguments,
    add_vector_store_location_arguments,
)


def register_sharepoint_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register one manual Graph import that reuses local indexing settings."""
    parser = subparsers.add_parser(
        "index-sharepoint",
        help="Import supported files from one SharePoint document library.",
        description=(
            "Read one Microsoft Graph drive recursively, then replace its listed "
            "sources in a local Qdrant collection. This is a manual import: "
            "missing remote files are not deleted and SharePoint ACLs are not "
            "enforced on retrieval. Credentials come only from environment variables."
        ),
    )
    parser.add_argument(
        "drive_id",
        help=(
            "Microsoft Graph document-library drive ID to scan recursively. "
            "Choose a library your Entra app can read; the ID scopes source "
            "identity and every Graph request."
        ),
    )
    parser.add_argument(
        "--acknowledge-no-acl",
        action="store_true",
        help=(
            "Required confirmation that indexed SharePoint content has no "
            "per-user access control in this pipeline. Use only a trusted "
            "collection; this flag does not add authorization."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=100,
        help=(
            "Maximum supported files accepted from the drive before indexing. "
            "Raise for larger libraries at the cost of more Graph calls and RAM "
            "(default: 100)."
        ),
    )
    parser.add_argument(
        "--max-file-mib",
        type=int,
        default=20,
        help=(
            "Maximum download size per file in MiB; Graph metadata and actual "
            "bytes are checked. Larger files use more memory (default: 20)."
        ),
    )
    parser.add_argument(
        "--max-total-mib",
        type=int,
        default=100,
        help=(
            "Maximum sum of downloaded supported files in MiB. The whole "
            "extracted snapshot stays in memory until indexing; each file "
            "also uses temporary disk during extraction (default: 100)."
        ),
    )
    add_chunking_arguments(parser)
    add_embedding_arguments(parser)
    add_vector_store_location_arguments(
        parser, default_collection_name="sharepoint_documents"
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
    parser.set_defaults(_handler=run_index_sharepoint)


def run_index_sharepoint(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Validate risk and settings, then fetch and index one remote drive.

    This command reads Entra credentials from environment variables, sends
    Microsoft Graph requests, may initialize/download local models, and
    mutates only the selected local collection's listed sources.
    """
    from rag_pipeline.application.sharepoint import index_sharepoint_drive
    from rag_pipeline.cli.config import build_index_command_config
    from rag_pipeline.exceptions import (
        InvalidChunkingConfigurationError,
        InvalidEmbeddingConfigurationError,
        InvalidPipelineConfigurationError,
        InvalidVectorStoreConfigurationError,
        SharePointConfigurationError,
        SharePointInputError,
        SharePointProviderError,
    )
    from rag_pipeline.infrastructure.sharepoint import (
        SharePointImportLimits,
        create_sharepoint_client,
    )

    if not args.acknowledge_no_acl:
        parser.error(
            "index-sharepoint requires --acknowledge-no-acl because SharePoint "
            "permissions are not enforced on retrieval."
        )
    try:
        config = build_index_command_config(args)
        limits = SharePointImportLimits(
            max_files=args.max_files,
            max_file_bytes=args.max_file_mib * 1024 * 1024,
            max_total_bytes=args.max_total_mib * 1024 * 1024,
        )
        with create_sharepoint_client(args.drive_id) as reader:
            result = index_sharepoint_drive(reader, config=config, limits=limits)
    except (
        InvalidChunkingConfigurationError,
        InvalidEmbeddingConfigurationError,
        InvalidPipelineConfigurationError,
        InvalidVectorStoreConfigurationError,
        SharePointConfigurationError,
        SharePointInputError,
        SharePointProviderError,
    ) as exc:
        parser.error(str(exc))

    print(
        f"Indexed {result.indexed_count} SharePoint chunk(s) into "
        f"{result.collection_name!r}; collection now contains "
        f"{result.total_count} point(s)."
    )
    if result.removed_count:
        print(f"Removed {result.removed_count} obsolete point(s).")
    return 0
