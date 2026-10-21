"""Expose bounded Google Drive folder indexing with explicit ACL acknowledgement."""

from __future__ import annotations

import argparse

from rag_pipeline.cli.options import (
    add_chunking_arguments,
    add_embedding_arguments,
    add_hybrid_search_arguments,
    add_vector_store_location_arguments,
)


def register_google_drive_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register one manual Drive import using the local indexing settings."""
    parser = subparsers.add_parser(
        "index-google-drive",
        help="Import supported files and Google Docs from one Drive folder.",
        description=(
            "Read one Google Drive folder recursively using application default "
            "credentials, then replace listed sources in local Qdrant. Missing "
            "remote files are not deleted and Drive ACLs are not enforced on "
            "retrieval. This command never writes to Google Drive."
        ),
    )
    parser.add_argument(
        "folder_id",
        help=(
            "Google Drive folder ID to scan recursively. The authenticated "
            "identity must be able to list and download its files; the ID "
            "scopes this import, not later query permissions."
        ),
    )
    parser.add_argument(
        "--acknowledge-no-acl",
        action="store_true",
        help=(
            "Required confirmation that indexed Drive content has no per-user "
            "access control in this pipeline. Use a trusted collection; this "
            "flag does not add authorization."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=100,
        help=(
            "Maximum supported files accepted from the folder before indexing. "
            "Raise for larger trees at the cost of more API work and RAM "
            "(default: 100)."
        ),
    )
    parser.add_argument(
        "--max-file-mib",
        type=int,
        default=20,
        help=(
            "Maximum downloaded bytes per file in MiB; Google Docs exports "
            "also have a 10 MB provider cap. Larger values use more memory "
            "(default: 20)."
        ),
    )
    parser.add_argument(
        "--max-total-mib",
        type=int,
        default=100,
        help=(
            "Maximum combined downloaded bytes in MiB. The extracted snapshot "
            "stays in RAM until indexing (default: 100)."
        ),
    )
    add_chunking_arguments(parser)
    add_embedding_arguments(parser)
    add_vector_store_location_arguments(
        parser, default_collection_name="google_drive_documents"
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
    parser.set_defaults(_handler=run_index_google_drive)


def run_index_google_drive(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Validate risk and settings, then fetch and index one Drive folder.

    ADC may refresh credentials over the network; the workflow reads Drive,
    initializes local models, and mutates only the selected Qdrant sources.
    """
    from rag_pipeline.application.google_drive import index_google_drive_folder
    from rag_pipeline.cli.config import build_index_command_config
    from rag_pipeline.exceptions import (
        GoogleDriveConfigurationError,
        GoogleDriveInputError,
        GoogleDriveProviderError,
        InvalidChunkingConfigurationError,
        InvalidEmbeddingConfigurationError,
        InvalidPipelineConfigurationError,
        InvalidVectorStoreConfigurationError,
    )
    from rag_pipeline.infrastructure.google_drive import (
        GoogleDriveImportLimits,
        create_google_drive_client,
    )

    if not args.acknowledge_no_acl:
        parser.error(
            "index-google-drive requires --acknowledge-no-acl because Drive "
            "permissions are not enforced on retrieval."
        )
    try:
        config = build_index_command_config(args)
        limits = GoogleDriveImportLimits(
            max_files=args.max_files,
            max_file_bytes=args.max_file_mib * 1024 * 1024,
            max_total_bytes=args.max_total_mib * 1024 * 1024,
        )
        with create_google_drive_client(args.folder_id) as reader:
            result = index_google_drive_folder(reader, config=config, limits=limits)
    except (
        GoogleDriveConfigurationError,
        GoogleDriveInputError,
        GoogleDriveProviderError,
        InvalidChunkingConfigurationError,
        InvalidEmbeddingConfigurationError,
        InvalidPipelineConfigurationError,
        InvalidVectorStoreConfigurationError,
    ) as exc:
        parser.error(str(exc))

    print(
        f"Indexed {result.indexed_count} Google Drive chunk(s) into "
        f"{result.collection_name!r}; collection now contains "
        f"{result.total_count} point(s)."
    )
    if result.removed_count:
        print(f"Removed {result.removed_count} obsolete point(s).")
    return 0
