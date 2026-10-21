"""Expose acknowledged, bounded Slack and Teams channel imports in the CLI."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import AbstractContextManager

from rag_pipeline.application.channel_messages import ChannelReader
from rag_pipeline.cli.options import (
    add_chunking_arguments,
    add_embedding_arguments,
    add_hybrid_search_arguments,
    add_vector_store_location_arguments,
)


def register_channel_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register manual message imports using existing local indexing settings."""
    slack = subparsers.add_parser(
        "index-slack",
        help="Import one Slack channel and its threads into local Qdrant.",
        description=(
            "Read one bot-visible Slack channel and its replies. Reimport replaces "
            "listed messages, but remote deletions and Slack ACLs are not "
            "applied to local retrieval. Requires RAG_SLACK_BOT_TOKEN."
        ),
    )
    slack.add_argument(
        "channel_id",
        help=(
            "Slack conversation ID to read, not its display name. The bot must "
            "be a member with a matching history scope; this ID scopes import."
        ),
    )
    _add_shared_options(slack, collection_name="slack_messages")
    slack.set_defaults(_handler=run_index_slack)

    teams = subparsers.add_parser(
        "index-teams",
        help="Import one Teams channel and its replies into local Qdrant.",
        description=(
            "Read one Teams channel with app-only Microsoft Graph access. "
            "Reimport replaces listed messages, but remote deletions and Teams "
            "ACLs are not applied to local retrieval. Requires RAG_TEAMS_* credentials."
        ),
    )
    teams.add_argument(
        "team_id",
        help=(
            "Teams group UUID containing the channel. It scopes Graph requests "
            "and stable citation IDs; use the team's actual ID, not its name."
        ),
    )
    teams.add_argument(
        "channel_id",
        help=(
            "Teams channel ID to read, including its thread suffix when present. "
            "The Entra app needs channel-message read consent for this team."
        ),
    )
    _add_shared_options(teams, collection_name="teams_messages")
    teams.set_defaults(_handler=run_index_teams)


def _add_shared_options(
    parser: argparse.ArgumentParser, *, collection_name: str
) -> None:
    parser.add_argument(
        "--acknowledge-no-acl",
        action="store_true",
        help=(
            "Required confirmation that imported message permissions are not "
            "enforced on local retrieval. Use only a trusted collection; this "
            "flag does not grant or check user access."
        ),
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=100,
        help=(
            "Maximum root and reply messages accepted in one full traversal. "
            "Raise for larger channels at the cost of more API calls and RAM "
            "(default: 100)."
        ),
    )
    parser.add_argument(
        "--max-message-kib",
        type=int,
        default=64,
        help=(
            "Maximum raw UTF-8 content per message in KiB. Lower it to cap "
            "memory; an oversized message aborts the import (default: 64)."
        ),
    )
    parser.add_argument(
        "--max-total-mib",
        type=int,
        default=2,
        help=(
            "Maximum combined raw message text in MiB. The completed traversal "
            "stays in RAM before indexing (default: 2)."
        ),
    )
    add_chunking_arguments(parser)
    add_embedding_arguments(parser)
    add_vector_store_location_arguments(parser, default_collection_name=collection_name)
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


def run_index_slack(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Read a Slack channel, then replace its listed local message sources."""
    from rag_pipeline.infrastructure.slack import SlackChannelClient

    return _run_channel_import(
        args,
        parser,
        platform="slack",
        reader_factory=lambda: SlackChannelClient.from_environment(args.channel_id),
    )


def run_index_teams(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Read a Teams channel, then replace its listed local message sources."""
    from rag_pipeline.infrastructure.teams import TeamsChannelClient

    return _run_channel_import(
        args,
        parser,
        platform="teams",
        reader_factory=lambda: TeamsChannelClient.from_environment(
            args.team_id, args.channel_id
        ),
    )


def _run_channel_import(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    platform: str,
    reader_factory: Callable[[], AbstractContextManager[ChannelReader]],
) -> int:
    """Validate risk and import limits before provider I/O or local writes."""
    from rag_pipeline.application.channel_messages import index_channel_messages
    from rag_pipeline.cli.config import build_index_command_config
    from rag_pipeline.exceptions import (
        ChannelConfigurationError,
        ChannelInputError,
        ChannelProviderError,
        InvalidChunkingConfigurationError,
        InvalidEmbeddingConfigurationError,
        InvalidPipelineConfigurationError,
        InvalidVectorStoreConfigurationError,
    )
    from rag_pipeline.ingestion.channel_messages import ChannelImportLimits

    if not args.acknowledge_no_acl:
        parser.error(
            f"index-{platform} requires --acknowledge-no-acl because channel "
            "permissions are not enforced on retrieval."
        )
    try:
        config = build_index_command_config(args)
        limits = ChannelImportLimits(
            max_messages=args.max_messages,
            max_message_bytes=args.max_message_kib * 1024,
            max_total_bytes=args.max_total_mib * 1024 * 1024,
        )
        with reader_factory() as reader:
            result = index_channel_messages(
                reader,
                platform=platform,
                channel_id=args.channel_id,
                config=config,
                limits=limits,
            )
    except (
        ChannelConfigurationError,
        ChannelInputError,
        ChannelProviderError,
        InvalidChunkingConfigurationError,
        InvalidEmbeddingConfigurationError,
        InvalidPipelineConfigurationError,
        InvalidVectorStoreConfigurationError,
    ) as exc:
        parser.error(str(exc))

    print(
        f"Indexed {result.indexed_count} {platform.title()} chunk(s) into "
        f"{result.collection_name!r}; collection now contains "
        f"{result.total_count} point(s)."
    )
    if result.removed_count:
        print(f"Removed {result.removed_count} obsolete point(s).")
    return 0
