"""Check channel message staging, provenance, and CLI risk gates offline."""

from __future__ import annotations

import io
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

from rag_pipeline.application.channel_messages import index_channel_messages
from rag_pipeline.cli import build_parser, main
from rag_pipeline.exceptions import ChannelInputError
from rag_pipeline.ingestion.channel_messages import ChannelImportLimits, ChannelMessage


class FakeReader:
    """Return a fixed message snapshot without a provider or database."""

    def __init__(self, messages: list[ChannelMessage]) -> None:
        self.messages = messages

    def list_messages(self, _limits: ChannelImportLimits) -> list[ChannelMessage]:
        return self.messages


def _message(
    message_id: str, content: str = "Policy text", *, content_type: str = "text"
) -> ChannelMessage:
    return ChannelMessage(
        source=f"teams://team/channel/{message_id}",
        platform="teams",
        channel_id="channel",
        message_id=message_id,
        content=content,
        content_type=content_type,
    )


class ChannelIndexingTests(unittest.TestCase):
    """Ensure all message validation precedes model inference and writes."""

    def test_html_is_normalized_and_sources_are_preserved(self) -> None:
        reader = FakeReader(
            [
                _message(
                    "1",
                    "<p>Expense policy</p><script>secret</script>",
                    content_type="html",
                )
            ]
        )
        with patch(
            "rag_pipeline.application.channel_messages.index_extracted_documents"
        ) as index:
            index_channel_messages(
                reader,
                platform="teams",
                channel_id="channel",
                config=Mock(),
                limits=ChannelImportLimits(),
            )
        document = index.call_args.args[0][0]
        self.assertEqual(document.page_content, "Expense policy")
        self.assertEqual(document.metadata["source"], "teams://team/channel/1")
        self.assertEqual(index.call_args.kwargs["sources"], {"teams://team/channel/1"})

    def test_invalid_later_message_never_starts_indexer(self) -> None:
        cases = (
            ([_message("1"), _message("1")], ChannelImportLimits()),
            (
                [_message("1"), _message("2", "long")],
                ChannelImportLimits(max_message_bytes=4),
            ),
            ([_message("1"), _message("2")], ChannelImportLimits(max_messages=1)),
        )
        for messages, limits in cases:
            with (
                self.subTest(messages=messages, limits=limits),
                patch(
                    "rag_pipeline.application.channel_messages.index_extracted_documents"
                ) as index,
            ):
                with self.assertRaises(ChannelInputError):
                    index_channel_messages(
                        FakeReader(messages),
                        platform="teams",
                        channel_id="channel",
                        config=Mock(),
                        limits=limits,
                    )
                index.assert_not_called()

    def test_empty_snapshot_fails_without_writes(self) -> None:
        with patch(
            "rag_pipeline.application.channel_messages.index_extracted_documents"
        ) as index:
            with self.assertRaises(ChannelInputError):
                index_channel_messages(
                    FakeReader([]),
                    platform="teams",
                    channel_id="channel",
                    config=Mock(),
                    limits=ChannelImportLimits(),
                )
            index.assert_not_called()

    def test_blank_edited_message_replaces_its_previous_source(self) -> None:
        with patch(
            "rag_pipeline.application.channel_messages.index_extracted_documents"
        ) as index:
            index_channel_messages(
                FakeReader([_message("1", "   ")]),
                platform="teams",
                channel_id="channel",
                config=Mock(),
                limits=ChannelImportLimits(),
            )
        self.assertEqual(index.call_args.args[0], [])
        self.assertEqual(index.call_args.kwargs["sources"], {"teams://team/channel/1"})

    def test_malformed_unicode_aborts_before_indexing(self) -> None:
        with patch(
            "rag_pipeline.application.channel_messages.index_extracted_documents"
        ) as index:
            with self.assertRaisesRegex(ChannelInputError, "encoding"):
                index_channel_messages(
                    FakeReader([_message("1", "\ud800")]),
                    platform="teams",
                    channel_id="channel",
                    config=Mock(),
                    limits=ChannelImportLimits(),
                )
            index.assert_not_called()


class ChannelCliTests(unittest.TestCase):
    """Verify separate defaults and explicit acknowledgement for both adapters."""

    def test_help_defaults_and_risk_gate(self) -> None:
        for command, positional, collection in (
            ("index-slack", ["C123"], "slack_messages"),
            (
                "index-teams",
                ["fbe2bf47-16c8-47cf-b4a5-4b9b187c508b", "19:abc@thread.tacv2"],
                "teams_messages",
            ),
        ):
            with self.subTest(command=command):
                args = build_parser().parse_args(
                    [command, *positional, "--acknowledge-no-acl"]
                )
                self.assertEqual(args.collection_name, collection)
                self.assertEqual(args.max_messages, 100)
                error = io.StringIO()
                with redirect_stderr(error), self.assertRaises(SystemExit):
                    main([command, *positional])
                self.assertIn("--acknowledge-no-acl", error.getvalue())

    def test_invalid_limit_precedes_credentials(self) -> None:
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit):
            main(["index-slack", "C123", "--acknowledge-no-acl", "--max-messages", "0"])
        self.assertIn("max_messages", error.getvalue())

    def test_success_uses_separate_provider_and_collection(self) -> None:
        result = Mock(
            indexed_count=2,
            collection_name="slack_messages",
            total_count=2,
            removed_count=0,
        )
        output = io.StringIO()
        with (
            patch(
                "rag_pipeline.infrastructure.slack.SlackChannelClient.from_environment",
                return_value=nullcontext(object()),
            ) as create,
            patch(
                "rag_pipeline.application.channel_messages.index_channel_messages",
                return_value=result,
            ) as index,
            redirect_stdout(output),
        ):
            self.assertEqual(main(["index-slack", "C123", "--acknowledge-no-acl"]), 0)
        create.assert_called_once_with("C123")
        self.assertEqual(
            index.call_args.kwargs["config"].vector_store.collection_name,
            "slack_messages",
        )
        self.assertIn("Indexed 2 Slack chunk(s)", output.getvalue())

    def test_teams_success_passes_team_and_channel(self) -> None:
        team_id = "fbe2bf47-16c8-47cf-b4a5-4b9b187c508b"
        channel_id = "19:abc@thread.tacv2"
        result = Mock(
            indexed_count=1,
            collection_name="teams_messages",
            total_count=1,
            removed_count=0,
        )
        with (
            patch(
                "rag_pipeline.infrastructure.teams.TeamsChannelClient.from_environment",
                return_value=nullcontext(object()),
            ) as create,
            patch(
                "rag_pipeline.application.channel_messages.index_channel_messages",
                return_value=result,
            ) as index,
        ):
            self.assertEqual(
                main(["index-teams", team_id, channel_id, "--acknowledge-no-acl"]), 0
            )
        create.assert_called_once_with(team_id, channel_id)
        self.assertEqual(
            index.call_args.kwargs["config"].vector_store.collection_name,
            "teams_messages",
        )
