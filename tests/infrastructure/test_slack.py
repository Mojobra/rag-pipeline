"""Exercise Slack channel pagination, thread coverage, and sanitized failures."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from rag_pipeline.exceptions import (
    ChannelConfigurationError,
    ChannelInputError,
    ChannelProviderError,
)
from rag_pipeline.infrastructure.slack import SlackChannelClient
from rag_pipeline.ingestion.channel_messages import ChannelImportLimits


class SlackChannelClientTests(unittest.TestCase):
    """Verify read-only channel traversal against deterministic API responses."""

    def test_history_and_replies_are_scoped_and_paginated(self) -> None:
        requests: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            self.assertEqual(request.headers["Authorization"], "Bearer secret")
            if request.url.path.endswith("/auth.test"):
                return httpx.Response(200, json={"ok": True, "team_id": "T123"})
            if request.url.path.endswith("/conversations.replies"):
                self.assertEqual(request.url.params["ts"], "123.001")
                return httpx.Response(
                    200,
                    json={
                        "ok": True,
                        "messages": [
                            {"type": "message", "ts": "123.001", "text": "Parent"},
                            {
                                "type": "message",
                                "ts": "124.001",
                                "thread_ts": "123.001",
                                "text": "Reply",
                            },
                        ],
                    },
                )
            if request.url.params.get("cursor") == "next":
                return httpx.Response(
                    200,
                    json={
                        "ok": True,
                        "messages": [
                            {
                                "type": "message",
                                "subtype": "bot_message",
                                "ts": "122.001",
                                "text": "Older",
                            },
                        ],
                    },
                )
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "messages": [
                        {
                            "type": "message",
                            "ts": "123.001",
                            "text": "Parent",
                            "reply_count": 1,
                        }
                    ],
                    "has_more": True,
                    "response_metadata": {"next_cursor": "next"},
                },
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            reader = SlackChannelClient("C123", token="secret", http_client=client)
            messages = reader.list_messages(ChannelImportLimits())

        self.assertEqual(len(requests), 4)
        self.assertEqual(
            {message.source for message in messages},
            {
                "slack://T123/C123/122.001",
                "slack://T123/C123/123.001",
                "slack://T123/C123/124.001",
            },
        )
        self.assertEqual(
            next(m.parent_id for m in messages if m.message_id == "124.001"), "123.001"
        )
        self.assertTrue(all(request.method == "GET" for request in requests))

    def test_rejects_incomplete_or_repeated_cursor(self) -> None:
        for cursor in (None, "same"):
            with self.subTest(cursor=cursor):

                def respond(
                    request: httpx.Request, *, next_cursor: str | None = cursor
                ) -> httpx.Response:
                    if request.url.path.endswith("/auth.test"):
                        return httpx.Response(200, json={"ok": True, "team_id": "T123"})
                    return httpx.Response(
                        200,
                        json={
                            "ok": True,
                            "messages": [],
                            "has_more": True,
                            "response_metadata": {"next_cursor": next_cursor},
                        },
                    )

                with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                    reader = SlackChannelClient(
                        "C123", token="secret", http_client=client
                    )
                    with self.assertRaises(ChannelInputError):
                        reader.list_messages(ChannelImportLimits())

    def test_rejects_limits_and_foreign_reply(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/auth.test"):
                return httpx.Response(200, json={"ok": True, "team_id": "T123"})
            if request.url.path.endswith("/conversations.replies"):
                return httpx.Response(
                    200,
                    json={
                        "ok": True,
                        "messages": [
                            {
                                "type": "message",
                                "ts": "124.001",
                                "thread_ts": "other",
                                "text": "Reply",
                            }
                        ],
                    },
                )
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "messages": [
                        {
                            "type": "message",
                            "ts": "123.001",
                            "text": "Parent",
                            "reply_count": 1,
                        }
                    ],
                },
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            reader = SlackChannelClient("C123", token="secret", http_client=client)
            with self.assertRaisesRegex(ChannelInputError, "max_message_bytes"):
                reader.list_messages(ChannelImportLimits(max_message_bytes=3))
            with self.assertRaisesRegex(ChannelInputError, "outside its thread"):
                reader.list_messages(ChannelImportLimits())

    def test_provider_errors_do_not_expose_token(self) -> None:
        def respond(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": False, "error": "secret"})

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            reader = SlackChannelClient("C123", token="secret", http_client=client)
            with self.assertRaises(ChannelProviderError) as raised:
                reader.list_messages(ChannelImportLimits())
        self.assertNotIn("secret", str(raised.exception))

    def test_invalid_configuration(self) -> None:
        with self.assertRaises(ChannelConfigurationError):
            SlackChannelClient("bad/id", token="secret")
        with self.assertRaises(ChannelConfigurationError):
            SlackChannelClient("C123", token="")
        with patch.dict("os.environ", {"RAG_SLACK_BOT_TOKEN": ""}):
            with self.assertRaises(ChannelConfigurationError):
                SlackChannelClient.from_environment("C123")
