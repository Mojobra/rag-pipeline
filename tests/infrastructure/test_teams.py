"""Exercise Graph channel and reply pagination without a live Entra tenant."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from rag_pipeline.exceptions import (
    ChannelConfigurationError,
    ChannelInputError,
    ChannelProviderError,
)
from rag_pipeline.infrastructure.teams import TeamsChannelClient
from rag_pipeline.ingestion.channel_messages import ChannelImportLimits

_TEAM = "fbe2bf47-16c8-47cf-b4a5-4b9b187c508b"
_CHANNEL = "19:abc@thread.tacv2"


def _message(message_id: str, *, reply_to: str | None = None) -> dict[str, object]:
    return {
        "id": message_id,
        "replyToId": reply_to,
        "messageType": "message",
        "body": {"contentType": "html", "content": f"<p>Message {message_id}</p>"},
        "webUrl": f"https://teams.microsoft.com/l/message/{message_id}",
    }


class TeamsChannelClientTests(unittest.TestCase):
    """Check Graph scope, reply completeness, and fail-closed parsing."""

    def test_root_and_reply_continuations(self) -> None:
        requests: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            self.assertEqual(request.headers["Authorization"], "Bearer token")
            if request.url.path.endswith("/1/replies"):
                return httpx.Response(
                    200, json={"value": [_message("3", reply_to="1")]}
                )
            if request.url.params.get("$skiptoken") == "next":
                return httpx.Response(
                    200, json={"value": [{**_message("4"), "replies": []}]}
                )
            return httpx.Response(
                200,
                json={
                    "value": [
                        {
                            **_message("1"),
                            "replies": [_message("2", reply_to="1")],
                            "replies@odata.nextLink": f"https://graph.microsoft.com/v1.0/teams/{_TEAM}/channels/{_CHANNEL}/messages/1/replies?$skiptoken=reply",
                        }
                    ],
                    "@odata.nextLink": f"https://graph.microsoft.com/v1.0/teams/{_TEAM}/channels/{_CHANNEL}/messages?$skiptoken=next",
                },
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            reader = TeamsChannelClient(
                _TEAM, _CHANNEL, token_provider=lambda: "token", http_client=client
            )
            messages = reader.list_messages(ChannelImportLimits())

        self.assertEqual(len(requests), 3)
        self.assertEqual(
            {message.message_id for message in messages}, {"1", "2", "3", "4"}
        )
        self.assertEqual(
            next(m.parent_id for m in messages if m.message_id == "3"), "1"
        )
        self.assertEqual(
            next(m.source for m in messages if m.message_id == "3"),
            f"teams://{_TEAM}/19%3Aabc%40thread.tacv2/1/replies/3",
        )
        self.assertTrue(all(request.method == "GET" for request in requests))

    def test_system_event_without_replies_is_skipped(self) -> None:
        def respond(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "value": [
                        {"id": "1", "messageType": "systemEventMessage"},
                        {**_message("2"), "replies": []},
                    ]
                },
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            reader = TeamsChannelClient(
                _TEAM, _CHANNEL, token_provider=lambda: "token", http_client=client
            )
            messages = reader.list_messages(ChannelImportLimits())
        self.assertEqual([message.message_id for message in messages], ["2"])

    def test_equal_reply_ids_in_different_threads_keep_distinct_sources(self) -> None:
        def respond(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "value": [
                        {**_message("1"), "replies": [_message("9", reply_to="1")]},
                        {**_message("2"), "replies": [_message("9", reply_to="2")]},
                    ]
                },
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            reader = TeamsChannelClient(
                _TEAM, _CHANNEL, token_provider=lambda: "token", http_client=client
            )
            messages = reader.list_messages(ChannelImportLimits())
        self.assertEqual(len(messages), 4)
        self.assertEqual(len({message.source for message in messages}), 4)

    def test_rejects_cross_channel_and_repeated_links(self) -> None:
        for link in (
            "https://attacker.example/v1.0/teams/other/channels/x/messages",
            f"https://graph.microsoft.com/v1.0/teams/{_TEAM}/channels/other/messages",
            f"https://graph.microsoft.com/v1.0/teams/{_TEAM}/channels/{_CHANNEL}/messages?$top=50&$expand=replies",
        ):
            with self.subTest(link=link):

                def respond(
                    _request: httpx.Request, *, next_link: str = link
                ) -> httpx.Response:
                    return httpx.Response(
                        200, json={"value": [], "@odata.nextLink": next_link}
                    )

                with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                    reader = TeamsChannelClient(
                        _TEAM,
                        _CHANNEL,
                        token_provider=lambda: "token",
                        http_client=client,
                    )
                    with self.assertRaises(ChannelInputError):
                        reader.list_messages(ChannelImportLimits())

    def test_rejects_missing_replies_and_limit_overflow(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            if request.url.params.get("$top") == "50":
                return httpx.Response(
                    200,
                    json={
                        "value": [
                            {**_message("1"), "replies": [_message("2", reply_to="1")]}
                        ]
                    },
                )
            raise AssertionError("Unexpected request")

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            reader = TeamsChannelClient(
                _TEAM, _CHANNEL, token_provider=lambda: "token", http_client=client
            )
            with self.assertRaises(ChannelInputError):
                reader.list_messages(ChannelImportLimits(max_messages=1))

        with httpx.Client(
            transport=httpx.MockTransport(
                lambda _r: httpx.Response(200, json={"value": [_message("1")]})
            )
        ) as client:
            reader = TeamsChannelClient(
                _TEAM, _CHANNEL, token_provider=lambda: "token", http_client=client
            )
            with self.assertRaisesRegex(ChannelInputError, "expanded replies"):
                reader.list_messages(ChannelImportLimits())

    def test_token_failure_is_sanitized(self) -> None:
        with httpx.Client(
            transport=httpx.MockTransport(
                lambda _r: httpx.Response(200, json={"value": []})
            )
        ) as client:
            reader = TeamsChannelClient(
                _TEAM, _CHANNEL, token_provider=lambda: "", http_client=client
            )
            with self.assertRaises(ChannelProviderError):
                reader.list_messages(ChannelImportLimits())

    def test_invalid_configuration(self) -> None:
        with self.assertRaises(ChannelConfigurationError):
            TeamsChannelClient("bad", _CHANNEL, token_provider=lambda: "token")
        with self.assertRaises(ChannelConfigurationError):
            TeamsChannelClient(_TEAM, "bad/id", token_provider=lambda: "token")
        with patch.dict(
            "os.environ",
            {
                "RAG_TEAMS_TENANT_ID": "",
                "RAG_TEAMS_CLIENT_ID": "",
                "RAG_TEAMS_CLIENT_SECRET": "",
            },
        ):
            with self.assertRaisesRegex(
                ChannelConfigurationError, "RAG_TEAMS_TENANT_ID"
            ):
                TeamsChannelClient.from_environment(_TEAM, _CHANNEL)
