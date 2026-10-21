"""Read one Slack channel and its threads through bounded Web API calls."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import httpx

from rag_pipeline.exceptions import (
    ChannelConfigurationError,
    ChannelInputError,
    ChannelProviderError,
)
from rag_pipeline.ingestion.channel_messages import (
    ChannelImportLimits,
    ChannelMessage,
    message_content_size,
)

_API_ROOT = "https://slack.com/api/"
_PAGE_SIZE = 15
_MAX_REQUESTS = 500
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024


class SlackChannelClient:
    """Read one token-visible channel without exposing the bearer credential.

    The caller owns the context manager. Injected HTTP clients remain under
    their creator's ownership so all transport paths can be tested offline.
    """

    def __init__(
        self,
        channel_id: str,
        *,
        token: str,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not _valid_id(channel_id):
            raise ChannelConfigurationError("Slack channel_id must be an ASCII ID.")
        if (
            not isinstance(token, str)
            or not token.strip()
            or "\n" in token
            or "\r" in token
        ):
            raise ChannelConfigurationError("RAG_SLACK_BOT_TOKEN is required.")
        self.channel_id = channel_id
        self._token = token
        self._client = (
            http_client
            if http_client is not None
            else httpx.Client(timeout=30.0, follow_redirects=False)
        )
        self._owns_client = http_client is None
        self._requests = 0

    @classmethod
    def from_environment(cls, channel_id: str) -> SlackChannelClient:
        """Read the bot token from the process environment, never CLI args."""
        return cls(channel_id, token=os.getenv("RAG_SLACK_BOT_TOKEN", ""))

    def __enter__(self) -> SlackChannelClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close only the connection pool created by this adapter."""
        if self._owns_client:
            self._client.close()

    def list_messages(self, limits: ChannelImportLimits) -> list[ChannelMessage]:
        """Finish a cursor-paginated channel and thread traversal before writes.

        The workspace identity comes from auth.test, not a caller-supplied ID.
        Malformed pagination, limits, and API failures abort the full import.
        """
        if not isinstance(limits, ChannelImportLimits):
            raise TypeError("limits must be ChannelImportLimits.")
        self._requests = 0
        auth = self._get("auth.test", {})
        team_id = auth.get("team_id")
        if not isinstance(team_id, str) or not _valid_id(team_id):
            raise ChannelInputError("Slack returned no valid workspace ID.")

        seen: set[str] = set()
        messages: list[ChannelMessage] = []
        total_bytes = 0
        for raw in self._pages("conversations.history", {"channel": self.channel_id}):
            parent_id = _message_id(raw)
            if raw.get("type") != "message" or raw.get("subtype") not in (
                None,
                "bot_message",
            ):
                continue
            total_bytes = self._append(
                raw, team_id, None, limits, seen, messages, total_bytes
            )
            reply_count = raw.get("reply_count", 0)
            if (
                isinstance(reply_count, bool)
                or not isinstance(reply_count, int)
                or reply_count < 0
            ):
                raise ChannelInputError("Slack returned an invalid reply count.")
            if reply_count:
                for reply in self._pages(
                    "conversations.replies",
                    {"channel": self.channel_id, "ts": parent_id},
                ):
                    reply_id = _message_id(reply)
                    if reply_id == parent_id:
                        continue
                    if reply.get("thread_ts") != parent_id:
                        raise ChannelInputError(
                            "Slack returned a reply outside its thread."
                        )
                    if reply.get("type") != "message" or reply.get("subtype") not in (
                        None,
                        "bot_message",
                    ):
                        continue
                    total_bytes = self._append(
                        reply, team_id, parent_id, limits, seen, messages, total_bytes
                    )
        return sorted(messages, key=lambda message: message.source)

    def _append(
        self,
        raw: dict[str, object],
        team_id: str,
        parent_id: str | None,
        limits: ChannelImportLimits,
        seen: set[str],
        messages: list[ChannelMessage],
        total_bytes: int,
    ) -> int:
        message_id = _message_id(raw)
        if message_id in seen:
            raise ChannelInputError("Slack returned a duplicate message.")
        seen.add(message_id)
        content = raw.get("text")
        if not isinstance(content, str):
            raise ChannelInputError("Slack returned an invalid message body.")
        size = message_content_size(content)
        if size > limits.max_message_bytes:
            raise ChannelInputError("A channel message exceeds max_message_bytes.")
        if len(messages) >= limits.max_messages:
            raise ChannelInputError("Channel exceeds max_messages.")
        if total_bytes + size > limits.max_total_bytes:
            raise ChannelInputError("Channel exceeds max_total_bytes.")
        messages.append(
            ChannelMessage(
                source=f"slack://{team_id}/{self.channel_id}/{message_id}",
                platform="slack",
                channel_id=self.channel_id,
                message_id=message_id,
                content=content,
                parent_id=parent_id,
                created_at=message_id,
            )
        )
        return total_bytes + size

    def _pages(
        self, method: str, params: dict[str, str]
    ) -> Iterator[dict[str, object]]:
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            query = {**params, "limit": str(_PAGE_SIZE)}
            if cursor is not None:
                query["cursor"] = cursor
            page = self._get(method, query)
            raw_messages = page.get("messages")
            if not isinstance(raw_messages, list):
                raise ChannelInputError("Slack returned an invalid message page.")
            for raw in raw_messages:
                if not isinstance(raw, dict):
                    raise ChannelInputError("Slack returned an invalid message.")
                yield raw
            has_more = page.get("has_more", False)
            if not isinstance(has_more, bool):
                raise ChannelInputError("Slack returned invalid pagination metadata.")
            if not has_more:
                return
            metadata = page.get("response_metadata")
            next_cursor = (
                metadata.get("next_cursor") if isinstance(metadata, dict) else None
            )
            if (
                not isinstance(next_cursor, str)
                or not next_cursor
                or next_cursor in seen_cursors
                or len(next_cursor) > 4096
            ):
                raise ChannelInputError("Slack pagination repeated or is incomplete.")
            seen_cursors.add(next_cursor)
            cursor = next_cursor

    def _get(self, method: str, params: dict[str, str]) -> dict[str, object]:
        if self._requests >= _MAX_REQUESTS:
            raise ChannelInputError("Slack request count exceeds its limit.")
        self._requests += 1
        try:
            with self._client.stream(
                "GET",
                f"{_API_ROOT}{method}",
                params=params,
                headers={"Authorization": f"Bearer {self._token}"},
                follow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    raise ChannelProviderError(
                        f"Slack {method} failed (HTTP {response.status_code})."
                    )
                content = _bounded_bytes(response)
        except httpx.HTTPError as exc:
            raise ChannelProviderError(f"Slack {method} request failed.") from exc
        try:
            data: object = json.loads(content)
        except (ValueError, RecursionError) as exc:
            raise ChannelInputError("Slack returned invalid JSON.") from exc
        if not isinstance(data, dict):
            raise ChannelInputError("Slack returned an invalid response.")
        if data.get("ok") is not True:
            raise ChannelProviderError(
                f"Slack {method} failed; check token scopes and channel membership."
            )
        return data


def _message_id(raw: dict[str, object]) -> str:
    value = raw.get("ts")
    if (
        not isinstance(value, str)
        or len(value) > 32
        or value.count(".") != 1
        or not all(part.isascii() and part.isdecimal() for part in value.split("."))
    ):
        raise ChannelInputError("Slack returned an invalid message timestamp.")
    return value


def _valid_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and value.isascii()
        and all(char.isalnum() or char in "-_" for char in value)
    )


def _bounded_bytes(response: httpx.Response) -> bytes:
    result = bytearray()
    for block in response.iter_bytes():
        if len(result) + len(block) > _MAX_RESPONSE_BYTES:
            raise ChannelInputError("Channel API response exceeds the size limit.")
        result.extend(block)
    return bytes(result)
