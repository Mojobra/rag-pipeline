"""Read a bounded Microsoft Teams channel and replies through Graph v1.0."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from urllib.parse import quote, unquote, urlencode, urlsplit
from uuid import UUID

import httpx

from rag_pipeline.exceptions import (
    ChannelConfigurationError,
    ChannelInputError,
    ChannelProviderError,
    SharePointConfigurationError,
    SharePointProviderError,
)
from rag_pipeline.infrastructure.sharepoint import MsalTokenProvider
from rag_pipeline.ingestion.channel_messages import (
    ChannelImportLimits,
    ChannelMessage,
    message_content_size,
)

_GRAPH_ROOT = "https://graph.microsoft.com/v1.0"
_MAX_REQUESTS = 500
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024


class TeamsChannelClient:
    """Use an app-only Graph token to read one explicitly selected channel.

    MSAL and the HTTP client are injectable for offline tests. The caller owns
    the context manager; only internally created HTTP pools are closed here.
    """

    def __init__(
        self,
        team_id: str,
        channel_id: str,
        *,
        token_provider: Callable[[], str],
        http_client: httpx.Client | None = None,
    ) -> None:
        try:
            canonical_team_id = str(UUID(team_id))
        except (ValueError, TypeError) as exc:
            raise ChannelConfigurationError("Teams team_id must be a UUID.") from exc
        if (
            not isinstance(channel_id, str)
            or not channel_id
            or len(channel_id) > 512
            or not channel_id.isascii()
            or any(
                char.isspace() or char in "/\\" or ord(char) < 32 for char in channel_id
            )
        ):
            raise ChannelConfigurationError("Teams channel_id must be an ASCII ID.")
        self.team_id = canonical_team_id
        self.channel_id = channel_id
        self._token_provider = token_provider
        self._client = (
            http_client
            if http_client is not None
            else httpx.Client(timeout=30.0, follow_redirects=False)
        )
        self._owns_client = http_client is None
        self._requests = 0

    @classmethod
    def from_environment(cls, team_id: str, channel_id: str) -> TeamsChannelClient:
        """Build a Graph reader from dedicated Teams Entra app credentials."""
        names = (
            "RAG_TEAMS_TENANT_ID",
            "RAG_TEAMS_CLIENT_ID",
            "RAG_TEAMS_CLIENT_SECRET",
        )
        values = [os.getenv(name) for name in names]
        missing = [name for name, value in zip(names, values, strict=True) if not value]
        if missing:
            raise ChannelConfigurationError(
                f"Missing Teams environment variable(s): {', '.join(missing)}."
            )
        try:
            provider = MsalTokenProvider(
                values[0] or "", values[1] or "", values[2] or ""
            )
        except (SharePointConfigurationError, SharePointProviderError) as exc:
            raise ChannelConfigurationError(
                "Invalid Teams Entra app credentials."
            ) from exc
        return cls(team_id, channel_id, token_provider=provider)

    def __enter__(self) -> TeamsChannelClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        """Release only a connection pool created by this adapter."""
        if self._owns_client:
            self._client.close()

    def list_messages(self, limits: ChannelImportLimits) -> list[ChannelMessage]:
        """Read every root and reply in one bounded channel traversal.

        Root pages request expanded replies. A reply continuation URL is
        followed on the same Graph endpoint so large threads are not truncated.
        """
        if not isinstance(limits, ChannelImportLimits):
            raise TypeError("limits must be ChannelImportLimits.")
        self._requests = 0
        path = self._channel_path()
        query = urlencode({"$top": 50, "$expand": "replies"})
        messages: list[ChannelMessage] = []
        seen: set[str] = set()
        total_bytes = 0
        for raw in self._pages(f"{_GRAPH_ROOT}{path}?{query}", path):
            parent_id = _message_id(raw)
            if raw.get("messageType") != "message":
                continue
            parent = self._parse(raw, parent_id=None)
            if parent is not None:
                total_bytes = self._append(parent, limits, seen, messages, total_bytes)
            replies = raw.get("replies")
            if not isinstance(replies, list):
                raise ChannelInputError(
                    "Graph returned a root without expanded replies."
                )
            for reply in replies:
                if not isinstance(reply, dict):
                    raise ChannelInputError("Graph returned an invalid reply.")
                parsed = self._parse(reply, parent_id=parent_id)
                if parsed is not None:
                    total_bytes = self._append(
                        parsed, limits, seen, messages, total_bytes
                    )
            next_link = raw.get("replies@odata.nextLink")
            if next_link is not None:
                reply_path = f"{path}/{quote(parent_id, safe='')}/replies"
                if not isinstance(next_link, str):
                    raise ChannelInputError("Graph returned an invalid reply nextLink.")
                for reply in self._pages(next_link, reply_path):
                    parsed = self._parse(reply, parent_id=parent_id)
                    if parsed is not None:
                        total_bytes = self._append(
                            parsed, limits, seen, messages, total_bytes
                        )
        return sorted(messages, key=lambda message: message.source)

    def _append(
        self,
        message: ChannelMessage,
        limits: ChannelImportLimits,
        seen: set[str],
        messages: list[ChannelMessage],
        total_bytes: int,
    ) -> int:
        if message.source in seen:
            raise ChannelInputError("Graph returned a duplicate channel message.")
        seen.add(message.source)
        size = message_content_size(message.content)
        if size > limits.max_message_bytes:
            raise ChannelInputError("A channel message exceeds max_message_bytes.")
        if (
            len(messages) >= limits.max_messages
            or total_bytes + size > limits.max_total_bytes
        ):
            raise ChannelInputError("Teams channel exceeds configured import limits.")
        messages.append(message)
        return total_bytes + size

    def _parse(
        self, raw: dict[str, object], *, parent_id: str | None
    ) -> ChannelMessage | None:
        message_id = _message_id(raw)
        reply_to = raw.get("replyToId")
        if parent_id is None and reply_to is not None:
            raise ChannelInputError("Graph returned a reply in the root listing.")
        if parent_id is not None and reply_to != parent_id:
            raise ChannelInputError("Graph returned a reply outside its thread.")
        if (
            raw.get("messageType") != "message"
            or raw.get("deletedDateTime") is not None
        ):
            return None
        body = raw.get("body")
        if not isinstance(body, dict):
            raise ChannelInputError("Graph returned an invalid message body.")
        content = body.get("content")
        content_type = body.get("contentType")
        if not isinstance(content, str) or content_type not in {"text", "html"}:
            raise ChannelInputError("Graph returned an unsupported message body.")
        web_url = raw.get("webUrl")
        if web_url is not None and (
            not isinstance(web_url, str) or not _safe_teams_url(web_url)
        ):
            raise ChannelInputError("Graph returned an invalid Teams message URL.")
        created_at = raw.get("createdDateTime")
        if created_at is not None and not isinstance(created_at, str):
            raise ChannelInputError("Graph returned an invalid message timestamp.")
        suffix = quote(message_id, safe="")
        if parent_id is not None:
            suffix = f"{quote(parent_id, safe='')}/replies/{suffix}"
        return ChannelMessage(
            source=f"teams://{self.team_id}/{quote(self.channel_id, safe='')}/{suffix}",
            platform="teams",
            channel_id=self.channel_id,
            message_id=message_id,
            content=content,
            content_type=content_type,
            parent_id=parent_id,
            web_url=web_url,
            created_at=created_at,
        )

    def _pages(self, first_url: str, expected_path: str) -> Iterator[dict[str, object]]:
        url: str | None = first_url
        seen_urls: set[str] = set()
        while url is not None:
            if url in seen_urls:
                raise ChannelInputError("Graph pagination repeated.")
            seen_urls.add(url)
            page = self._get(url, expected_path)
            items = page.get("value")
            if not isinstance(items, list):
                raise ChannelInputError("Graph returned an invalid message page.")
            for item in items:
                if not isinstance(item, dict):
                    raise ChannelInputError(
                        "Graph returned an invalid channel message."
                    )
                yield item
            next_link = page.get("@odata.nextLink")
            if next_link is not None and not isinstance(next_link, str):
                raise ChannelInputError("Graph returned an invalid nextLink.")
            url = next_link

    def _get(self, url: str, expected_path: str) -> dict[str, object]:
        _validate_graph_url(url, expected_path)
        if self._requests >= _MAX_REQUESTS:
            raise ChannelInputError("Graph request count exceeds its limit.")
        self._requests += 1
        try:
            token = self._token_provider()
        except Exception as exc:
            raise ChannelProviderError(
                "Could not acquire a Teams Graph token."
            ) from exc
        if not isinstance(token, str) or not token:
            raise ChannelProviderError("Teams Graph token provider returned no token.")
        try:
            with self._client.stream(
                "GET",
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
                follow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    raise ChannelProviderError(
                        f"Teams Graph request failed (HTTP {response.status_code})."
                    )
                content = _bounded_bytes(response)
        except httpx.HTTPError as exc:
            raise ChannelProviderError("Teams Graph request failed.") from exc
        try:
            data: object = json.loads(content)
        except (ValueError, RecursionError) as exc:
            raise ChannelInputError("Graph returned invalid channel JSON.") from exc
        if not isinstance(data, dict):
            raise ChannelInputError("Graph returned an invalid channel response.")
        return data

    def _channel_path(self) -> str:
        return (
            f"/teams/{quote(self.team_id, safe='')}"
            f"/channels/{quote(self.channel_id, safe='')}/messages"
        )


def _validate_graph_url(url: str, expected_path: str) -> None:
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise ChannelInputError("Graph returned an unsafe continuation URL.") from exc
    if (
        parts.scheme != "https"
        or parts.netloc != "graph.microsoft.com"
        or unquote(parts.path) != f"/v1.0{unquote(expected_path)}"
        or parts.fragment
        or len(url) > 8192
    ):
        raise ChannelInputError("Graph returned an unsafe continuation URL.")


def _message_id(raw: dict[str, object]) -> str:
    value = raw.get("id")
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdecimal()
        or len(value) > 64
    ):
        raise ChannelInputError("Graph returned an invalid message ID.")
    return value


def _safe_teams_url(value: object) -> bool:
    if not isinstance(value, str) or len(value) > 4096:
        return False
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    return (
        parts.scheme == "https"
        and parts.netloc in {"teams.microsoft.com", "teams.cloud.microsoft"}
        and not parts.fragment
        and not any(ord(char) < 32 for char in value)
    )


def _bounded_bytes(response: httpx.Response) -> bytes:
    result = bytearray()
    for block in response.iter_bytes():
        if len(result) + len(block) > _MAX_RESPONSE_BYTES:
            raise ChannelInputError("Channel API response exceeds the size limit.")
        result.extend(block)
    return bytes(result)
