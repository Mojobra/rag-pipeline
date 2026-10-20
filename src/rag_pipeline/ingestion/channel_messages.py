"""Represent bounded channel messages before shared RAG indexing."""

from __future__ import annotations

from dataclasses import dataclass

from rag_pipeline.exceptions import ChannelConfigurationError, ChannelInputError

_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ChannelImportLimits:
    """Bound one manual channel traversal and its in-memory message text."""

    max_messages: int = 100
    max_message_bytes: int = 64 * 1024
    max_total_bytes: int = 2 * _MIB

    def __post_init__(self) -> None:
        for name in ("max_messages", "max_message_bytes", "max_total_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ChannelConfigurationError(f"{name} must be a positive integer.")


@dataclass(frozen=True, slots=True)
class ChannelMessage:
    """Carry one platform-scoped message and its stable citation source."""

    source: str
    platform: str
    channel_id: str
    message_id: str
    content: str
    content_type: str = "text"
    parent_id: str | None = None
    web_url: str | None = None
    created_at: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.source
            or self.platform not in {"slack", "teams"}
            or not self.channel_id
            or not self.message_id
            or not isinstance(self.content, str)
            or self.content_type not in {"text", "html"}
        ):
            raise ChannelInputError("Provider returned an invalid channel message.")


def message_content_size(content: str) -> int:
    """Measure source UTF-8 bytes and reject malformed provider Unicode."""
    try:
        return len(content.encode("utf-8"))
    except UnicodeError as exc:
        raise ChannelInputError("Provider returned invalid message encoding.") from exc
