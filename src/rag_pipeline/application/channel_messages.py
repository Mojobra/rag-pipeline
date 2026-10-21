"""Stage complete Slack or Teams channel reads for the shared local indexer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_core.documents import Document

from rag_pipeline.application.indexing import (
    IndexingPipelineConfig,
    index_extracted_documents,
)
from rag_pipeline.exceptions import ChannelInputError
from rag_pipeline.infrastructure.vector_store import IndexingResult
from rag_pipeline.ingestion import html_to_visible_text
from rag_pipeline.ingestion.channel_messages import (
    ChannelImportLimits,
    ChannelMessage,
    message_content_size,
)


class ChannelReader(Protocol):
    """Supply one complete bounded channel traversal without transport details."""

    def list_messages(self, limits: ChannelImportLimits) -> Sequence[ChannelMessage]:
        """Return all readable channel messages or fail before indexing."""


def index_channel_messages(
    reader: ChannelReader,
    *,
    platform: str,
    channel_id: str,
    config: IndexingPipelineConfig,
    limits: ChannelImportLimits,
) -> IndexingResult:
    """Index listed messages only after a full validated provider traversal.

    The provider and HTML conversion run before model inference or Qdrant writes.
    Reimport replaces listed sources; remote deletions are not reconciled.
    """
    messages = list(reader.list_messages(limits))
    if not messages:
        raise ChannelInputError("No readable messages were found in this channel.")
    if len(messages) > limits.max_messages:
        raise ChannelInputError("Channel exceeds max_messages.")

    documents: list[Document] = []
    sources: set[str] = set()
    total_bytes = 0
    for message in messages:
        if (
            not isinstance(message, ChannelMessage)
            or message.platform != platform
            or message.channel_id != channel_id
            or message.source in sources
        ):
            raise ChannelInputError(
                "Channel listing contains invalid or duplicate messages."
            )
        byte_count = message_content_size(message.content)
        if byte_count > limits.max_message_bytes:
            raise ChannelInputError("A channel message exceeds max_message_bytes.")
        total_bytes += byte_count
        if total_bytes > limits.max_total_bytes:
            raise ChannelInputError("Channel exceeds max_total_bytes.")
        sources.add(message.source)
        content = (
            html_to_visible_text(message.content)
            if message.content_type == "html"
            else message.content
        )
        if not content.strip():
            continue
        metadata: dict[str, str] = {
            "source": message.source,
            "platform": platform,
            "channel_id": channel_id,
            "message_id": message.message_id,
        }
        if message.parent_id is not None:
            metadata["parent_id"] = message.parent_id
        if message.web_url is not None:
            metadata["web_url"] = message.web_url
        if message.created_at is not None:
            metadata["created_at"] = message.created_at
        documents.append(Document(page_content=content, metadata=metadata))

    return index_extracted_documents(documents, sources=sources, config=config)
