"""Create bounded retrieval chunks from embedding-based topic transitions.

The experimental splitter detects sentence-level semantic breakpoints through
LangChain's ``Embeddings`` interface, while enforcing hard character limits and
exact source offsets needed by the rest of the pipeline.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline.exceptions import ChunkingError, InvalidChunkingConfigurationError

_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])(?:[ \t]+|\r?\n+)|(?:\r?\n[ \t]*){2,}")


@dataclass(frozen=True, slots=True)
class SemanticChunkingConfig:
    """Validated controls for bounded embedding-aware chunking.

    ``max_chunk_size`` is a hard character cap, ``min_chunk_size`` suppresses
    tiny semantic groups where possible, and the percentile selects unusually
    large adjacent-sentence cosine distances. ``buffer_size`` adds neighboring
    sentences to each embedding context without changing emitted source text.
    """

    max_chunk_size: int = 1000
    min_chunk_size: int = 200
    breakpoint_percentile: float = 95.0
    buffer_size: int = 1

    def __post_init__(self) -> None:
        """Reject unsafe limits before model inference or document processing."""
        _validate_positive_integer("max_chunk_size", self.max_chunk_size)
        _validate_positive_integer("min_chunk_size", self.min_chunk_size)
        if self.min_chunk_size > self.max_chunk_size:
            raise InvalidChunkingConfigurationError(
                "min_chunk_size cannot exceed max_chunk_size."
            )
        if isinstance(self.buffer_size, bool) or not isinstance(self.buffer_size, int):
            raise InvalidChunkingConfigurationError("buffer_size must be an integer.")
        if self.buffer_size < 0:
            raise InvalidChunkingConfigurationError("buffer_size cannot be negative.")
        if isinstance(self.breakpoint_percentile, bool) or not isinstance(
            self.breakpoint_percentile, Real
        ):
            raise InvalidChunkingConfigurationError(
                "breakpoint_percentile must be a number."
            )
        percentile = float(self.breakpoint_percentile)
        if not math.isfinite(percentile) or not 0 < percentile <= 100:
            raise InvalidChunkingConfigurationError(
                "breakpoint_percentile must be greater than 0 and at most 100."
            )
        object.__setattr__(self, "breakpoint_percentile", percentile)


@dataclass(frozen=True, slots=True)
class _TextSpan:
    """Half-open source interval used while grouping semantic units."""

    start: int
    end: int


def chunk_documents_semantically(
    documents: Iterable[Document],
    *,
    embeddings: Embeddings,
    config: SemanticChunkingConfig | None = None,
) -> list[Document]:
    """Group source sentences around unusually large embedding distances.

    The function performs embedding inference for documents with multiple text
    units. It never mutates input documents, skips blank content, applies a hard
    size cap even to long sentences, and emits exact source offsets plus normal
    chunk metadata. Provider failures propagate from the supplied LangChain
    embedding implementation; malformed vectors raise ``ChunkingError``.
    """
    if not isinstance(embeddings, Embeddings):
        raise TypeError("embeddings must implement LangChain's Embeddings interface.")
    settings = config or SemanticChunkingConfig()
    chunks: list[Document] = []

    for document in documents:
        if not isinstance(document, Document):
            raise TypeError("documents must contain LangChain Document objects.")
        if not document.page_content.strip():
            continue

        units = _build_units(
            document.page_content,
            max_chunk_size=settings.max_chunk_size,
        )
        spans = _group_semantic_units(
            document.page_content,
            units,
            embeddings=embeddings,
            config=settings,
        )
        chunk_count = len(spans)
        for chunk_index, span in enumerate(spans):
            content = document.page_content[span.start : span.end]
            chunks.append(
                Document(
                    page_content=content,
                    metadata={
                        **document.metadata,
                        "start_index": span.start,
                        "end_index": span.end,
                        "chunk_index": chunk_index,
                        "chunk_count": chunk_count,
                        "chunk_char_count": len(content),
                        "chunking_strategy": "semantic",
                    },
                )
            )
    return chunks


def _build_units(text: str, *, max_chunk_size: int) -> tuple[_TextSpan, ...]:
    """Find sentence/paragraph spans and hard-split any oversized unit."""
    units: list[_TextSpan] = []
    cursor = 0
    for boundary in _BOUNDARY_PATTERN.finditer(text):
        _append_trimmed_span(units, text, cursor, boundary.start())
        cursor = boundary.end()
    _append_trimmed_span(units, text, cursor, len(text))

    bounded_units: list[_TextSpan] = []
    for unit in units:
        if unit.end - unit.start <= max_chunk_size:
            bounded_units.append(unit)
            continue
        bounded_units.extend(
            _split_oversized_unit(
                text,
                unit,
                max_chunk_size=max_chunk_size,
            )
        )
    return tuple(bounded_units)


def _append_trimmed_span(
    spans: list[_TextSpan],
    text: str,
    start: int,
    end: int,
) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        spans.append(_TextSpan(start, end))


def _split_oversized_unit(
    text: str,
    unit: _TextSpan,
    *,
    max_chunk_size: int,
) -> tuple[_TextSpan, ...]:
    """Apply LangChain's recursive splitter as a zero-overlap safety fallback."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=0,
        length_function=len,
        add_start_index=True,
    )
    local_chunks = splitter.create_documents([text[unit.start : unit.end]])
    spans = []
    for chunk in local_chunks:
        local_start = chunk.metadata.get("start_index")
        if (
            isinstance(local_start, bool)
            or not isinstance(local_start, int)
            or local_start < 0
        ):
            raise ChunkingError("LangChain did not provide a valid start_index.")
        start = unit.start + local_start
        spans.append(_TextSpan(start, start + len(chunk.page_content)))
    return tuple(spans)


def _group_semantic_units(
    text: str,
    units: tuple[_TextSpan, ...],
    *,
    embeddings: Embeddings,
    config: SemanticChunkingConfig,
) -> tuple[_TextSpan, ...]:
    """Calculate semantic breakpoints and greedily enforce size constraints."""
    if not units:
        return ()
    if len(units) == 1:
        return units

    contexts = [
        text[
            units[max(0, index - config.buffer_size)].start : units[
                min(len(units) - 1, index + config.buffer_size)
            ].end
        ]
        for index in range(len(units))
    ]
    vectors = _validated_vectors(
        embeddings.embed_documents(contexts),
        expected_count=len(contexts),
    )
    distances = tuple(
        _cosine_distance(left, right) for left, right in pairwise(vectors)
    )
    threshold = _percentile(distances, config.breakpoint_percentile)
    semantic_breaks = {
        index for index, distance in enumerate(distances) if distance > threshold
    }
    return _build_bounded_groups(
        units,
        semantic_breaks=semantic_breaks,
        min_chunk_size=config.min_chunk_size,
        max_chunk_size=config.max_chunk_size,
    )


def _build_bounded_groups(
    units: tuple[_TextSpan, ...],
    *,
    semantic_breaks: set[int],
    min_chunk_size: int,
    max_chunk_size: int,
) -> tuple[_TextSpan, ...]:
    """Prefer semantic boundaries while treating the maximum as an invariant."""
    groups: list[_TextSpan] = []
    group_start = units[0].start

    for index, unit in enumerate(units):
        if index > 0 and unit.end - group_start > max_chunk_size:
            groups.append(_TextSpan(group_start, units[index - 1].end))
            group_start = unit.start

        group_end = unit.end
        if (
            index < len(units) - 1
            and index in semantic_breaks
            and group_end - group_start >= min_chunk_size
        ):
            groups.append(_TextSpan(group_start, group_end))
            group_start = units[index + 1].start

    groups.append(_TextSpan(group_start, units[-1].end))
    if (
        len(groups) > 1
        and groups[-1].end - groups[-1].start < min_chunk_size
        and groups[-1].end - groups[-2].start <= max_chunk_size
    ):
        groups[-2:] = [_TextSpan(groups[-2].start, groups[-1].end)]
    return tuple(groups)


def _validated_vectors(
    raw_vectors: Sequence[Sequence[float]],
    *,
    expected_count: int,
) -> tuple[tuple[float, ...], ...]:
    """Normalize finite, nonzero, consistently sized provider vectors."""
    try:
        vectors = list(raw_vectors)
    except TypeError as exc:
        raise ChunkingError(
            "semantic embedding provider output is not iterable."
        ) from exc
    if len(vectors) != expected_count:
        raise ChunkingError(
            "semantic embedding provider returned "
            f"{len(vectors)} vector(s) for {expected_count} text unit(s)."
        )

    normalized: list[tuple[float, ...]] = []
    expected_dimension: int | None = None
    for vector_index, raw_vector in enumerate(vectors):
        try:
            raw_values = list(raw_vector)
        except TypeError as exc:
            raise ChunkingError(
                f"semantic embedding vector {vector_index} is not iterable."
            ) from exc
        values: list[float] = []
        for value_index, value in enumerate(raw_values):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ChunkingError(
                    "semantic embedding vector "
                    f"{vector_index} has a non-numeric value at {value_index}."
                )
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ChunkingError(
                    f"semantic embedding vector {vector_index} is not finite."
                )
            values.append(numeric_value)
        if not values:
            raise ChunkingError(f"semantic embedding vector {vector_index} is empty.")
        if expected_dimension is None:
            expected_dimension = len(values)
        elif len(values) != expected_dimension:
            raise ChunkingError(
                "semantic embedding vector dimensions are inconsistent."
            )
        if not any(values):
            raise ChunkingError(
                f"semantic embedding vector {vector_index} has zero magnitude."
            )
        normalized.append(tuple(values))
    return tuple(normalized)


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute bounded cosine distance without squaring large vector values."""
    left_norm = math.hypot(*left)
    right_norm = math.hypot(*right)
    similarity = sum(
        (left_value / left_norm) * (right_value / right_norm)
        for left_value, right_value in zip(left, right, strict=True)
    )
    return 1.0 - max(-1.0, min(1.0, similarity))


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sequence."""
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _validate_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidChunkingConfigurationError(f"{name} must be an integer.")
    if value <= 0:
        raise InvalidChunkingConfigurationError(f"{name} must be greater than zero.")
