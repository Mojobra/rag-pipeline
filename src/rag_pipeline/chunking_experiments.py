"""Compare character-based chunking policies on one document snapshot.

The experiment layer reports structural size and duplication metrics without
embedding, indexing, retrieval, or model calls, keeping comparisons reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from langchain_core.documents import Document

from rag_pipeline.chunking import ChunkingConfig, chunk_documents
from rag_pipeline.exceptions import (
    InvalidChunkingConfigurationError,
    InvalidChunkingExperimentError,
)


DEFAULT_CHUNKING_CANDIDATES = (
    ChunkingConfig(chunk_size=500, chunk_overlap=100),
    ChunkingConfig(chunk_size=1000, chunk_overlap=200),
    ChunkingConfig(chunk_size=1500, chunk_overlap=300),
)


@dataclass(frozen=True, slots=True)
class ChunkingMetrics:
    """Structural cost and size measurements for one chunking candidate.

    Metrics describe emitted chunk lengths and duplicated source characters.
    They are diagnostic signals, not retrieval or answer-quality scores.
    """

    chunk_count: int
    total_chunk_characters: int
    min_chunk_characters: int
    mean_chunk_characters: float
    p95_chunk_characters: int
    max_chunk_characters: int
    duplicated_characters: int
    duplication_percentage: float


@dataclass(frozen=True, slots=True)
class ChunkingExperimentResult:
    """Pair one validated chunking policy with its measured output.

    Results form the per-candidate rows in both human-readable and JSON reports.
    """

    config: ChunkingConfig
    metrics: ChunkingMetrics


@dataclass(frozen=True, slots=True)
class ChunkingExperimentReport:
    """Complete comparison produced from one materialized document snapshot.

    Sharing one input snapshot prevents document iteration or input variance
    from being mistaken for a difference between chunking configurations.
    """

    input_document_count: int
    chunked_document_count: int
    source_character_count: int
    results: tuple[ChunkingExperimentResult, ...]


def parse_chunking_candidate(value: str) -> ChunkingConfig:
    """Convert a CLI ``SIZE:OVERLAP`` value into validated settings.

    Syntax and configuration errors are normalized to
    ``InvalidChunkingExperimentError`` so the command layer can report one
    consistent user-facing failure category.
    """
    if not isinstance(value, str):
        raise TypeError("chunking candidate must be a string.")

    parts = [part.strip() for part in value.split(":")]
    if len(parts) != 2 or not all(parts):
        raise InvalidChunkingExperimentError(
            "chunking candidate must use SIZE:OVERLAP, for example 1000:200."
        )

    try:
        chunk_size, chunk_overlap = (int(part) for part in parts)
    except ValueError as exc:
        raise InvalidChunkingExperimentError(
            "chunking candidate size and overlap must be integers."
        ) from exc

    try:
        return ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except InvalidChunkingConfigurationError as exc:
        raise InvalidChunkingExperimentError(
            f"invalid chunking candidate {value!r}: {exc}"
        ) from exc


def run_chunking_experiment(
    documents: Iterable[Document],
    *,
    candidates: Iterable[ChunkingConfig] = DEFAULT_CHUNKING_CANDIDATES,
) -> ChunkingExperimentReport:
    """Measure every candidate against the same in-memory document snapshot.

    Input iterables are consumed once, validated, and reused for every policy.
    The function is deterministic and side-effect free beyond CPU and memory
    use; it does not call model providers or mutate persistent state.
    """
    document_snapshot = tuple(documents)
    _validate_documents(document_snapshot)
    candidate_snapshot = _validate_candidates(tuple(candidates))

    chunked_document_count = sum(
        1 for document in document_snapshot if document.page_content.strip()
    )
    source_character_count = sum(
        len(document.page_content)
        for document in document_snapshot
        if document.page_content.strip()
    )
    results = tuple(
        _measure_candidate(document_snapshot, candidate)
        for candidate in candidate_snapshot
    )

    return ChunkingExperimentReport(
        input_document_count=len(document_snapshot),
        chunked_document_count=chunked_document_count,
        source_character_count=source_character_count,
        results=results,
    )


def chunking_experiment_to_dict(
    report: ChunkingExperimentReport,
) -> dict[str, object]:
    """Serialize an experiment report to stable JSON-compatible primitives.

    Field names intentionally form a machine-readable reporting contract for
    recording or comparing experiment runs outside the terminal table.
    """
    return {
        "input_document_count": report.input_document_count,
        "chunked_document_count": report.chunked_document_count,
        "source_character_count": report.source_character_count,
        "candidates": [
            {
                "chunk_size": result.config.chunk_size,
                "chunk_overlap": result.config.chunk_overlap,
                "chunk_count": result.metrics.chunk_count,
                "total_chunk_characters": result.metrics.total_chunk_characters,
                "min_chunk_characters": result.metrics.min_chunk_characters,
                "mean_chunk_characters": result.metrics.mean_chunk_characters,
                "p95_chunk_characters": result.metrics.p95_chunk_characters,
                "max_chunk_characters": result.metrics.max_chunk_characters,
                "duplicated_characters": result.metrics.duplicated_characters,
                "duplication_percentage": result.metrics.duplication_percentage,
            }
            for result in report.results
        ],
    }


def format_chunking_experiment_table(report: ChunkingExperimentReport) -> str:
    """Render an experiment report as a width-aligned terminal table.

    Formatting has no I/O side effect; the caller decides where to print or
    persist the returned text.
    """
    headers = (
        "Size",
        "Overlap",
        "Chunks",
        "Min",
        "Mean",
        "P95",
        "Max",
        "Total chars",
        "Duplicated",
    )
    rows = [
        (
            str(result.config.chunk_size),
            str(result.config.chunk_overlap),
            str(result.metrics.chunk_count),
            str(result.metrics.min_chunk_characters),
            f"{result.metrics.mean_chunk_characters:.1f}",
            str(result.metrics.p95_chunk_characters),
            str(result.metrics.max_chunk_characters),
            str(result.metrics.total_chunk_characters),
            (
                f"{result.metrics.duplicated_characters} "
                f"({result.metrics.duplication_percentage:.1f}%)"
            ),
        )
        for result in report.results
    ]
    widths = tuple(
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    )

    def format_row(values: Sequence[str]) -> str:
        return "  ".join(
            value.rjust(width) for value, width in zip(values, widths, strict=True)
        )

    separator = "  ".join("-" * width for width in widths)
    summary = (
        f"Documents: {report.input_document_count} input, "
        f"{report.chunked_document_count} non-blank | "
        f"Source characters: {report.source_character_count}"
    )
    return "\n".join(
        (
            "Chunking experiment",
            summary,
            "",
            format_row(headers),
            separator,
            *(format_row(row) for row in rows),
        )
    )


def _validate_documents(documents: tuple[Document, ...]) -> None:
    for document in documents:
        if not isinstance(document, Document):
            raise TypeError("documents must contain LangChain Document objects.")


def _validate_candidates(
    candidates: tuple[ChunkingConfig, ...],
) -> tuple[ChunkingConfig, ...]:
    """Require at least one unique, validated chunking configuration.

    Returning the original tuple preserves caller order in the final report.
    """
    if not candidates:
        raise InvalidChunkingExperimentError(
            "at least one chunking candidate is required."
        )

    seen: set[tuple[int, int]] = set()
    for candidate in candidates:
        if not isinstance(candidate, ChunkingConfig):
            raise TypeError("candidates must contain ChunkingConfig objects.")
        key = (candidate.chunk_size, candidate.chunk_overlap)
        if key in seen:
            raise InvalidChunkingExperimentError(
                f"duplicate chunking candidate {key[0]}:{key[1]}."
            )
        seen.add(key)
    return candidates


def _measure_candidate(
    documents: tuple[Document, ...],
    config: ChunkingConfig,
) -> ChunkingExperimentResult:
    """Chunk all documents and calculate one candidate's aggregate metrics.

    Covered source intervals are measured per input document so equal character
    offsets from different files are never merged. No input document is mutated.
    """
    chunk_lengths: list[int] = []
    covered_character_count = 0

    for document in documents:
        document_chunks = chunk_documents([document], config=config)
        chunk_lengths.extend(len(chunk.page_content) for chunk in document_chunks)
        covered_character_count += _count_covered_characters(document_chunks)

    total_chunk_characters = sum(chunk_lengths)
    duplicated_characters = total_chunk_characters - covered_character_count
    duplication_percentage = (
        duplicated_characters / total_chunk_characters * 100
        if total_chunk_characters
        else 0.0
    )
    sorted_lengths = sorted(chunk_lengths)

    metrics = ChunkingMetrics(
        chunk_count=len(chunk_lengths),
        total_chunk_characters=total_chunk_characters,
        min_chunk_characters=sorted_lengths[0] if sorted_lengths else 0,
        mean_chunk_characters=(
            round(total_chunk_characters / len(chunk_lengths), 2)
            if chunk_lengths
            else 0.0
        ),
        p95_chunk_characters=_nearest_rank_percentile(sorted_lengths, 0.95),
        max_chunk_characters=sorted_lengths[-1] if sorted_lengths else 0,
        duplicated_characters=duplicated_characters,
        duplication_percentage=round(duplication_percentage, 2),
    )
    return ChunkingExperimentResult(config=config, metrics=metrics)


def _count_covered_characters(chunks: Sequence[Document]) -> int:
    """Return the union length of provenance intervals for one source document.

    Overlapping and touching chunk ranges are merged before counting, allowing
    duplicated characters to be derived from actual emitted chunks rather than
    the configured target overlap. Invalid provenance fails the experiment.
    """
    intervals: list[tuple[int, int]] = []
    for chunk in chunks:
        start_index = chunk.metadata.get("start_index")
        end_index = chunk.metadata.get("end_index")
        if (
            isinstance(start_index, bool)
            or not isinstance(start_index, int)
            or isinstance(end_index, bool)
            or not isinstance(end_index, int)
            or start_index < 0
            or end_index < start_index
        ):
            raise InvalidChunkingExperimentError(
                "chunk provenance must contain valid start_index and end_index values."
            )
        intervals.append((start_index, end_index))

    if not intervals:
        return 0

    intervals.sort()
    covered_characters = 0
    current_start, current_end = intervals[0]
    for start_index, end_index in intervals[1:]:
        if start_index <= current_end:
            current_end = max(current_end, end_index)
            continue
        covered_characters += current_end - current_start
        current_start, current_end = start_index, end_index
    return covered_characters + current_end - current_start


def _nearest_rank_percentile(sorted_values: Sequence[int], percentile: float) -> int:
    """Select a nearest-rank percentile from an already sorted sequence.

    Empty inputs return zero, matching the report's representation of a corpus
    that produced no chunks.
    """
    if not sorted_values:
        return 0
    index = math.ceil(percentile * len(sorted_values)) - 1
    return sorted_values[index]
