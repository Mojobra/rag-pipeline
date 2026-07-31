"""Run isolated full-pipeline RAG benchmarks and persist result artifacts.

The module rebuilds an exact temporary index, executes the existing retrieval
and answer evaluators through shared LangChain services, captures wall-clock
timings, and assembles a portable report. Artifact validation and comparison
live separately in :mod:`rag_pipeline.benchmarking.artifacts`.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias, cast

from langchain_core.documents import Document

from rag_pipeline.benchmarking.artifacts import (
    evaluate_benchmark_thresholds,
)
from rag_pipeline.benchmarking.config import BenchmarkChunkingConfig, BenchmarkConfig
from rag_pipeline.benchmarking.provenance import (
    CorpusFingerprint,
    DatasetFingerprint,
    fingerprint_corpus,
    fingerprint_dataset,
    read_source_revision,
    runtime_environment,
)
from rag_pipeline.benchmarking.reporting import (
    BenchmarkReport,
    benchmark_report_to_dict,
    format_benchmark_summary,
    validate_benchmark_output_path,
    write_benchmark_report,
)
from rag_pipeline.benchmarking.thresholds import (
    BenchmarkThresholdProfile,
    validate_benchmark_threshold_applicability,
)
from rag_pipeline.benchmarking.timing import (
    CaseTiming,
    Clock,
    TimingSummary,
    summarize_case_timings,
)
from rag_pipeline.benchmarking.timing import (
    StageRecorder as _StageRecorder,
)
from rag_pipeline.benchmarking.timing import (
    elapsed_seconds as _elapsed_seconds,
)
from rag_pipeline.evaluation.answers import (
    AnswerEvaluationDataset,
    AnswerEvaluationReport,
    evaluate_answers,
    load_answer_evaluation_dataset,
)
from rag_pipeline.evaluation.retrieval import (
    RetrievalEvaluationDataset,
    RetrievalEvaluationReport,
    evaluate_retrieval,
    load_retrieval_evaluation_dataset,
)
from rag_pipeline.exceptions import (
    BenchmarkInputError,
    InvalidBenchmarkConfigurationError,
)
from rag_pipeline.generation import (
    AnswerGenerator,
    GeneratedAnswer,
    create_local_answer_generator,
)
from rag_pipeline.generation.prompting import GROUNDED_ANSWER_PROMPT_ID
from rag_pipeline.infrastructure.embeddings import (
    EmbeddedDocument,
    EmbeddingService,
    create_local_embedding_service,
)
from rag_pipeline.infrastructure.sparse_embeddings import (
    SparseEmbeddingService,
    SparseEmbeddingVector,
    create_local_sparse_embedding_service,
)
from rag_pipeline.infrastructure.vector_store import (
    IndexingResult,
    LocalVectorStore,
    SearchMode,
    VectorStoreConfig,
)
from rag_pipeline.ingestion import load_documents
from rag_pipeline.ingestion.chunking import (
    ChunkingConfig,
    StructureAwareChunkingConfig,
    chunk_documents,
    chunk_documents_with_structure,
)
from rag_pipeline.ingestion.semantic_chunking import (
    SemanticChunkingConfig,
    chunk_documents_semantically,
)
from rag_pipeline.retrieval import RetrievalConfig, RetrievalResult, RetrieverService
from rag_pipeline.retrieval.reranking import (
    RerankerService,
    RerankingConfig,
    create_local_reranker_service,
)

__all__ = [
    "BENCHMARK_COLLECTION_NAME",
    "BenchmarkConfig",
    "BenchmarkReport",
    "CaseTiming",
    "TimingSummary",
    "benchmark_report_to_dict",
    "format_benchmark_summary",
    "run_benchmark",
    "summarize_case_timings",
    "validate_benchmark_output_path",
    "write_benchmark_report",
]

BENCHMARK_COLLECTION_NAME = "benchmark_documents"

Now: TypeAlias = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class _PreparedInputs:
    """Validated datasets, fingerprints, and extracted documents for one run."""

    retrieval_dataset: RetrievalEvaluationDataset
    answer_dataset: AnswerEvaluationDataset
    retrieval_fingerprint: DatasetFingerprint
    answer_fingerprint: DatasetFingerprint
    corpus_fingerprint: CorpusFingerprint
    documents: tuple[Document, ...]


@dataclass(frozen=True, slots=True)
class _PreparedPipeline:
    """Loaded shared model services and corpus vectors used by both suites."""

    chunks: tuple[Document, ...]
    embedding_service: EmbeddingService
    embedded_documents: tuple[EmbeddedDocument, ...]
    sparse_embedding_service: SparseEmbeddingService | None
    sparse_vectors: tuple[SparseEmbeddingVector, ...] | None
    reranker: RerankerService | None
    answer_generator: AnswerGenerator


@dataclass(frozen=True, slots=True)
class _ExecutionResult:
    """Index, quality reports, storage, and latency returned by execution."""

    indexing: IndexingResult
    storage_bytes: int
    retrieval_report: RetrievalEvaluationReport
    answer_report: AnswerEvaluationReport
    retrieval_timing: TimingSummary
    answer_timing: TimingSummary


def run_benchmark(
    corpus_path: str | Path,
    retrieval_dataset_path: str | Path,
    answer_dataset_path: str | Path,
    *,
    config: BenchmarkConfig,
    thresholds: BenchmarkThresholdProfile | None = None,
    clock: Clock = time.perf_counter,
    now: Now = lambda: datetime.now(UTC),
) -> BenchmarkReport:
    """Build an isolated index and benchmark retrieval plus grounded answers.

    The function reads and hashes inputs, may download/cache and run local
    models, creates then deletes a temporary Qdrant database, and returns no
    partial report after failures. The caller owns persistence of the result.
    """
    _validate_run_arguments(
        config=config,
        thresholds=thresholds,
        clock=clock,
        now=now,
    )
    started_at = _utc_datetime(now(), context="benchmark start")
    run_started = clock()
    stages = _StageRecorder(clock)

    inputs = _prepare_inputs(
        corpus_path,
        retrieval_dataset_path,
        answer_dataset_path,
        stages=stages,
    )
    if thresholds is not None:
        validate_benchmark_threshold_applicability(
            thresholds,
            corpus_sha256=inputs.corpus_fingerprint.sha256,
            retrieval_dataset_sha256=inputs.retrieval_fingerprint.sha256,
            answer_dataset_sha256=inputs.answer_fingerprint.sha256,
            top_k=config.final_top_k,
        )
    pipeline = _prepare_pipeline(config=config, inputs=inputs, stages=stages)
    execution = _execute_pipeline(
        config=config,
        inputs=inputs,
        pipeline=pipeline,
        stages=stages,
        clock=clock,
    )
    source_revision = read_source_revision()
    environment = runtime_environment()
    finished_at = _utc_datetime(now(), context="benchmark finish")
    total_seconds = _elapsed_seconds(
        run_started,
        clock(),
        context="benchmark run",
    )
    report = BenchmarkReport(
        name=config.name,
        started_at=started_at,
        finished_at=finished_at,
        total_seconds=total_seconds,
        corpus=inputs.corpus_fingerprint,
        retrieval_dataset=inputs.retrieval_fingerprint,
        answer_dataset=inputs.answer_fingerprint,
        source_revision=source_revision,
        environment=environment,
        configuration=_configuration_to_dict(config),
        index={
            "document_count": len(inputs.documents),
            "chunk_count": len(pipeline.chunks),
            "point_count": execution.indexing.total_count,
            "embedding_model": execution.indexing.embedding_model,
            "embedding_dimension": execution.indexing.embedding_dimension,
            "sparse_embedding_model": (execution.indexing.sparse_embedding_model),
            "search_mode": execution.indexing.search_mode.value,
            "storage_bytes": execution.storage_bytes,
            "collection_lifecycle": "temporary",
        },
        stage_seconds=stages.durations,
        retrieval_timing=execution.retrieval_timing,
        answer_timing=execution.answer_timing,
        retrieval_report=execution.retrieval_report,
        answer_report=execution.answer_report,
        threshold_gate=None,
        reproducibility_warnings=_reproducibility_warnings(
            config,
            source_revision=source_revision,
        ),
    )
    if thresholds is not None:
        gate = evaluate_benchmark_thresholds(
            benchmark_report_to_dict(report),
            thresholds,
        )
        report = replace(report, threshold_gate=gate)
    return report


def _prepare_inputs(
    corpus_path: str | Path,
    retrieval_dataset_path: str | Path,
    answer_dataset_path: str | Path,
    *,
    stages: _StageRecorder,
) -> _PreparedInputs:
    """Load, fingerprint, and extract all benchmark inputs once."""
    with stages.measure("dataset_load"):
        retrieval_dataset = load_retrieval_evaluation_dataset(retrieval_dataset_path)
        answer_dataset = load_answer_evaluation_dataset(answer_dataset_path)
        retrieval_fingerprint = fingerprint_dataset(
            retrieval_dataset_path,
            name=retrieval_dataset.name,
            schema_version=retrieval_dataset.schema_version,
            case_count=len(retrieval_dataset.cases),
        )
        answer_fingerprint = fingerprint_dataset(
            answer_dataset_path,
            name=answer_dataset.name,
            schema_version=answer_dataset.schema_version,
            case_count=len(answer_dataset.cases),
        )
    with stages.measure("corpus_fingerprint"):
        corpus_fingerprint = fingerprint_corpus(corpus_path)
    with stages.measure("document_load"):
        documents = tuple(load_documents([corpus_path]))
    if not documents:
        raise BenchmarkInputError(
            "benchmark corpus did not produce any extracted documents."
        )
    if not any(document.page_content.strip() for document in documents):
        raise BenchmarkInputError(
            "benchmark corpus did not contain any non-empty extracted text."
        )
    return _PreparedInputs(
        retrieval_dataset=retrieval_dataset,
        answer_dataset=answer_dataset,
        retrieval_fingerprint=retrieval_fingerprint,
        answer_fingerprint=answer_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
        documents=documents,
    )


def _chunk_for_benchmark(
    documents: tuple[Document, ...],
    *,
    config: BenchmarkChunkingConfig,
    embedding_service: EmbeddingService,
) -> tuple[Document, ...]:
    """Dispatch an explicit strategy while reusing the benchmark's dense model."""
    if isinstance(config, SemanticChunkingConfig):
        return tuple(
            chunk_documents_semantically(
                documents,
                embeddings=embedding_service.as_langchain_embeddings(),
                config=config,
            )
        )
    if isinstance(config, StructureAwareChunkingConfig):
        return tuple(chunk_documents_with_structure(documents, config=config))
    if isinstance(config, ChunkingConfig):
        return tuple(chunk_documents(documents, config=config))
    raise TypeError("unsupported benchmark chunking configuration.")


def _prepare_pipeline(
    *,
    config: BenchmarkConfig,
    inputs: _PreparedInputs,
    stages: _StageRecorder,
) -> _PreparedPipeline:
    """Initialize each provider once, chunk the corpus, and embed its output."""
    with stages.measure("embedding_model_load"):
        embedding_service = create_local_embedding_service(config.embedding)
    with stages.measure("chunking"):
        chunks = _chunk_for_benchmark(
            inputs.documents,
            config=config.chunking,
            embedding_service=embedding_service,
        )
    if not chunks:
        raise BenchmarkInputError(
            "benchmark corpus did not produce any non-empty chunks."
        )
    with stages.measure("dense_embedding"):
        embedded_documents = tuple(embedding_service.embed_documents(chunks))

    sparse_service = None
    sparse_vectors = None
    if config.sparse_embedding is not None:
        with stages.measure("sparse_model_load"):
            sparse_service = create_local_sparse_embedding_service(
                config.sparse_embedding
            )
        with stages.measure("sparse_embedding"):
            sparse_vectors = tuple(sparse_service.embed_documents(chunks))

    reranker = None
    if config.local_reranker is not None:
        with stages.measure("reranker_model_load"):
            reranker = create_local_reranker_service(config.local_reranker)
    with stages.measure("generation_model_load"):
        answer_generator = create_local_answer_generator(config.local_generation)
    return _PreparedPipeline(
        chunks=chunks,
        embedding_service=embedding_service,
        embedded_documents=embedded_documents,
        sparse_embedding_service=sparse_service,
        sparse_vectors=sparse_vectors,
        reranker=reranker,
        answer_generator=answer_generator,
    )


def _execute_pipeline(
    *,
    config: BenchmarkConfig,
    inputs: _PreparedInputs,
    pipeline: _PreparedPipeline,
    stages: _StageRecorder,
    clock: Clock,
) -> _ExecutionResult:
    """Index into temporary Qdrant, run both suites, then measure disk use."""
    work_directory = _prepare_work_directory(config.work_directory)
    with tempfile.TemporaryDirectory(
        prefix="rag-benchmark-",
        dir=None if work_directory is None else str(work_directory),
    ) as temporary_directory:
        qdrant_path = Path(temporary_directory) / "qdrant"
        store_config = VectorStoreConfig(
            path=qdrant_path,
            collection_name=BENCHMARK_COLLECTION_NAME,
            write_batch_size=config.write_batch_size,
            search_mode=cast(SearchMode, config.search_mode),
        )
        with LocalVectorStore(store_config) as vector_store:
            with stages.measure("vector_index"):
                indexing = vector_store.index(
                    pipeline.embedded_documents,
                    model_identifier=(pipeline.embedding_service.model_identifier),
                    sparse_vectors=pipeline.sparse_vectors,
                    sparse_model_identifier=(
                        None
                        if pipeline.sparse_embedding_service is None
                        else (pipeline.sparse_embedding_service.model_identifier)
                    ),
                )
            retriever = RetrieverService(
                pipeline.embedding_service,
                vector_store,
                pipeline.sparse_embedding_service,
            )
            retrieval_report, retrieval_timing = _run_retrieval_suite(
                inputs.retrieval_dataset,
                retriever=retriever,
                config=config,
                reranker=pipeline.reranker,
                stages=stages,
                clock=clock,
            )
            answer_report, answer_timing = _run_answer_suite(
                inputs.answer_dataset,
                retriever=retriever,
                answer_generator=pipeline.answer_generator,
                config=config,
                reranker=pipeline.reranker,
                stages=stages,
                clock=clock,
            )
        storage_bytes = _directory_size(qdrant_path)
    return _ExecutionResult(
        indexing=indexing,
        storage_bytes=storage_bytes,
        retrieval_report=retrieval_report,
        answer_report=answer_report,
        retrieval_timing=retrieval_timing,
        answer_timing=answer_timing,
    )


def _run_retrieval_suite(
    dataset: RetrievalEvaluationDataset,
    *,
    retriever: RetrieverService,
    config: BenchmarkConfig,
    reranker: RerankerService | None,
    stages: _StageRecorder,
    clock: Clock,
) -> tuple[RetrievalEvaluationReport, TimingSummary]:
    """Evaluate retrieval while preserving one latency value per case."""
    case_durations = []

    def retrieve(query: str) -> list[RetrievalResult]:
        """Run and time the configured retrieval path for one labeled query."""
        case_started = clock()
        try:
            return _retrieve_and_rerank(
                query,
                retriever=retriever,
                retrieval_config=config.retrieval,
                reranker=reranker,
                reranking_config=config.reranking,
            )
        finally:
            case_durations.append(
                _elapsed_seconds(
                    case_started,
                    clock(),
                    context="retrieval benchmark case",
                )
            )

    with stages.measure("retrieval_evaluation"):
        report = evaluate_retrieval(
            dataset,
            retrieve,
            top_k=config.final_top_k,
        )
    timing = summarize_case_timings(
        tuple(case.case_id for case in dataset.cases),
        case_durations,
    )
    return report, timing


def _run_answer_suite(
    dataset: AnswerEvaluationDataset,
    *,
    retriever: RetrieverService,
    answer_generator: AnswerGenerator,
    config: BenchmarkConfig,
    reranker: RerankerService | None,
    stages: _StageRecorder,
    clock: Clock,
) -> tuple[AnswerEvaluationReport, TimingSummary]:
    """Evaluate end-to-end answers while preserving one latency per case."""
    case_durations = []

    def answer(query: str) -> GeneratedAnswer:
        """Run and time retrieval, optional reranking, and generation."""
        case_started = clock()
        try:
            results = _retrieve_and_rerank(
                query,
                retriever=retriever,
                retrieval_config=config.retrieval,
                reranker=reranker,
                reranking_config=config.reranking,
            )
            return answer_generator.generate(
                query,
                results,
                config=config.generation,
            )
        finally:
            case_durations.append(
                _elapsed_seconds(
                    case_started,
                    clock(),
                    context="answer benchmark case",
                )
            )

    with stages.measure("answer_evaluation"):
        report = evaluate_answers(dataset, answer)
    timing = summarize_case_timings(
        tuple(case.case_id for case in dataset.cases),
        case_durations,
    )
    return report, timing


def _retrieve_and_rerank(
    query: str,
    *,
    retriever: RetrieverService,
    retrieval_config: RetrievalConfig,
    reranker: RerankerService | None,
    reranking_config: RerankingConfig | None,
) -> list[RetrievalResult]:
    results = retriever.retrieve(query, config=retrieval_config)
    if reranker is None:
        return results
    if reranking_config is None:
        raise RuntimeError("Reranker service has no result-limit configuration.")
    return reranker.rerank(query, results, config=reranking_config)


def _configuration_to_dict(config: BenchmarkConfig) -> dict[str, object]:
    """Serialize behavioral settings while omitting private local paths."""
    sparse = config.sparse_embedding
    reranker = config.local_reranker
    return {
        "chunking": _chunking_configuration_to_dict(config.chunking),
        "embedding": {
            "provider": "local_hugging_face",
            "model": config.embedding.model_name,
            "model_revision": config.embedding.model_revision,
            "device": config.embedding.device,
            "batch_size": config.embedding.batch_size,
            "normalize_embeddings": config.embedding.normalize_embeddings,
        },
        "sparse_embedding": (
            None
            if sparse is None
            else {
                "provider": "local_fastembed",
                "model": sparse.model_name,
                "batch_size": sparse.batch_size,
                "threads": sparse.threads,
            }
        ),
        "indexing": {
            "vector_store": "local_qdrant",
            "search_mode": cast(SearchMode, config.search_mode).value,
            "write_batch_size": config.write_batch_size,
            "temporary_storage": (
                "system_default" if config.work_directory is None else "custom_parent"
            ),
        },
        "retrieval": {
            "candidate_k": config.retrieval.top_k,
            "top_k": config.final_top_k,
            "score_threshold": config.retrieval.score_threshold,
            "metadata_filters": [
                {"field": item.field, "value": item.value}
                for item in config.retrieval.metadata_filters
            ],
        },
        "reranking": (
            {"enabled": False}
            if reranker is None
            else {
                "enabled": True,
                "provider": "local_sentence_transformers",
                "model": reranker.model_name,
                "model_revision": reranker.model_revision,
                "device": reranker.device,
                "batch_size": reranker.batch_size,
                "max_length_tokens": reranker.max_length,
            }
        ),
        "generation": {
            "provider": "local_hugging_face",
            "model": config.local_generation.model_name,
            "model_revision": config.local_generation.model_revision,
            "device": config.local_generation.device,
            "max_new_tokens": config.local_generation.max_new_tokens,
            "temperature": config.local_generation.temperature,
            "prompt_identifier": GROUNDED_ANSWER_PROMPT_ID,
            "max_context_characters": (config.generation.max_context_characters),
            "max_input_tokens": config.generation.max_input_tokens,
            "token_safety_margin": config.generation.token_safety_margin,
        },
    }


def _chunking_configuration_to_dict(
    config: BenchmarkChunkingConfig,
) -> dict[str, object]:
    """Serialize only controls that affect the selected chunking strategy."""
    if isinstance(config, SemanticChunkingConfig):
        return {
            "strategy": "semantic",
            "max_chunk_size_characters": config.max_chunk_size,
            "min_chunk_size_characters": config.min_chunk_size,
            "breakpoint_percentile": config.breakpoint_percentile,
            "sentence_buffer_size": config.buffer_size,
            "chunk_overlap_characters": 0,
        }
    if isinstance(config, StructureAwareChunkingConfig):
        return {
            "strategy": "structure_aware_recursive",
            "chunk_size_characters": config.chunk_size,
            "chunk_overlap_characters": config.chunk_overlap,
            "recognized_markup": ["html", "markdown"],
            "fallback": "recursive_character",
        }
    return {
        "strategy": "recursive_character",
        "chunk_size_characters": config.chunk_size,
        "chunk_overlap_characters": config.chunk_overlap,
    }


def _reproducibility_warnings(
    config: BenchmarkConfig,
    *,
    source_revision: Mapping[str, object],
) -> tuple[str, ...]:
    """Explain configuration choices that weaken exact run reproduction."""
    warnings = []
    if config.embedding.model_revision is None:
        warnings.append("Dense embedding model revision is not pinned.")
    if (
        config.local_reranker is not None
        and config.local_reranker.model_revision is None
    ):
        warnings.append("Reranker model revision is not pinned.")
    if config.local_generation.model_revision is None:
        warnings.append("Generation model revision is not pinned.")
    if config.local_generation.temperature > 0:
        warnings.append("Generation sampling is enabled; repeated answers may differ.")
    if source_revision.get("tracked_worktree_dirty") is True:
        warnings.append("Tracked source files differ from the recorded Git commit.")
    if source_revision.get("commit") is None:
        warnings.append("Git source revision could not be determined.")
    return tuple(warnings)


def _validate_run_arguments(
    *,
    config: BenchmarkConfig,
    thresholds: BenchmarkThresholdProfile | None,
    clock: Clock,
    now: Now,
) -> None:
    if not isinstance(config, BenchmarkConfig):
        raise TypeError("config must be a BenchmarkConfig.")
    if thresholds is not None and not isinstance(thresholds, BenchmarkThresholdProfile):
        raise TypeError("thresholds must be a BenchmarkThresholdProfile or None.")
    if not callable(clock):
        raise TypeError("clock must be callable.")
    if not callable(now):
        raise TypeError("now must be callable.")


def _prepare_work_directory(
    work_directory: str | Path | None,
) -> Path | None:
    if work_directory is None:
        return None
    path = Path(work_directory).expanduser().resolve()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InvalidBenchmarkConfigurationError(
            f"failed to create benchmark work directory {path}: {exc}"
        ) from exc
    if not path.is_dir():
        raise InvalidBenchmarkConfigurationError(
            f"benchmark work directory is not a directory: {path}"
        )
    return path


def _directory_size(path: Path) -> int:
    try:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    except OSError as exc:
        raise BenchmarkInputError(
            f"failed to measure temporary index storage: {exc}"
        ) from exc


def _utc_datetime(value: datetime, *, context: str) -> str:
    if not isinstance(value, datetime):
        raise BenchmarkInputError(f"{context} must return a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise BenchmarkInputError(
            f"{context} datetime must include timezone information."
        )
    return (
        value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )
