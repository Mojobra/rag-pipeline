"""Run isolated full-pipeline RAG benchmarks and persist result artifacts.

The module rebuilds an exact temporary index, executes the existing retrieval
and answer evaluators through shared LangChain services, captures wall-clock
timings, and assembles a portable report. Artifact validation and comparison
live separately in :mod:`rag_pipeline.benchmark_artifacts`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
from math import ceil, floor, isfinite
import os
from pathlib import Path
import tempfile
import time
from typing import TypeAlias

from langchain_core.documents import Document

from rag_pipeline.answer_evaluation import (
    AnswerEvaluationDataset,
    AnswerEvaluationReport,
    answer_evaluation_to_dict,
    evaluate_answers,
    load_answer_evaluation_dataset,
)
from rag_pipeline.benchmark_artifacts import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkThresholdProfile,
    ThresholdGateResult,
    evaluate_benchmark_thresholds,
    threshold_gate_to_dict,
    validate_benchmark_threshold_applicability,
)
from rag_pipeline.benchmark_provenance import (
    CorpusFingerprint,
    DatasetFingerprint,
    corpus_fingerprint_to_dict,
    dataset_fingerprint_to_dict,
    fingerprint_corpus,
    fingerprint_dataset,
    read_source_revision,
    runtime_environment,
)
from rag_pipeline.chunking import ChunkingConfig, chunk_documents
from rag_pipeline.embeddings import (
    EmbeddedDocument,
    EmbeddingService,
    LocalEmbeddingConfig,
    create_local_embedding_service,
)
from rag_pipeline.exceptions import (
    BenchmarkInputError,
    InvalidBenchmarkConfigurationError,
)
from rag_pipeline.generation import (
    GROUNDED_ANSWER_PROMPT_ID,
    AnswerGenerator,
    GeneratedAnswer,
    GenerationConfig,
    LocalGenerationConfig,
    create_local_answer_generator,
)
from rag_pipeline.ingestion import load_documents
from rag_pipeline.reranking import (
    LocalRerankerConfig,
    RerankerService,
    RerankingConfig,
    create_local_reranker_service,
)
from rag_pipeline.retrieval import RetrievalConfig, RetrievalResult, RetrieverService
from rag_pipeline.retrieval_evaluation import (
    RetrievalEvaluationDataset,
    RetrievalEvaluationReport,
    evaluate_retrieval,
    load_retrieval_evaluation_dataset,
    retrieval_evaluation_to_dict,
)
from rag_pipeline.sparse_embeddings import (
    LocalSparseEmbeddingConfig,
    SparseEmbeddingService,
    SparseEmbeddingVector,
    create_local_sparse_embedding_service,
)
from rag_pipeline.vector_store import (
    IndexingResult,
    LocalVectorStore,
    SearchMode,
    VectorStoreConfig,
)


BENCHMARK_COLLECTION_NAME = "benchmark_documents"

Clock: TypeAlias = Callable[[], float]
Now: TypeAlias = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class CaseTiming:
    """Wall-clock duration associated with one evaluation case."""

    case_id: str
    seconds: float


@dataclass(frozen=True, slots=True)
class TimingSummary:
    """Aggregate and per-case latency for one measured evaluation pass.

    Percentiles use linear interpolation. They support like-for-like regression
    checks, but one pass is not a statistically rigorous service load test.
    """

    total_seconds: float
    mean_seconds: float
    p50_seconds: float
    p95_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    cases: tuple[CaseTiming, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Behavioral settings for one isolated full-pipeline benchmark.

    The benchmark owns a temporary Qdrant collection. These settings select the
    chunking, models, retrieval, optional reranking, generation, and indexing
    behavior that becomes part of the saved reproducibility manifest.
    """

    name: str
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: LocalEmbeddingConfig = field(
        default_factory=LocalEmbeddingConfig
    )
    search_mode: SearchMode = SearchMode.DENSE
    sparse_embedding: LocalSparseEmbeddingConfig | None = None
    write_batch_size: int = 64
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    local_reranker: LocalRerankerConfig | None = None
    reranking: RerankingConfig | None = None
    local_generation: LocalGenerationConfig = field(
        default_factory=LocalGenerationConfig
    )
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    work_directory: str | Path | None = None

    def __post_init__(self) -> None:
        """Validate cross-stage invariants before filesystem or model work."""
        normalized_name = _non_empty_string(
            self.name,
            context="benchmark name",
            error_type=InvalidBenchmarkConfigurationError,
        )
        try:
            search_mode = (
                self.search_mode
                if isinstance(self.search_mode, SearchMode)
                else SearchMode(self.search_mode)
            )
        except (TypeError, ValueError) as exc:
            raise InvalidBenchmarkConfigurationError(
                "search_mode must be 'dense' or 'hybrid'."
            ) from exc
        _validate_config_types(self)
        _validate_positive_integer(
            self.write_batch_size,
            context="write_batch_size",
        )
        _validate_search_mode_contract(self, search_mode=search_mode)
        _validate_reranking_contract(self)
        _validate_work_directory(self.work_directory)
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "search_mode", search_mode)

    @property
    def final_top_k(self) -> int:
        """Return the result cutoff shared by both quality evaluators."""
        if self.reranking is None:
            return self.retrieval.top_k
        return self.reranking.top_n


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Complete in-memory result before conversion to a JSON artifact."""

    name: str
    started_at: str
    finished_at: str
    total_seconds: float
    corpus: CorpusFingerprint
    retrieval_dataset: DatasetFingerprint
    answer_dataset: DatasetFingerprint
    source_revision: Mapping[str, object]
    environment: Mapping[str, object]
    configuration: Mapping[str, object]
    index: Mapping[str, object]
    stage_seconds: Mapping[str, float]
    retrieval_timing: TimingSummary
    answer_timing: TimingSummary
    retrieval_report: RetrievalEvaluationReport
    answer_report: AnswerEvaluationReport
    threshold_gate: ThresholdGateResult | None
    reproducibility_warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PreparedInputs:
    """Validated datasets, fingerprints, documents, and chunks for one run."""

    retrieval_dataset: RetrievalEvaluationDataset
    answer_dataset: AnswerEvaluationDataset
    retrieval_fingerprint: DatasetFingerprint
    answer_fingerprint: DatasetFingerprint
    corpus_fingerprint: CorpusFingerprint
    documents: tuple[Document, ...]
    chunks: tuple[Document, ...]


@dataclass(frozen=True, slots=True)
class _PreparedPipeline:
    """Loaded shared model services and corpus vectors used by both suites."""

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


class _StageRecorder:
    """Record one non-overlapping wall-clock duration per named pipeline stage."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._durations: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Measure a stage even when its body raises, rejecting duplicate names."""
        if name in self._durations:
            raise RuntimeError(f"benchmark stage {name!r} was measured twice.")
        started = self._clock()
        try:
            yield
        finally:
            self._durations[name] = _elapsed_seconds(
                started,
                self._clock(),
                context=f"benchmark stage {name!r}",
            )

    @property
    def durations(self) -> dict[str, float]:
        return dict(self._durations)


def summarize_case_timings(
    case_ids: Sequence[str],
    durations: Sequence[float],
) -> TimingSummary:
    """Summarize one measured duration per case in dataset order.

    Inputs must be non-empty, aligned, finite, and non-negative. The diagnostic
    case list preserves dataset order while percentiles sort only numeric values.
    """
    if isinstance(case_ids, (str, bytes)) or isinstance(
        durations, (str, bytes)
    ):
        raise BenchmarkInputError("case_ids and durations must be sequences.")
    case_id_values = tuple(case_ids)
    duration_values = tuple(durations)
    if not case_id_values:
        raise BenchmarkInputError("at least one case timing is required.")
    if len(case_id_values) != len(duration_values):
        raise BenchmarkInputError(
            "case_ids and durations must contain the same number of values."
        )

    timings = tuple(
        CaseTiming(
            case_id=_non_empty_string(
                case_id,
                context=f"case_ids[{index}]",
                error_type=BenchmarkInputError,
            ),
            seconds=_duration(duration, context=f"durations[{index}]"),
        )
        for index, (case_id, duration) in enumerate(
            zip(case_id_values, duration_values, strict=True)
        )
    )
    seconds = tuple(timing.seconds for timing in timings)
    total = sum(seconds)
    return TimingSummary(
        total_seconds=total,
        mean_seconds=total / len(seconds),
        p50_seconds=_percentile(seconds, 0.50),
        p95_seconds=_percentile(seconds, 0.95),
        minimum_seconds=min(seconds),
        maximum_seconds=max(seconds),
        cases=timings,
    )


def validate_benchmark_output_path(
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate a JSON artifact destination without modifying the filesystem.

    Existing files are rejected unless overwrite is explicitly enabled. Callers
    should run this before model work; the writer repeats the check before I/O.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("benchmark output path must be a string or pathlib.Path.")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")
    resolved_path = Path(path).expanduser().resolve()
    if resolved_path.suffix.lower() != ".json":
        raise InvalidBenchmarkConfigurationError(
            "benchmark output path must use a .json extension."
        )
    if resolved_path.exists():
        if resolved_path.is_dir():
            raise InvalidBenchmarkConfigurationError(
                f"benchmark output path is a directory: {resolved_path}"
            )
        if not overwrite:
            raise InvalidBenchmarkConfigurationError(
                f"benchmark output already exists: {resolved_path}; "
                "choose another path or enable overwrite."
            )
    return resolved_path


def run_benchmark(
    corpus_path: str | Path,
    retrieval_dataset_path: str | Path,
    answer_dataset_path: str | Path,
    *,
    config: BenchmarkConfig,
    thresholds: BenchmarkThresholdProfile | None = None,
    clock: Clock = time.perf_counter,
    now: Now = lambda: datetime.now(timezone.utc),
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
        config=config,
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
            "chunk_count": len(inputs.chunks),
            "point_count": execution.indexing.total_count,
            "embedding_model": execution.indexing.embedding_model,
            "embedding_dimension": execution.indexing.embedding_dimension,
            "sparse_embedding_model": (
                execution.indexing.sparse_embedding_model
            ),
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


def benchmark_report_to_dict(report: BenchmarkReport) -> dict[str, object]:
    """Serialize a report to the stable schema-v1 artifact contract."""
    if not isinstance(report, BenchmarkReport):
        raise TypeError("report must be a BenchmarkReport.")
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "run": {
            "name": report.name,
            "started_at": report.started_at,
            "finished_at": report.finished_at,
        },
        "provenance": {
            "source": dict(report.source_revision),
            "corpus": corpus_fingerprint_to_dict(report.corpus),
            "datasets": {
                "retrieval": dataset_fingerprint_to_dict(
                    report.retrieval_dataset
                ),
                "answer": dataset_fingerprint_to_dict(
                    report.answer_dataset
                ),
            },
            "environment": _json_copy(report.environment),
        },
        "configuration": _json_copy(report.configuration),
        "index": _json_copy(report.index),
        "timings": {
            "total_seconds": report.total_seconds,
            "stages": dict(report.stage_seconds),
            "retrieval": _timing_to_dict(report.retrieval_timing),
            "answer": _timing_to_dict(report.answer_timing),
        },
        "results": {
            "retrieval": retrieval_evaluation_to_dict(
                report.retrieval_report
            ),
            "answer": answer_evaluation_to_dict(report.answer_report),
        },
        "threshold_gate": (
            None
            if report.threshold_gate is None
            else threshold_gate_to_dict(report.threshold_gate)
        ),
        "reproducibility_warnings": list(report.reproducibility_warnings),
    }


def write_benchmark_report(
    report: BenchmarkReport,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically persist a benchmark artifact as UTF-8 JSON.

    Parent directories are created. Existing artifacts are preserved unless
    overwrite is explicitly enabled. The function returns the resolved path
    after the temporary file has been flushed and replaced successfully.
    """
    resolved_path = validate_benchmark_output_path(path, overwrite=overwrite)
    try:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InvalidBenchmarkConfigurationError(
            f"failed to create benchmark output directory "
            f"{resolved_path.parent}: {exc}"
        ) from exc
    payload = json.dumps(
        benchmark_report_to_dict(report),
        ensure_ascii=True,
        indent=2,
        allow_nan=False,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{resolved_path.name}.",
            suffix=".tmp",
            dir=resolved_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_file.write(f"{payload}\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        if resolved_path.exists() and not overwrite:
            raise InvalidBenchmarkConfigurationError(
                f"benchmark output already exists: {resolved_path}; "
                "choose another path or enable overwrite."
            )
        os.replace(temporary_path, resolved_path)
        temporary_path = None
    except OSError as exc:
        raise InvalidBenchmarkConfigurationError(
            f"failed to write benchmark artifact {resolved_path}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return resolved_path


def format_benchmark_summary(report: BenchmarkReport) -> str:
    """Render the highest-signal quality, latency, and gate outcomes."""
    if not isinstance(report, BenchmarkReport):
        raise TypeError("report must be a BenchmarkReport.")
    retrieval = report.retrieval_report.aggregate
    answer = report.answer_report.aggregate
    gate_status = (
        "not configured"
        if report.threshold_gate is None
        else ("PASS" if report.threshold_gate.passed else "FAIL")
    )
    return "\n".join(
        (
            f"Benchmark: {report.name}",
            (
                f"Corpus: {report.corpus.file_count} file(s), "
                f"{report.index['chunk_count']} chunk(s)"
            ),
            (
                f"Retrieval: Hit@{report.retrieval_report.top_k}="
                f"{retrieval.hit_rate_at_k:.3f}, "
                f"Recall@{report.retrieval_report.top_k}="
                f"{retrieval.mean_recall_at_k:.3f}, "
                f"p95={report.retrieval_timing.p95_seconds:.4f}s"
            ),
            (
                f"Answer: EM={_optional_ratio(answer.exact_match_rate)}, "
                f"F1={_optional_ratio(answer.mean_token_f1)}, "
                f"abstention_accuracy={answer.abstention_accuracy:.3f}, "
                f"p95={report.answer_timing.p95_seconds:.4f}s"
            ),
            f"Total runtime: {report.total_seconds:.4f}s",
            f"Threshold gate: {gate_status}",
        )
    )


def _prepare_inputs(
    corpus_path: str | Path,
    retrieval_dataset_path: str | Path,
    answer_dataset_path: str | Path,
    *,
    config: BenchmarkConfig,
    stages: _StageRecorder,
) -> _PreparedInputs:
    """Load, fingerprint, extract, and chunk all benchmark inputs once."""
    with stages.measure("dataset_load"):
        retrieval_dataset = load_retrieval_evaluation_dataset(
            retrieval_dataset_path
        )
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
    with stages.measure("chunking"):
        chunks = tuple(chunk_documents(documents, config=config.chunking))
    if not chunks:
        raise BenchmarkInputError(
            "benchmark corpus did not produce any non-empty chunks."
        )
    return _PreparedInputs(
        retrieval_dataset=retrieval_dataset,
        answer_dataset=answer_dataset,
        retrieval_fingerprint=retrieval_fingerprint,
        answer_fingerprint=answer_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
        documents=documents,
        chunks=chunks,
    )


def _prepare_pipeline(
    *,
    config: BenchmarkConfig,
    inputs: _PreparedInputs,
    stages: _StageRecorder,
) -> _PreparedPipeline:
    """Initialize each configured provider once and embed the corpus."""
    with stages.measure("embedding_model_load"):
        embedding_service = create_local_embedding_service(config.embedding)
    with stages.measure("dense_embedding"):
        embedded_documents = tuple(
            embedding_service.embed_documents(inputs.chunks)
        )

    sparse_service = None
    sparse_vectors = None
    if config.sparse_embedding is not None:
        with stages.measure("sparse_model_load"):
            sparse_service = create_local_sparse_embedding_service(
                config.sparse_embedding
            )
        with stages.measure("sparse_embedding"):
            sparse_vectors = tuple(
                sparse_service.embed_documents(inputs.chunks)
            )

    reranker = None
    if config.local_reranker is not None:
        with stages.measure("reranker_model_load"):
            reranker = create_local_reranker_service(config.local_reranker)
    with stages.measure("generation_model_load"):
        answer_generator = create_local_answer_generator(
            config.local_generation
        )
    return _PreparedPipeline(
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
            search_mode=config.search_mode,
        )
        with LocalVectorStore(store_config) as vector_store:
            with stages.measure("vector_index"):
                indexing = vector_store.index(
                    pipeline.embedded_documents,
                    model_identifier=(
                        pipeline.embedding_service.model_identifier
                    ),
                    sparse_vectors=pipeline.sparse_vectors,
                    sparse_model_identifier=(
                        None
                        if pipeline.sparse_embedding_service is None
                        else (
                            pipeline.sparse_embedding_service.model_identifier
                        )
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
        "chunking": {
            "strategy": "recursive_character",
            "chunk_size_characters": config.chunking.chunk_size,
            "chunk_overlap_characters": config.chunking.chunk_overlap,
        },
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
            "search_mode": config.search_mode.value,
            "write_batch_size": config.write_batch_size,
            "temporary_storage": (
                "system_default"
                if config.work_directory is None
                else "custom_parent"
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
            "max_context_characters": (
                config.generation.max_context_characters
            ),
            "max_input_tokens": config.generation.max_input_tokens,
            "token_safety_margin": config.generation.token_safety_margin,
        },
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
        warnings.append(
            "Generation sampling is enabled; repeated answers may differ."
        )
    if source_revision.get("tracked_worktree_dirty") is True:
        warnings.append(
            "Tracked source files differ from the recorded Git commit."
        )
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
    if thresholds is not None and not isinstance(
        thresholds, BenchmarkThresholdProfile
    ):
        raise TypeError(
            "thresholds must be a BenchmarkThresholdProfile or None."
        )
    if not callable(clock):
        raise TypeError("clock must be callable.")
    if not callable(now):
        raise TypeError("now must be callable.")


def _validate_config_types(config: BenchmarkConfig) -> None:
    expected_types = (
        ("chunking", config.chunking, ChunkingConfig),
        ("embedding", config.embedding, LocalEmbeddingConfig),
        ("retrieval", config.retrieval, RetrievalConfig),
        ("local_generation", config.local_generation, LocalGenerationConfig),
        ("generation", config.generation, GenerationConfig),
    )
    for field_name, value, expected_type in expected_types:
        if not isinstance(value, expected_type):
            raise TypeError(f"{field_name} must be a {expected_type.__name__}.")


def _validate_search_mode_contract(
    config: BenchmarkConfig,
    *,
    search_mode: SearchMode,
) -> None:
    if search_mode == SearchMode.HYBRID:
        if not isinstance(
            config.sparse_embedding,
            LocalSparseEmbeddingConfig,
        ):
            raise InvalidBenchmarkConfigurationError(
                "hybrid benchmarks require sparse embedding settings."
            )
    elif config.sparse_embedding is not None:
        raise InvalidBenchmarkConfigurationError(
            "sparse embedding settings are only valid for hybrid benchmarks."
        )


def _validate_reranking_contract(config: BenchmarkConfig) -> None:
    if (config.local_reranker is None) != (config.reranking is None):
        raise InvalidBenchmarkConfigurationError(
            "reranker model and result settings must be enabled together."
        )
    if config.local_reranker is not None and not isinstance(
        config.local_reranker,
        LocalRerankerConfig,
    ):
        raise TypeError("local_reranker must be a LocalRerankerConfig or None.")
    if config.reranking is not None:
        if not isinstance(config.reranking, RerankingConfig):
            raise TypeError("reranking must be a RerankingConfig or None.")
        if config.retrieval.top_k < config.reranking.top_n:
            raise InvalidBenchmarkConfigurationError(
                "retrieval candidate count must be at least the reranked "
                "result count."
            )


def _validate_work_directory(work_directory: str | Path | None) -> None:
    if work_directory is None:
        return
    if not isinstance(work_directory, (str, Path)):
        raise InvalidBenchmarkConfigurationError(
            "work_directory must be a string, Path, or None."
        )
    if isinstance(work_directory, str) and not work_directory.strip():
        raise InvalidBenchmarkConfigurationError(
            "work_directory cannot be empty."
        )


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
        return sum(
            item.stat().st_size
            for item in path.rglob("*")
            if item.is_file()
        )
    except OSError as exc:
        raise BenchmarkInputError(
            f"failed to measure temporary index storage: {exc}"
        ) from exc


def _timing_to_dict(summary: TimingSummary) -> dict[str, object]:
    return {
        "total_seconds": summary.total_seconds,
        "mean_seconds": summary.mean_seconds,
        "p50_seconds": summary.p50_seconds,
        "p95_seconds": summary.p95_seconds,
        "minimum_seconds": summary.minimum_seconds,
        "maximum_seconds": summary.maximum_seconds,
        "cases": [
            {"id": case.case_id, "seconds": case.seconds}
            for case in summary.cases
        ],
    }


def _elapsed_seconds(started: float, finished: float, *, context: str) -> float:
    if isinstance(started, bool) or not isinstance(started, (int, float)):
        raise BenchmarkInputError(f"{context} start time must be numeric.")
    if isinstance(finished, bool) or not isinstance(finished, (int, float)):
        raise BenchmarkInputError(f"{context} finish time must be numeric.")
    elapsed = float(finished) - float(started)
    if not isfinite(elapsed) or elapsed < 0:
        raise BenchmarkInputError(
            f"{context} clock must produce finite monotonic values."
        )
    return elapsed


def _duration(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkInputError(f"{context} must be numeric seconds.")
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0:
        raise BenchmarkInputError(
            f"{context} must be finite non-negative seconds."
        )
    return normalized


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower_index = floor(position)
    upper_index = ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return (
        ordered[lower_index] * (1.0 - weight)
        + ordered[upper_index] * weight
    )


def _utc_datetime(value: datetime, *, context: str) -> str:
    if not isinstance(value, datetime):
        raise BenchmarkInputError(f"{context} must return a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise BenchmarkInputError(
            f"{context} datetime must include timezone information."
        )
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _non_empty_string(
    value: object,
    *,
    context: str,
    error_type: type[ValueError],
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{context} must be a non-empty string.")
    return value.strip()


def _validate_positive_integer(value: object, *, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidBenchmarkConfigurationError(
            f"{context} must be an integer."
        )
    if value <= 0:
        raise InvalidBenchmarkConfigurationError(
            f"{context} must be greater than zero."
        )


def _json_copy(value: Mapping[str, object]) -> dict[str, object]:
    return json.loads(
        json.dumps(value, ensure_ascii=True, allow_nan=False)
    )


def _optional_ratio(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"
