"""Define and validate cross-stage configuration for isolated benchmarks.

The configuration binds chunking, model, retrieval, reranking, generation, and
temporary-storage settings before any filesystem, provider, or vector-store
side effects occur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rag_pipeline.chunking import ChunkingConfig
from rag_pipeline.embeddings import LocalEmbeddingConfig
from rag_pipeline.exceptions import InvalidBenchmarkConfigurationError
from rag_pipeline.generation import GenerationConfig, LocalGenerationConfig
from rag_pipeline.reranking import LocalRerankerConfig, RerankingConfig
from rag_pipeline.retrieval import RetrievalConfig
from rag_pipeline.sparse_embeddings import LocalSparseEmbeddingConfig
from rag_pipeline.vector_store import SearchMode


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Behavioral settings for one isolated full-pipeline benchmark.

    The benchmark owns a temporary Qdrant collection. These settings select the
    chunking, models, retrieval, optional reranking, generation, and indexing
    behavior recorded in the reproducibility manifest. Search mode accepts its
    enum or the corresponding CLI-style string and is normalized after
    validation.
    """

    name: str
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: LocalEmbeddingConfig = field(default_factory=LocalEmbeddingConfig)
    search_mode: SearchMode | str = SearchMode.DENSE
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


def _validate_config_types(config: BenchmarkConfig) -> None:
    """Reject invalid component objects when callers bypass static typing."""
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
    """Require sparse settings exactly when hybrid search is selected."""
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
    """Validate paired reranking settings and candidate/result widths."""
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
                "retrieval candidate count must be at least the reranked result count."
            )


def _validate_work_directory(work_directory: str | Path | None) -> None:
    """Validate the optional parent path without creating it."""
    if work_directory is None:
        return
    if not isinstance(work_directory, (str, Path)):
        raise InvalidBenchmarkConfigurationError(
            "work_directory must be a string, Path, or None."
        )
    if isinstance(work_directory, str) and not work_directory.strip():
        raise InvalidBenchmarkConfigurationError("work_directory cannot be empty.")


def _non_empty_string(value: object, *, context: str) -> str:
    """Normalize a required benchmark configuration string."""
    if not isinstance(value, str) or not value.strip():
        raise InvalidBenchmarkConfigurationError(
            f"{context} must be a non-empty string."
        )
    return value.strip()


def _validate_positive_integer(value: object, *, context: str) -> None:
    """Validate a positive integer benchmark setting."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidBenchmarkConfigurationError(f"{context} must be an integer.")
    if value <= 0:
        raise InvalidBenchmarkConfigurationError(
            f"{context} must be greater than zero."
        )
