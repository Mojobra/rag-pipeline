"""Validated second-stage reranking for retrieved LangChain documents."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document

from rag_pipeline.exceptions import (
    InvalidRerankingConfigurationError,
    RerankingInputError,
    RerankingProviderError,
)
from rag_pipeline.retrieval import RetrievalResult


DEFAULT_LOCAL_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEFAULT_RERANKER_CACHE_DIR = Path(".rag_data/rerankers")


class CrossEncoderScorer(Protocol):
    """LangChain-style cross-encoder behavior required by the service."""

    def score(self, text_pairs: list[tuple[str, str]]) -> object: ...


class CrossEncoderPredictor(Protocol):
    """Subset of Sentence Transformers used by the local adapter."""

    def predict(
        self,
        inputs: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class LocalRerankerConfig:
    """Settings for the local Sentence Transformers cross-encoder."""

    model_name: str = DEFAULT_LOCAL_RERANKER_MODEL
    model_revision: str | None = None
    device: str = "cpu"
    cache_dir: str | Path | None = DEFAULT_RERANKER_CACHE_DIR
    batch_size: int = 16
    max_length: int = 512

    def __post_init__(self) -> None:
        _validate_non_empty_string("reranker model_name", self.model_name)
        _validate_non_empty_string("reranker device", self.device)
        if self.model_revision is not None:
            _validate_non_empty_string(
                "reranker model_revision",
                self.model_revision,
            )
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)):
                raise InvalidRerankingConfigurationError(
                    "reranker cache_dir must be a string, Path, or None."
                )
            if isinstance(self.cache_dir, str) and not self.cache_dir.strip():
                raise InvalidRerankingConfigurationError(
                    "reranker cache_dir must be non-empty when provided."
                )
        _validate_positive_integer("reranker batch_size", self.batch_size)
        _validate_positive_integer("reranker max_length", self.max_length)

    @property
    def model_identifier(self) -> str:
        if self.model_revision is None:
            return self.model_name
        return f"{self.model_name}@{self.model_revision}"

    @property
    def resolved_cache_dir(self) -> Path | None:
        if self.cache_dir is None:
            return None
        return Path(self.cache_dir).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class RerankingConfig:
    """Controls how many scored candidates survive reranking."""

    top_n: int = 4

    def __post_init__(self) -> None:
        _validate_positive_integer("top_n", self.top_n)


class RerankerService:
    """Score retrieved chunks jointly with the query and reorder them."""

    def __init__(
        self,
        scorer: CrossEncoderScorer,
        *,
        model_identifier: str,
    ) -> None:
        if not callable(getattr(scorer, "score", None)):
            raise TypeError("scorer must provide a score(text_pairs) method.")
        _validate_non_empty_string("model_identifier", model_identifier)
        self._scorer = scorer
        self._model_identifier = model_identifier

    @property
    def model_identifier(self) -> str:
        return self._model_identifier

    def rerank(
        self,
        query: str,
        candidates: Iterable[RetrievalResult],
        *,
        config: RerankingConfig | None = None,
    ) -> list[RetrievalResult]:
        """Return the strongest candidates ordered by cross-encoder score."""
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        normalized_query = query.strip()
        if not normalized_query:
            raise RerankingInputError("query cannot be empty.")
        if config is not None and not isinstance(config, RerankingConfig):
            raise TypeError("config must be a RerankingConfig.")

        settings = config or RerankingConfig()
        prepared_candidates = list(candidates)
        if not prepared_candidates:
            return []

        text_pairs = _prepare_text_pairs(
            normalized_query,
            prepared_candidates,
        )
        try:
            raw_scores = self._scorer.score(text_pairs)
        except Exception as exc:
            raise RerankingProviderError(
                f"Reranking model {self.model_identifier} failed."
            ) from exc

        scores = _normalize_scores(
            raw_scores,
            expected_count=len(prepared_candidates),
        )
        scored_candidates = list(
            zip(prepared_candidates, scores, range(len(scores)), strict=True)
        )
        scored_candidates.sort(
            key=lambda item: (-item[1], item[0].rank, item[2])
        )

        results = []
        for candidate, score, _ in scored_candidates[: settings.top_n]:
            results.append(
                RetrievalResult(
                    document=candidate.document,
                    score=score,
                    rank=len(results) + 1,
                    score_kind="cross_encoder",
                    retrieval_score=candidate.score,
                    retrieval_rank=candidate.rank,
                    retrieval_score_kind=candidate.score_kind,
                    reranker_model=self.model_identifier,
                )
            )
        return results


def create_local_reranker_service(
    config: LocalRerankerConfig | None = None,
) -> RerankerService:
    """Create the local cross-encoder scorer without an external API."""
    settings = config or LocalRerankerConfig()

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RerankingProviderError(
            "Local reranking requires sentence-transformers."
        ) from exc

    cache_dir = settings.resolved_cache_dir
    try:
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        model_kwargs: dict[str, object] = {
            "model_name_or_path": settings.model_name,
            "device": settings.device,
            "cache_folder": None if cache_dir is None else str(cache_dir),
            "max_length": settings.max_length,
        }
        if settings.model_revision is not None:
            model_kwargs["revision"] = settings.model_revision
        predictor = CrossEncoder(**model_kwargs)
    except Exception as exc:
        raise RerankingProviderError(
            f"Failed to initialize reranking model {settings.model_identifier}."
        ) from exc

    return RerankerService(
        _SentenceTransformersScorer(
            predictor,
            batch_size=settings.batch_size,
        ),
        model_identifier=settings.model_identifier,
    )


class _SentenceTransformersScorer:
    """Adapt Sentence Transformers to LangChain's cross-encoder score shape."""

    def __init__(
        self,
        predictor: CrossEncoderPredictor,
        *,
        batch_size: int,
    ) -> None:
        self._predictor = predictor
        self._batch_size = batch_size

    def score(self, text_pairs: list[tuple[str, str]]) -> object:
        return self._predictor.predict(
            text_pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )


def _prepare_text_pairs(
    query: str,
    candidates: list[RetrievalResult],
) -> list[tuple[str, str]]:
    text_pairs = []
    seen_ranks: set[int] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, RetrievalResult):
            raise RerankingInputError(
                f"candidates[{index}] must be a RetrievalResult."
            )
        if candidate.score_kind not in ("cosine", "rrf"):
            raise RerankingInputError(
                f"candidates[{index}] has already been reranked."
            )
        if not isinstance(candidate.document, Document):
            raise RerankingInputError(
                f"candidates[{index}] has an invalid LangChain document."
            )
        content = candidate.document.page_content.strip()
        if not content:
            raise RerankingInputError(
                f"candidates[{index}] has empty page_content."
            )
        if (
            isinstance(candidate.rank, bool)
            or not isinstance(candidate.rank, int)
            or candidate.rank <= 0
        ):
            raise RerankingInputError(
                f"candidates[{index}] has an invalid retrieval rank."
            )
        if candidate.rank in seen_ranks:
            raise RerankingInputError(
                f"candidates[{index}] duplicates retrieval rank {candidate.rank}."
            )
        if isinstance(candidate.score, bool) or not isinstance(
            candidate.score,
            Real,
        ):
            raise RerankingInputError(
                f"candidates[{index}] has a non-numeric retrieval score."
            )
        if not isfinite(float(candidate.score)):
            raise RerankingInputError(
                f"candidates[{index}] has a non-finite retrieval score."
            )
        seen_ranks.add(candidate.rank)
        text_pairs.append((query, content))
    return text_pairs


def _normalize_scores(raw_scores: object, *, expected_count: int) -> list[float]:
    if isinstance(raw_scores, (str, bytes)):
        raise RerankingProviderError(
            "Reranking provider returned scores as text."
        )
    try:
        provider_scores = list(raw_scores)  # type: ignore[arg-type]
    except TypeError as exc:
        raise RerankingProviderError(
            "Reranking provider returned non-iterable scores."
        ) from exc
    if len(provider_scores) != expected_count:
        raise RerankingProviderError(
            "Reranking provider returned "
            f"{len(provider_scores)} score(s) for {expected_count} candidate(s)."
        )

    scores = []
    for index, score in enumerate(provider_scores):
        if isinstance(score, bool) or not isinstance(score, Real):
            raise RerankingProviderError(
                f"Reranking score {index} is not a scalar number."
            )
        numeric_score = float(score)
        if not isfinite(numeric_score):
            raise RerankingProviderError(
                f"Reranking score {index} is not finite."
            )
        scores.append(numeric_score)
    return scores


def _validate_non_empty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidRerankingConfigurationError(
            f"{name} must be a non-empty string."
        )


def _validate_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidRerankingConfigurationError(f"{name} must be an integer.")
    if value <= 0:
        raise InvalidRerankingConfigurationError(
            f"{name} must be greater than zero."
        )
