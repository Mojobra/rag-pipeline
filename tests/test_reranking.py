from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from rag_pipeline.exceptions import (  # noqa: E402
    InvalidRerankingConfigurationError,
    RerankingInputError,
    RerankingProviderError,
)
from rag_pipeline.reranking import (  # noqa: E402
    LocalRerankerConfig,
    RerankerService,
    RerankingConfig,
    create_local_reranker_service,
)
from rag_pipeline.retrieval import RetrievalResult  # noqa: E402


class FakeCrossEncoderScorer:
    def __init__(
        self,
        scores: object,
        *,
        error: Exception | None = None,
    ) -> None:
        self.scores = scores
        self.error = error
        self.requests: list[list[tuple[str, str]]] = []

    def score(self, text_pairs: list[tuple[str, str]]) -> object:
        self.requests.append(text_pairs)
        if self.error is not None:
            raise self.error
        return self.scores


def make_result(
    content: str,
    *,
    rank: int,
    score: float,
    score_kind: str = "cosine",
    source: str | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        document=Document(
            page_content=content,
            metadata={"source": source or f"document-{rank}.txt"},
        ),
        score=score,
        rank=rank,
        score_kind=score_kind,  # type: ignore[arg-type]
    )


class RerankerServiceTests(unittest.TestCase):
    def test_reranks_top_n_and_preserves_first_stage_provenance(self) -> None:
        scorer = FakeCrossEncoderScorer([0.1, 0.9, 0.9])
        service = RerankerService(
            scorer,
            model_identifier="test-cross-encoder",
        )
        candidates = [
            make_result("Broad policy overview.", rank=1, score=0.95),
            make_result("Receipts are optional.", rank=3, score=0.6),
            make_result(
                "Receipts are required for expenses.",
                rank=2,
                score=0.7,
                score_kind="rrf",
            ),
        ]

        results = service.rerank(
            "Are receipts required?",
            candidates,
            config=RerankingConfig(top_n=2),
        )

        self.assertEqual(
            scorer.requests,
            [
                [
                    ("Are receipts required?", "Broad policy overview."),
                    (
                        "Are receipts required?",
                        "Receipts are optional.",
                    ),
                    (
                        "Are receipts required?",
                        "Receipts are required for expenses.",
                    ),
                ]
            ],
        )
        self.assertEqual(
            [result.document.page_content for result in results],
            [
                "Receipts are required for expenses.",
                "Receipts are optional.",
            ],
        )
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertEqual([result.score for result in results], [0.9, 0.9])
        self.assertEqual(results[0].score_kind, "cross_encoder")
        self.assertEqual(results[0].retrieval_rank, 2)
        self.assertEqual(results[0].retrieval_score, 0.7)
        self.assertEqual(results[0].retrieval_score_kind, "rrf")
        self.assertEqual(results[0].reranker_model, "test-cross-encoder")

    def test_empty_candidates_do_not_call_the_model(self) -> None:
        scorer = FakeCrossEncoderScorer([])
        service = RerankerService(scorer, model_identifier="test-model")

        self.assertEqual(service.rerank("Question", []), [])
        self.assertEqual(scorer.requests, [])

    def test_rejects_invalid_configuration_and_inputs(self) -> None:
        invalid_configs = [
            ({"top_n": 0}, "top_n must be greater than zero"),
            ({"top_n": True}, "top_n must be an integer"),
        ]
        for settings, message in invalid_configs:
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(
                    InvalidRerankingConfigurationError,
                    message,
                ):
                    RerankingConfig(**settings)

        service = RerankerService(
            FakeCrossEncoderScorer([1.0]),
            model_identifier="test-model",
        )
        with self.assertRaisesRegex(RerankingInputError, "query cannot be empty"):
            service.rerank(" ", [])
        with self.assertRaisesRegex(RerankingInputError, "RetrievalResult"):
            service.rerank("Question", ["not a result"])  # type: ignore[list-item]
        with self.assertRaisesRegex(RerankingInputError, "already been reranked"):
            service.rerank(
                "Question",
                [
                    RetrievalResult(
                        document=Document(page_content="Content"),
                        score=1.0,
                        rank=1,
                        score_kind="cross_encoder",
                    )
                ],
            )

    def test_rejects_duplicate_ranks_and_invalid_candidate_values(self) -> None:
        service = RerankerService(
            FakeCrossEncoderScorer([1.0, 0.5]),
            model_identifier="test-model",
        )
        duplicate_ranks = [
            make_result("First", rank=1, score=0.9),
            make_result("Second", rank=1, score=0.8),
        ]
        with self.assertRaisesRegex(RerankingInputError, "duplicates retrieval rank"):
            service.rerank("Question", duplicate_ranks)

        with self.assertRaisesRegex(RerankingInputError, "non-finite retrieval"):
            service.rerank(
                "Question",
                [make_result("Content", rank=1, score=math.nan)],
            )

    def test_wraps_model_errors_and_rejects_invalid_scores(self) -> None:
        candidate = make_result("Content", rank=1, score=0.8)
        failing_service = RerankerService(
            FakeCrossEncoderScorer([], error=RuntimeError("inference failed")),
            model_identifier="test-model",
        )
        with self.assertRaisesRegex(RerankingProviderError, "test-model failed"):
            failing_service.rerank("Question", [candidate])

        invalid_responses = [
            ([], "0 score.*1 candidate"),
            (["high"], "not a scalar number"),
            ([math.inf], "not finite"),
            (1.0, "non-iterable"),
        ]
        for response, message in invalid_responses:
            with self.subTest(message=message):
                service = RerankerService(
                    FakeCrossEncoderScorer(response),
                    model_identifier="test-model",
                )
                with self.assertRaisesRegex(RerankingProviderError, message):
                    service.rerank("Question", [candidate])

    def test_validates_local_model_configuration(self) -> None:
        invalid_settings = [
            ({"model_name": " "}, "model_name must be a non-empty"),
            ({"model_revision": ""}, "model_revision must be a non-empty"),
            ({"device": " "}, "device must be a non-empty"),
            ({"cache_dir": ""}, "cache_dir must be non-empty"),
            ({"batch_size": 0}, "batch_size must be greater than zero"),
            ({"max_length": True}, "max_length must be an integer"),
        ]
        for settings, message in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(
                    InvalidRerankingConfigurationError,
                    message,
                ):
                    LocalRerankerConfig(**settings)

        config = LocalRerankerConfig(
            model_name="test-model",
            model_revision="abc123",
        )
        self.assertEqual(config.model_identifier, "test-model@abc123")

    def test_local_factory_configures_and_adapts_sentence_transformers(self) -> None:
        predictor = Mock()
        predictor.predict.return_value = [0.75]

        with tempfile.TemporaryDirectory() as temp_dir:
            config = LocalRerankerConfig(
                model_name="test-model",
                model_revision="abc123",
                device="cpu",
                cache_dir=temp_dir,
                batch_size=8,
                max_length=256,
            )
            with patch(
                "sentence_transformers.CrossEncoder",
                return_value=predictor,
            ) as factory:
                service = create_local_reranker_service(config)
                results = service.rerank(
                    "Question",
                    [make_result("Candidate", rank=1, score=0.5)],
                    config=RerankingConfig(top_n=1),
                )

        factory.assert_called_once_with(
            model_name_or_path="test-model",
            device="cpu",
            cache_folder=str(Path(temp_dir).resolve()),
            max_length=256,
            revision="abc123",
        )
        predictor.predict.assert_called_once_with(
            [("Question", "Candidate")],
            batch_size=8,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self.assertEqual(results[0].score, 0.75)
        self.assertEqual(service.model_identifier, "test-model@abc123")


if __name__ == "__main__":
    unittest.main()
