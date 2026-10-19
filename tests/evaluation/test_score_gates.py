"""Protect score-gate calibration math, validation, and CLI wiring offline."""

from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Literal
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_pipeline.evaluation.retrieval import (
    RelevantDocument,
    RetrievalEvaluationCase,
    RetrievalEvaluationDataset,
)
from rag_pipeline.evaluation.score_gates import (
    evaluate_score_gates,
    format_score_gate_comparison,
    score_gate_comparison_to_dict,
    validate_score_gate_settings,
)
from rag_pipeline.exceptions import RetrievalEvaluationInputError
from rag_pipeline.infrastructure.embeddings import EmbeddedDocument, EmbeddingService
from rag_pipeline.infrastructure.vector_store import LocalVectorStore, VectorStoreConfig
from rag_pipeline.retrieval import RetrievalConfig, RetrievalResult


def _dataset() -> RetrievalEvaluationDataset:
    return RetrievalEvaluationDataset(
        name="gate-fixture",
        schema_version=3,
        cases=(
            RetrievalEvaluationCase(
                case_id="positive",
                query="policy evidence",
                relevant_documents=(
                    RelevantDocument({"source": "first"}, "alpha"),
                    RelevantDocument({"source": "second"}, "beta"),
                ),
            ),
            RetrievalEvaluationCase(
                case_id="negative",
                query="unsupported question",
                relevant_documents=(),
                allow_empty_relevance=True,
            ),
        ),
    )


def _result(
    source: str,
    content: str,
    rank: int,
    kind: Literal["cosine", "rrf"],
) -> RetrievalResult:
    return RetrievalResult(
        document=Document(page_content=content, metadata={"source": source}),
        score=0.7 if rank == 1 else 0.1,
        rank=rank,
        score_kind=kind,
    )


class ScoreGateEvaluationTests(unittest.TestCase):
    def test_reports_positive_losses_and_negative_false_positive_reduction(
        self,
    ) -> None:
        calls: list[tuple[str, float | None]] = []

        def retrieve(query: str, config: RetrievalConfig) -> list[RetrievalResult]:
            calls.append((query, config.score_threshold))
            if query == "unsupported question":
                return (
                    []
                    if config.score_threshold is not None
                    else [_result("irrelevant", "gamma", 1, "cosine")]
                )
            if config.score_threshold == 0.8:
                return []
            results = [_result("first", "alpha", 1, "cosine")]
            if config.score_threshold is None:
                results.append(_result("second", "beta", 2, "cosine"))
            return results

        comparison = evaluate_score_gates(
            _dataset(),
            retrieve,
            top_k=2,
            search_mode="dense",
            baseline_gate=None,
            candidate_gates=(0.2, 0.8),
        )
        data = score_gate_comparison_to_dict(comparison)

        self.assertEqual(data["nominal_score_kind"], "cosine")
        self.assertEqual(data["baseline"]["score_threshold"], None)
        self.assertEqual(data["baseline"]["cases"][0]["observed_score_kind"], "cosine")
        self.assertEqual(data["candidates"][0]["score_threshold"], 0.2)
        self.assertEqual(
            data["candidates"][0]["cases"][0]["lost_selector_indices"], [1]
        )
        self.assertEqual(data["candidates"][0]["cases"][1]["false_positive_at_k"], 0.0)
        self.assertEqual(data["candidates"][0]["cases"][1]["unwanted_hit_change"], -1)
        self.assertEqual(
            data["candidates"][1]["cases"][0]["lost_selector_indices"], [0, 1]
        )
        self.assertEqual(data["candidates"][1]["cases"][0]["observed_score_kind"], None)
        self.assertEqual(len(calls), 6)
        self.assertIn("gate=0.2", format_score_gate_comparison(comparison))
        self.assertIn("score=cosine->no hit", format_score_gate_comparison(comparison))

    def test_hybrid_fallback_records_actual_cosine_kind(self) -> None:
        def retrieve(query: str, config: RetrievalConfig) -> list[RetrievalResult]:
            if query == "unsupported question" or config.score_threshold is not None:
                return []
            return [_result("first", "alpha", 1, "cosine")]

        data = score_gate_comparison_to_dict(
            evaluate_score_gates(
                _dataset(),
                retrieve,
                top_k=1,
                search_mode="hybrid",
                baseline_gate=None,
                candidate_gates=(0.2,),
            )
        )
        self.assertEqual(data["nominal_score_kind"], "rrf")
        self.assertEqual(data["baseline"]["cases"][0]["observed_score_kind"], "cosine")

    def test_lower_candidate_gate_can_add_unwanted_negative_context(self) -> None:
        def retrieve(query: str, config: RetrievalConfig) -> list[RetrievalResult]:
            if query == "unsupported question" and config.score_threshold == 0.1:
                return [_result("unwanted", "gamma", 1, "rrf")]
            return []

        data = score_gate_comparison_to_dict(
            evaluate_score_gates(
                _dataset(),
                retrieve,
                top_k=1,
                search_mode="hybrid",
                baseline_gate=0.5,
                candidate_gates=(0.1,),
            )
        )
        negative = data["candidates"][0]["cases"][1]
        self.assertEqual(negative["observed_score_kind"], "rrf")
        self.assertEqual(negative["unwanted_hit_change"], 1)
        self.assertEqual(negative["false_positive_at_k"], 1.0)

    def test_rejects_invalid_settings_before_query(self) -> None:
        calls = 0

        def retrieve(query: str, config: RetrievalConfig) -> list[RetrievalResult]:
            nonlocal calls
            calls += 1
            return []

        for candidates in ((math.nan,), (1.1,), (0.2, 0.2), ()):
            with self.subTest(candidates=candidates):
                with self.assertRaises(RetrievalEvaluationInputError):
                    evaluate_score_gates(
                        _dataset(),
                        retrieve,
                        top_k=2,
                        search_mode="dense",
                        baseline_gate=None,
                        candidate_gates=candidates,
                    )
        self.assertEqual(calls, 0)
        with self.assertRaisesRegex(RetrievalEvaluationInputError, "distinct"):
            validate_score_gate_settings(
                _dataset(),
                top_k=2,
                search_mode="dense",
                baseline_gate=0.2,
                candidate_gates=(0.2,),
            )
        with self.assertRaisesRegex(RetrievalEvaluationInputError, "schema-v3"):
            validate_score_gate_settings(
                RetrievalEvaluationDataset(
                    name="positive-only-v3",
                    schema_version=3,
                    cases=(_dataset().cases[0],),
                ),
                top_k=2,
                search_mode="dense",
                baseline_gate=None,
                candidate_gates=(0.2,),
            )

    def test_rejects_mixed_score_kinds_in_one_query(self) -> None:
        with self.assertRaisesRegex(RetrievalEvaluationInputError, "inconsistent"):
            evaluate_score_gates(
                _dataset(),
                lambda query, config: [
                    _result("first", "alpha", 1, "cosine"),
                    _result("second", "beta", 2, "rrf"),
                ],
                top_k=2,
                search_mode="hybrid",
                baseline_gate=None,
                candidate_gates=(0.2,),
            )

    def test_gate_is_applied_by_real_local_qdrant_retrieval(self) -> None:
        from rag_pipeline.retrieval import RetrieverService

        class QueryVectors(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                raise AssertionError("calibration does not index documents")

            def embed_query(self, text: str) -> list[float]:
                return [1.0, 0.0] if text == "policy evidence" else [-1.0, 0.0]

        documents = (
            EmbeddedDocument(
                document=Document(page_content="alpha", metadata={"source": "first"}),
                embedding=(1.0, 0.0),
            ),
            EmbeddedDocument(
                document=Document(page_content="beta", metadata={"source": "second"}),
                embedding=(0.0, 1.0),
            ),
        )
        with LocalVectorStore(
            VectorStoreConfig(path=None, collection_name="gate-test")
        ) as store:
            store.index(documents, model_identifier="gate-test-model")
            retriever = RetrieverService(
                EmbeddingService(QueryVectors(), model_name="gate-test-model"),
                store,
            )
            comparison = evaluate_score_gates(
                _dataset(),
                lambda query, config: retriever.retrieve(query, config=config),
                top_k=2,
                search_mode="dense",
                baseline_gate=None,
                candidate_gates=(0.2,),
            )

        report = score_gate_comparison_to_dict(comparison)
        self.assertEqual(
            report["baseline"]["cases"][0]["matched_selector_indices"], [0, 1]
        )
        self.assertEqual(
            report["candidates"][0]["cases"][0]["lost_selector_indices"], [1]
        )
        self.assertEqual(report["baseline"]["cases"][1]["false_positive_at_k"], 1.0)
        self.assertEqual(
            report["candidates"][0]["cases"][1]["false_positive_at_k"], 0.0
        )


class ScoreGateCliTests(unittest.TestCase):
    def test_answer_evaluation_can_explicitly_disable_default_gate(self) -> None:
        from rag_pipeline.cli import build_parser

        parser = build_parser()
        default = parser.parse_args(["evaluate-answer", "labels.json"])
        disabled = parser.parse_args(
            ["evaluate-answer", "labels.json", "--no-score-threshold"]
        )
        explicit = parser.parse_args(
            ["evaluate-retrieval", "labels.json", "--score-threshold", "0.2"]
        )
        self.assertEqual(default.score_threshold, 0.2)
        self.assertIsNone(disabled.score_threshold)
        self.assertEqual(explicit.score_threshold, 0.2)

    def test_compare_path_preserves_filter_and_single_pipeline(self) -> None:
        from rag_pipeline.cli import main

        calls: list[tuple[str, RetrievalConfig]] = []
        opened = 0

        class PipelineDouble:
            def retrieve(
                self, query: str, *, retrieval_config: RetrievalConfig | None = None
            ) -> list[RetrievalResult]:
                assert retrieval_config is not None
                calls.append((query, retrieval_config))
                if query == "unsupported question":
                    return []
                return [_result("first", "alpha", 1, "cosine")]

        @contextmanager
        def open_pipeline(config: object) -> Iterator[PipelineDouble]:
            nonlocal opened
            opened += 1
            yield PipelineDouble()

        dataset_payload = {
            "schema_version": 3,
            "name": "gate-cli",
            "cases": [
                {
                    "id": "positive",
                    "query": "policy evidence",
                    "relevant": [
                        {"metadata": {"source": "first"}, "content_contains": "alpha"}
                    ],
                },
                {"id": "negative", "query": "unsupported question", "relevant": []},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "labels.json"
            path.write_text(json.dumps(dataset_payload), encoding="utf-8")
            with patch(
                "rag_pipeline.application.retrieval.open_local_retrieval_pipeline",
                open_pipeline,
            ):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = main(
                        [
                            "evaluate-retrieval",
                            str(path),
                            "--top-k",
                            "1",
                            "--filter",
                            "department=finance",
                            "--compare-score-threshold",
                            "0.2",
                            "--output-format",
                            "json",
                        ]
                    )
                error = io.StringIO()
                with redirect_stderr(error), self.assertRaises(SystemExit):
                    main(
                        [
                            "evaluate-retrieval",
                            str(path),
                            "--compare-score-threshold",
                            "0.2",
                            "--compare-score-threshold",
                            "0.2",
                        ]
                    )
                rerank_error = io.StringIO()
                with redirect_stderr(rerank_error), self.assertRaises(SystemExit):
                    main(
                        [
                            "evaluate-retrieval",
                            str(path),
                            "--compare-score-threshold",
                            "0.2",
                            "--rerank",
                        ]
                    )

        data = json.loads(output.getvalue())
        self.assertEqual(data["search_mode"], "dense")
        self.assertEqual(code, 0)
        self.assertEqual(opened, 1)
        self.assertEqual(len(calls), 4)
        self.assertEqual(
            [config.score_threshold for _, config in calls], [None, None, 0.2, 0.2]
        )
        self.assertTrue(
            all(config.metadata_filters[0].value == "finance" for _, config in calls)
        )
        self.assertIn("distinct", error.getvalue())
        self.assertIn("omit --rerank", rerank_error.getvalue())


if __name__ == "__main__":
    unittest.main()
