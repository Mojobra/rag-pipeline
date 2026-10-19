"""Exercise candidate-pool comparisons and CLI validation without model I/O."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Literal
from unittest.mock import patch

from langchain_core.documents import Document

from rag_pipeline.evaluation.candidate_ranking import (
    candidate_ranking_to_dict,
    evaluate_candidate_ranking,
    format_candidate_ranking_table,
    validate_candidate_widths,
)
from rag_pipeline.evaluation.retrieval import (
    RelevantDocument,
    RetrievalEvaluationCase,
    RetrievalEvaluationDataset,
)
from rag_pipeline.exceptions import RetrievalEvaluationInputError
from rag_pipeline.retrieval import RetrievalConfig, RetrievalResult


def _dataset() -> RetrievalEvaluationDataset:
    return RetrievalEvaluationDataset(
        name="candidate-fixture",
        schema_version=3,
        cases=(
            RetrievalEvaluationCase(
                case_id="multi",
                query="both rules",
                relevant_documents=(
                    RelevantDocument({"source": "alpha"}, "alpha fact"),
                    RelevantDocument({"source": "beta"}, "beta fact"),
                ),
            ),
            RetrievalEvaluationCase(
                case_id="negative",
                query="unsupported",
                relevant_documents=(),
                allow_empty_relevance=True,
            ),
        ),
    )


def _result(
    source: str,
    rank: int,
    *,
    kind: Literal["cosine", "rrf"] = "cosine",
) -> RetrievalResult:
    return RetrievalResult(
        document=Document(
            page_content=f"{source} fact",
            metadata={"source": source},
        ),
        score=1 / rank,
        rank=rank,
        score_kind=kind,
    )


class CandidateRankingTests(unittest.TestCase):
    def test_attributes_pool_and_final_losses_for_each_width(self) -> None:
        calls: list[tuple[str, int, float | None]] = []

        def retrieve(query: str, config: RetrievalConfig) -> list[RetrievalResult]:
            calls.append((query, config.top_k, config.score_threshold))
            if query == "unsupported":
                return [_result("unwanted", 1)]
            return [
                _result(source, rank)
                for rank, source in enumerate(("alpha", "beta")[: config.top_k], 1)
            ]

        def rerank(
            query: str, candidates: list[RetrievalResult]
        ) -> list[RetrievalResult]:
            selected = candidates[-1]
            return [
                RetrievalResult(
                    document=selected.document,
                    score=2.0,
                    rank=1,
                    score_kind="cross_encoder",
                    retrieval_score=selected.score,
                    retrieval_rank=selected.rank,
                    retrieval_score_kind=selected.score_kind,
                    reranker_model="test-model",
                )
            ]

        comparison = evaluate_candidate_ranking(
            _dataset(),
            retrieve,
            rerank,
            top_k=1,
            search_mode="dense",
            candidate_widths=(1, 2),
            retrieval_config=RetrievalConfig(top_k=1, score_threshold=0.2),
        )
        data = candidate_ranking_to_dict(comparison)
        first = data["runs"][0]
        second = data["runs"][1]
        self.assertEqual(first["pool"]["metrics"]["mean_recall_at_k"], 0.5)
        self.assertEqual(second["pool"]["metrics"]["mean_recall_at_k"], 1.0)
        self.assertEqual(
            second["reranked"]["cases"][0]["lost_after_reranking_selector_indices"],
            [0],
        )
        self.assertEqual(
            second["reranked"]["cases"][0]["gained_vs_unreranked_selector_indices"],
            [1],
        )
        self.assertEqual(
            first["reranked"]["cases"][0]["first_stage_missing_selector_indices"],
            [1],
        )
        self.assertEqual(
            second["reranked"]["metrics"]["negative_no_hit_rate_at_k"], 0.0
        )
        self.assertEqual(
            calls,
            [
                ("both rules", 1, 0.2),
                ("unsupported", 1, 0.2),
                ("both rules", 2, 0.2),
                ("unsupported", 2, 0.2),
            ],
        )
        self.assertGreaterEqual(comparison.runs[0].retrieval_seconds, 0)
        self.assertIn("not a benchmark", format_candidate_ranking_table(comparison))

    def test_rejects_bad_widths_before_provider_calls(self) -> None:
        for widths in ((1,), (1, 1), (0, 2), (True, 2), (1, 0)):
            with self.subTest(widths=widths):
                with self.assertRaises(RetrievalEvaluationInputError):
                    validate_candidate_widths(
                        _dataset(),
                        top_k=1,
                        search_mode="dense",
                        candidate_widths=widths,
                    )
        with self.assertRaisesRegex(RetrievalEvaluationInputError, "schema-v3"):
            validate_candidate_widths(
                RetrievalEvaluationDataset(
                    name="v1",
                    cases=(
                        RetrievalEvaluationCase(
                            case_id="one",
                            query="one",
                            relevant_documents=(RelevantDocument({"source": "alpha"}),),
                        ),
                    ),
                ),
                top_k=1,
                search_mode="dense",
                candidate_widths=(1, 2),
            )

    def test_rejects_results_outside_pool(self) -> None:
        with self.assertRaisesRegex(RetrievalEvaluationInputError, "outside"):
            evaluate_candidate_ranking(
                _dataset(),
                lambda query, config: [_result("alpha", 1)],
                lambda query, candidates: [_result("injected", 1)],
                top_k=1,
                search_mode="dense",
                candidate_widths=(1, 2),
                retrieval_config=RetrievalConfig(),
            )

    def test_records_hybrid_score_kind_and_rejects_oversized_pool(self) -> None:
        comparison = evaluate_candidate_ranking(
            _dataset(),
            lambda query, config: [_result("alpha", 1, kind="rrf")],
            lambda query, candidates: candidates,
            top_k=1,
            search_mode="hybrid",
            candidate_widths=(1, 2),
            retrieval_config=RetrievalConfig(),
        )
        self.assertEqual(comparison.runs[0].observed_score_kinds, ("rrf", "rrf"))

        with self.assertRaisesRegex(RetrievalEvaluationInputError, "more than 1"):
            evaluate_candidate_ranking(
                _dataset(),
                lambda query, config: [_result("alpha", 1), _result("beta", 2)],
                lambda query, candidates: candidates,
                top_k=1,
                search_mode="dense",
                candidate_widths=(1, 2),
                retrieval_config=RetrievalConfig(),
            )


class CandidateRankingCliTests(unittest.TestCase):
    def test_cli_preserves_filters_and_opens_one_pipeline(self) -> None:
        from rag_pipeline.cli import main

        calls: list[RetrievalConfig] = []
        opened = 0

        class PipelineDouble:
            def retrieve_candidates(
                self, query: str, *, retrieval_config: RetrievalConfig
            ) -> list[RetrievalResult]:
                calls.append(retrieval_config)
                return [_result("alpha", 1)] if query == "known" else []

            def rerank_candidates(
                self, query: str, candidates: list[RetrievalResult]
            ) -> list[RetrievalResult]:
                return candidates

        @contextmanager
        def open_pipeline(config: object):
            nonlocal opened
            opened += 1
            yield PipelineDouble()

        payload = {
            "schema_version": 3,
            "name": "cli-fixture",
            "cases": [
                {
                    "id": "known",
                    "query": "known",
                    "relevant": [
                        {"metadata": {"source": "alpha"}, "content_contains": "alpha"}
                    ],
                },
                {"id": "unknown", "query": "unknown", "relevant": []},
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "labels.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
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
                            "--rerank",
                            "--top-k",
                            "1",
                            "--candidate-k",
                            "1",
                            "--compare-candidate-k",
                            "2",
                            "--filter",
                            "department=finance",
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
                            "--compare-candidate-k",
                            "2",
                        ]
                    )
                conflict_error = io.StringIO()
                with redirect_stderr(conflict_error), self.assertRaises(SystemExit):
                    main(
                        [
                            "evaluate-retrieval",
                            str(path),
                            "--rerank",
                            "--compare-candidate-k",
                            "2",
                            "--compare-score-threshold",
                            "0.2",
                        ]
                    )
        self.assertEqual(code, 0)
        self.assertEqual(opened, 1)
        self.assertEqual(len(calls), 4)
        self.assertEqual([config.top_k for config in calls], [1, 1, 2, 2])
        self.assertTrue(
            all(config.metadata_filters[0].value == "finance" for config in calls)
        )
        self.assertEqual(json.loads(output.getvalue())["runs"][1]["candidate_k"], 2)
        self.assertIn("requires --rerank", error.getvalue())
        self.assertIn("cannot be combined", conflict_error.getvalue())
