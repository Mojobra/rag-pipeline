"""Test benchmark provenance, regression gates, comparisons, and CLI execution."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake import FakeListLLM
from langchain_qdrant import SparseEmbeddings, SparseVector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


class CharacterTokenizer:
    """Provide deterministic character-level prompt limits for test generation."""

    model_max_length = 2000

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
        verbose: bool = False,
    ) -> list[int]:
        special_tokens = 1 if add_special_tokens else 0
        return [0] * (len(text) + special_tokens)


class PolicyEmbeddings(Embeddings):
    """Map supported and unsupported policy text to orthogonal vectors."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0] if "receipt" in text.lower() else [0.0, 1.0]
            for text in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return (
            [1.0, 0.0]
            if "receipt" in text.lower()
            else [0.0, 1.0]
        )


class PolicySparseEmbeddings(SparseEmbeddings):
    """Emit one lexical feature for supported receipt text and queries."""

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> SparseVector:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> SparseVector:
        if "receipt" in text.lower():
            return SparseVector(indices=[1], values=[1.0])
        return SparseVector(indices=[], values=[])


def make_artifact(
    *,
    name: str = "baseline",
    hit_rate: float = 0.8,
    answer_latency: float = 0.4,
) -> dict[str, object]:
    """Build the smallest valid schema-v1 artifact used by comparison tests."""
    quality_metrics = {
        "hit_rate_at_k": hit_rate,
        "mean_precision_at_k": 0.5,
        "mean_recall_at_k": 0.75,
        "mean_reciprocal_rank_at_k": 0.7,
    }
    answer_metrics = {
        "exact_match_rate": 0.5,
        "mean_token_f1": 0.6,
        "abstention_accuracy": 0.8,
        "abstention_precision": 0.75,
        "abstention_recall": 0.75,
        "answerable_response_rate": 0.9,
        "citation_behavior_rate": 1.0,
    }
    return {
        "schema_version": 1,
        "run": {
            "name": name,
            "started_at": "2026-01-01T00:00:00.000Z",
            "finished_at": "2026-01-01T00:00:01.000Z",
        },
        "provenance": {
            "source": {
                "commit": _SHA_A,
                "tracked_worktree_dirty": False,
            },
            "corpus": {"sha256": _SHA_A},
            "datasets": {
                "retrieval": {"sha256": _SHA_B},
                "answer": {"sha256": _SHA_C},
            },
            "environment": {
                "python": {"implementation": "CPython", "version": "3.11.0"},
                "platform": {
                    "system": "TestOS",
                    "release": "1",
                    "machine": "x86_64",
                    "processor": "test-cpu",
                    "logical_cpu_count": 8,
                },
                "accelerator": {
                    "cuda_available": False,
                    "cuda_version": None,
                    "device_names": [],
                },
                "packages": {"rag-pipeline": "0.1.0"},
            },
        },
        "configuration": {
            "embedding": {"device": "cpu"},
            "retrieval": {"top_k": 4},
            "reranking": {"enabled": False},
            "generation": {"device": "cpu"},
        },
        "index": {"storage_bytes": 1024},
        "timings": {
            "total_seconds": 2.0,
            "retrieval": {
                "mean_seconds": 0.1,
                "p95_seconds": 0.2,
            },
            "answer": {
                "mean_seconds": answer_latency,
                "p95_seconds": answer_latency + 0.1,
            },
        },
        "results": {
            "retrieval": {"metrics": quality_metrics},
            "answer": {"metrics": answer_metrics},
        },
        "threshold_gate": None,
        "reproducibility_warnings": [],
    }


class BenchmarkUtilityTests(unittest.TestCase):
    """Verify deterministic identities, timing math, profiles, and comparisons."""

    def test_fingerprints_corpus_by_relative_path_and_content(self) -> None:
        from rag_pipeline.benchmark_provenance import fingerprint_corpus

        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Path(temp_dir) / "corpus"
            nested = corpus / "policies"
            nested.mkdir(parents=True)
            (nested / "expense.md").write_text(
                "Receipts are required.",
                encoding="utf-8",
            )

            first = fingerprint_corpus(corpus)
            second = fingerprint_corpus(corpus)
            (nested / "expense.md").write_text(
                "Itemized receipts are required.",
                encoding="utf-8",
            )
            changed = fingerprint_corpus(corpus)

        self.assertEqual(first, second)
        self.assertNotEqual(first.sha256, changed.sha256)
        self.assertEqual(first.file_count, 1)
        self.assertEqual(
            first.files[0].relative_path,
            "policies/expense.md",
        )
        self.assertNotIn(temp_dir, first.files[0].relative_path)

    def test_summarizes_case_latency_with_interpolated_percentiles(self) -> None:
        from rag_pipeline.benchmarking import summarize_case_timings
        from rag_pipeline.exceptions import BenchmarkInputError

        summary = summarize_case_timings(
            ("one", "two", "three"),
            (1.0, 2.0, 3.0),
        )

        self.assertEqual(summary.total_seconds, 6.0)
        self.assertEqual(summary.mean_seconds, 2.0)
        self.assertEqual(summary.p50_seconds, 2.0)
        self.assertAlmostEqual(summary.p95_seconds, 2.9)
        self.assertEqual(summary.cases[1].case_id, "two")

        with self.assertRaisesRegex(
            BenchmarkInputError,
            "same number",
        ):
            summarize_case_timings(("one",), (1.0, 2.0))

    def test_loads_strict_threshold_profile_and_rejects_bad_bounds(self) -> None:
        from rag_pipeline.benchmark_artifacts import (
            load_benchmark_threshold_profile,
        )
        from rag_pipeline.exceptions import InvalidBenchmarkThresholdsError

        valid_profile = {
            "schema_version": 1,
            "name": "local-ci-v1",
            "applies_to": {
                "corpus_sha256": _SHA_A,
                "retrieval_dataset_sha256": _SHA_B,
                "answer_dataset_sha256": _SHA_C,
                "top_k": 4,
            },
            "checks": [
                {
                    "metric": "retrieval.mean_recall_at_k",
                    "operator": "minimum",
                    "value": 0.9,
                },
                {
                    "metric": "latency.answer.p95_seconds",
                    "operator": "maximum",
                    "value": 2.0,
                },
            ],
        }
        invalid_profile = deepcopy(valid_profile)
        invalid_profile["checks"][0]["value"] = 1.1

        with tempfile.TemporaryDirectory() as temp_dir:
            valid_path = Path(temp_dir) / "valid.json"
            valid_path.write_text(
                json.dumps(valid_profile),
                encoding="utf-8",
            )
            invalid_path = Path(temp_dir) / "invalid.json"
            invalid_path.write_text(
                json.dumps(invalid_profile),
                encoding="utf-8",
            )

            profile = load_benchmark_threshold_profile(valid_path)
            with self.assertRaisesRegex(
                InvalidBenchmarkThresholdsError,
                "between 0 and 1",
            ):
                load_benchmark_threshold_profile(invalid_path)

        self.assertEqual(profile.name, "local-ci-v1")
        self.assertEqual(len(profile.checks), 2)
        self.assertEqual(len(profile.sha256), 64)

    def test_compares_quality_and_marks_environment_drift(self) -> None:
        from rag_pipeline.benchmark_artifacts import (
            compare_benchmark_artifacts,
        )

        baseline = make_artifact()
        candidate = make_artifact(
            name="candidate",
            hit_rate=0.9,
            answer_latency=0.3,
        )
        candidate["configuration"]["generation"]["model"] = "candidate-model"
        comparison = compare_benchmark_artifacts(baseline, candidate)
        hit_rate = next(
            metric
            for metric in comparison.metrics
            if metric.metric == "retrieval.hit_rate_at_k"
        )

        self.assertTrue(comparison.latency_comparable)
        self.assertEqual(
            comparison.changed_configuration_sections,
            ("generation",),
        )
        self.assertAlmostEqual(hit_rate.delta, 0.1)
        self.assertTrue(hit_rate.comparable)

        candidate["provenance"]["environment"]["platform"]["machine"] = "arm64"
        changed_environment = compare_benchmark_artifacts(
            baseline,
            candidate,
        )
        answer_latency = next(
            metric
            for metric in changed_environment.metrics
            if metric.metric == "latency.answer.mean_seconds"
        )
        changed_hit_rate = next(
            metric
            for metric in changed_environment.metrics
            if metric.metric == "retrieval.hit_rate_at_k"
        )
        self.assertFalse(changed_environment.latency_comparable)
        self.assertFalse(answer_latency.comparable)
        self.assertTrue(changed_hit_rate.comparable)

    def test_rejects_comparison_when_ground_truth_differs(self) -> None:
        from rag_pipeline.benchmark_artifacts import (
            compare_benchmark_artifacts,
        )
        from rag_pipeline.exceptions import BenchmarkComparisonError

        baseline = make_artifact()
        candidate = make_artifact(name="candidate")
        candidate["provenance"]["datasets"]["answer"]["sha256"] = "d" * 64

        with self.assertRaisesRegex(
            BenchmarkComparisonError,
            "answer dataset fingerprint",
        ):
            compare_benchmark_artifacts(baseline, candidate)

    def test_undefined_optional_metric_fails_configured_gate(self) -> None:
        from rag_pipeline.benchmark_artifacts import (
            BenchmarkThreshold,
            BenchmarkThresholdApplicability,
            BenchmarkThresholdProfile,
            evaluate_benchmark_thresholds,
        )

        artifact = make_artifact()
        artifact["results"]["answer"]["metrics"]["abstention_precision"] = None
        profile = BenchmarkThresholdProfile(
            name="requires-abstention-precision",
            applies_to=BenchmarkThresholdApplicability(
                corpus_sha256=_SHA_A,
                retrieval_dataset_sha256=_SHA_B,
                answer_dataset_sha256=_SHA_C,
                top_k=4,
            ),
            checks=(
                BenchmarkThreshold(
                    metric="answer.abstention_precision",
                    operator="minimum",
                    value=0.5,
                ),
            ),
        )

        gate = evaluate_benchmark_thresholds(artifact, profile)

        self.assertFalse(gate.passed)
        self.assertEqual(
            gate.checks[0].reason,
            "metric is undefined for this dataset",
        )

    def test_rejects_threshold_profile_for_different_ground_truth(self) -> None:
        from rag_pipeline.benchmark_artifacts import (
            BenchmarkThreshold,
            BenchmarkThresholdApplicability,
            BenchmarkThresholdProfile,
            evaluate_benchmark_thresholds,
        )
        from rag_pipeline.exceptions import InvalidBenchmarkThresholdsError

        profile = BenchmarkThresholdProfile(
            name="different-corpus",
            applies_to=BenchmarkThresholdApplicability(
                corpus_sha256="d" * 64,
                retrieval_dataset_sha256=_SHA_B,
                answer_dataset_sha256=_SHA_C,
                top_k=4,
            ),
            checks=(
                BenchmarkThreshold(
                    metric="retrieval.hit_rate_at_k",
                    operator="minimum",
                    value=0.5,
                ),
            ),
        )

        with self.assertRaisesRegex(
            InvalidBenchmarkThresholdsError,
            "corpus_sha256",
        ):
            evaluate_benchmark_thresholds(make_artifact(), profile)


class BenchmarkCliTests(unittest.TestCase):
    """Exercise indexing, both evaluators, persistence, cleanup, and exit codes."""

    def _run_benchmark(
        self,
        temp_dir: str,
        *,
        threshold_checks: list[dict[str, object]],
        output_name: str,
        hybrid: bool = False,
    ) -> tuple[int, dict[str, object], str, Path]:
        """Run the CLI with local providers and return its saved artifact."""
        from rag_pipeline.__main__ import main
        from rag_pipeline.benchmark_artifacts import load_benchmark_artifact
        from rag_pipeline.benchmark_provenance import fingerprint_corpus
        from rag_pipeline.embeddings import EmbeddingService
        from rag_pipeline.generation import AnswerGenerator
        from rag_pipeline.sparse_embeddings import SparseEmbeddingService

        root = Path(temp_dir)
        corpus = root / "corpus"
        corpus.mkdir(exist_ok=True)
        (corpus / "expenses.md").write_text(
            "Expense claims require itemized receipts.",
            encoding="utf-8",
        )
        retrieval_dataset = root / "retrieval.json"
        retrieval_dataset.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": "benchmark-retrieval-v1",
                    "cases": [
                        {
                            "id": "receipts",
                            "query": "Which receipt is required?",
                            "relevant": [
                                {
                                    "file_name": "expenses.md",
                                    "chunk_index": 0,
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        answer_dataset = root / "answers.json"
        answer_dataset.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": "benchmark-answers-v1",
                    "cases": [
                        {
                            "id": "receipts",
                            "query": "Which receipt is required?",
                            "should_abstain": False,
                            "reference_answers": [
                                "Itemized receipts are required."
                            ],
                        },
                        {
                            "id": "unsupported",
                            "query": "Who founded the company?",
                            "should_abstain": True,
                            "reference_answers": [],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        threshold_path = root / f"{output_name}-thresholds.json"
        threshold_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": f"{output_name}-gate",
                    "applies_to": {
                        "corpus_sha256": fingerprint_corpus(corpus).sha256,
                        "retrieval_dataset_sha256": sha256(
                            retrieval_dataset.read_bytes()
                        ).hexdigest(),
                        "answer_dataset_sha256": sha256(
                            answer_dataset.read_bytes()
                        ).hexdigest(),
                        "top_k": 1,
                    },
                    "checks": threshold_checks,
                }
            ),
            encoding="utf-8",
        )
        output_path = root / f"{output_name}.json"
        work_directory = root / f"{output_name}-work"
        embedding_service = EmbeddingService(
            PolicyEmbeddings(),
            model_name="benchmark-test-embedding",
        )
        answer_generator = AnswerGenerator(
            FakeListLLM(responses=["Itemized receipts are required."]),
            model_identifier="benchmark-test-llm",
            tokenizer=CharacterTokenizer(),
        )
        sparse_service = SparseEmbeddingService(
            PolicySparseEmbeddings(),
            model_name="benchmark-test-sparse",
        )
        output = io.StringIO()
        command = [
            "benchmark",
            str(corpus),
            str(retrieval_dataset),
            str(answer_dataset),
            "--name",
            output_name,
            "--model",
            "benchmark-test-embedding",
            "--generation-model",
            "benchmark-test-llm",
            "--top-k",
            "1",
            "--thresholds",
            str(threshold_path),
            "--work-dir",
            str(work_directory),
            "--output",
            str(output_path),
        ]
        if hybrid:
            command.extend(
                [
                    "--search-mode",
                    "hybrid",
                    "--sparse-model",
                    "benchmark-test-sparse",
                ]
            )

        with patch(
            "rag_pipeline.benchmarking.create_local_embedding_service",
            return_value=embedding_service,
        ) as embedding_factory:
            with patch(
                "rag_pipeline.benchmarking."
                "create_local_sparse_embedding_service",
                return_value=sparse_service,
            ) as sparse_factory:
                with patch(
                    "rag_pipeline.benchmarking.create_local_answer_generator",
                    return_value=answer_generator,
                ) as generation_factory:
                    with redirect_stdout(output):
                        exit_code = main(command)

        self.assertEqual(embedding_factory.call_count, 1)
        self.assertEqual(sparse_factory.call_count, 1 if hybrid else 0)
        self.assertEqual(generation_factory.call_count, 1)
        artifact = load_benchmark_artifact(output_path)
        return exit_code, artifact, output.getvalue(), work_directory

    def test_cli_runs_full_benchmark_and_writes_passing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            exit_code, artifact, output, work_directory = self._run_benchmark(
                temp_dir,
                threshold_checks=[
                    {
                        "metric": "retrieval.mean_recall_at_k",
                        "operator": "minimum",
                        "value": 1.0,
                    },
                    {
                        "metric": "answer.exact_match_rate",
                        "operator": "minimum",
                        "value": 1.0,
                    },
                ],
                output_name="passing",
            )
            remaining_work_files = tuple(work_directory.iterdir())
            serialized_artifact = json.dumps(artifact)

        self.assertEqual(exit_code, 0)
        self.assertEqual(artifact["schema_version"], 1)
        self.assertEqual(artifact["run"]["name"], "passing")
        self.assertEqual(
            artifact["provenance"]["corpus"]["files"][0]["relative_path"],
            "expenses.md",
        )
        self.assertEqual(artifact["index"]["chunk_count"], 1)
        self.assertGreater(artifact["index"]["storage_bytes"], 0)
        self.assertEqual(
            artifact["results"]["retrieval"]["metrics"][
                "mean_recall_at_k"
            ],
            1.0,
        )
        self.assertEqual(
            artifact["results"]["answer"]["metrics"]["exact_match_rate"],
            1.0,
        )
        self.assertTrue(artifact["threshold_gate"]["passed"])
        self.assertTrue(
            artifact["threshold_gate"]["applicability_verified"]
        )
        self.assertEqual(len(artifact["timings"]["answer"]["cases"]), 2)
        self.assertEqual(remaining_work_files, ())
        serialized_temp_path = json.dumps(temp_dir)[1:-1]
        self.assertNotIn(serialized_temp_path, serialized_artifact)
        self.assertIn("Threshold gate: PASS", output)
        self.assertIn("Artifact:", output)

    def test_cli_writes_failed_gate_and_returns_one(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            exit_code, artifact, output, _ = self._run_benchmark(
                temp_dir,
                threshold_checks=[
                    {
                        "metric": "runtime.total_seconds",
                        "operator": "maximum",
                        "value": 0.0,
                    }
                ],
                output_name="failing",
            )

        self.assertEqual(exit_code, 1)
        self.assertFalse(artifact["threshold_gate"]["passed"])
        self.assertFalse(artifact["threshold_gate"]["checks"][0]["passed"])
        self.assertIn("Threshold gate: FAIL", output)

    def test_cli_benchmarks_hybrid_index_with_sparse_model_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            exit_code, artifact, _, _ = self._run_benchmark(
                temp_dir,
                threshold_checks=[
                    {
                        "metric": "retrieval.mean_recall_at_k",
                        "operator": "minimum",
                        "value": 1.0,
                    }
                ],
                output_name="hybrid",
                hybrid=True,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            artifact["configuration"]["indexing"]["search_mode"],
            "hybrid",
        )
        self.assertEqual(
            artifact["index"]["sparse_embedding_model"],
            "benchmark-test-sparse",
        )

    def test_compare_cli_outputs_compatible_metric_deltas_as_json(self) -> None:
        from rag_pipeline.__main__ import main

        baseline = make_artifact()
        candidate = make_artifact(name="candidate", hit_rate=0.9)

        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_path = Path(temp_dir) / "baseline.json"
            baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
            candidate_path = Path(temp_dir) / "candidate.json"
            candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "compare-benchmarks",
                        str(baseline_path),
                        str(candidate_path),
                        "--output-format",
                        "json",
                    ]
                )

        comparison = json.loads(output.getvalue())
        hit_rate = next(
            metric
            for metric in comparison["metrics"]
            if metric["metric"] == "retrieval.hit_rate_at_k"
        )
        self.assertEqual(exit_code, 0)
        self.assertTrue(comparison["ground_truth_compatible"])
        self.assertAlmostEqual(hit_rate["delta"], 0.1)


if __name__ == "__main__":
    unittest.main()
