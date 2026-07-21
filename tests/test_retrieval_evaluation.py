"""Test retrieval dataset validation, top-k metrics, and CLI orchestration."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))


def make_result(
    rank: int,
    *,
    document_id: str,
    chunk_index: int,
) -> object:
    """Build a ranked retrieval result with compact test metadata."""
    from rag_pipeline.retrieval import RetrievalResult

    return RetrievalResult(
        document=Document(
            page_content=f"Evidence from {document_id} chunk {chunk_index}.",
            metadata={
                "document_id": document_id,
                "chunk_index": chunk_index,
            },
        ),
        score=1.0 / rank,
        rank=rank,
    )


class RetrievalEvaluationTests(unittest.TestCase):
    """Verify strict labels and mathematically defined retrieval metrics."""

    def test_loads_versioned_dataset_with_exact_metadata_selectors(self) -> None:
        from rag_pipeline.retrieval_evaluation import (
            load_retrieval_evaluation_dataset,
        )

        payload = {
            "schema_version": 1,
            "name": "policy-queries-v1",
            "cases": [
                {
                    "id": "expense-receipts",
                    "query": "Which receipts are required?",
                    "relevant": [
                        {
                            "file_name": "expenses.pdf",
                            "page": 2,
                            "chunk_index": 0,
                        }
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "retrieval.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")
            dataset = load_retrieval_evaluation_dataset(dataset_path)

        self.assertEqual(dataset.name, "policy-queries-v1")
        self.assertEqual(dataset.schema_version, 1)
        self.assertEqual(dataset.cases[0].case_id, "expense-receipts")
        self.assertEqual(
            dict(dataset.cases[0].relevant_documents[0].metadata),
            payload["cases"][0]["relevant"][0],
        )

    def test_relevance_matching_requires_all_values_and_matching_types(self) -> None:
        from rag_pipeline.retrieval_evaluation import RelevantDocument

        relevant_document = RelevantDocument(
            {"file_name": "policy.pdf", "page": 1}
        )

        self.assertTrue(
            relevant_document.matches(
                Document(
                    page_content="Policy",
                    metadata={"file_name": "policy.pdf", "page": 1},
                )
            )
        )
        self.assertFalse(
            relevant_document.matches(
                Document(
                    page_content="Policy",
                    metadata={"file_name": "policy.pdf", "page": True},
                )
            )
        )
        self.assertFalse(
            relevant_document.matches(
                Document(
                    page_content="Policy",
                    metadata={"file_name": "other.pdf", "page": 1},
                )
            )
        )

    def test_calculates_per_case_and_macro_metrics_at_fixed_cutoff(self) -> None:
        from rag_pipeline.retrieval_evaluation import (
            RelevantDocument,
            RetrievalEvaluationCase,
            RetrievalEvaluationDataset,
            evaluate_retrieval,
            format_retrieval_evaluation_table,
            retrieval_evaluation_to_dict,
        )

        dataset = RetrievalEvaluationDataset(
            name="policies-v1",
            cases=(
                RetrievalEvaluationCase(
                    case_id="expense",
                    query="Expense query",
                    relevant_documents=(
                        RelevantDocument(
                            {"document_id": "expense", "chunk_index": 1}
                        ),
                        RelevantDocument(
                            {"document_id": "expense", "chunk_index": 2}
                        ),
                    ),
                ),
                RetrievalEvaluationCase(
                    case_id="leave",
                    query="Leave query",
                    relevant_documents=(
                        RelevantDocument(
                            {"document_id": "leave", "chunk_index": 0}
                        ),
                    ),
                ),
            ),
        )

        def retrieve(query: str) -> list[object]:
            if query == "Expense query":
                return [
                    make_result(1, document_id="unrelated", chunk_index=0),
                    make_result(2, document_id="expense", chunk_index=1),
                ]
            return []

        report = evaluate_retrieval(dataset, retrieve, top_k=3)
        expense_metrics = report.cases[0]

        self.assertEqual(expense_metrics.retrieved_count, 2)
        self.assertEqual(expense_metrics.matched_relevant_count, 1)
        self.assertEqual(expense_metrics.relevant_retrieved_count, 1)
        self.assertEqual(expense_metrics.first_relevant_rank, 2)
        self.assertEqual(expense_metrics.hit_at_k, 1.0)
        self.assertAlmostEqual(expense_metrics.precision_at_k, 1 / 3)
        self.assertEqual(expense_metrics.recall_at_k, 0.5)
        self.assertEqual(expense_metrics.reciprocal_rank_at_k, 0.5)
        self.assertEqual(report.aggregate.hit_rate_at_k, 0.5)
        self.assertAlmostEqual(report.aggregate.mean_precision_at_k, 1 / 6)
        self.assertEqual(report.aggregate.mean_recall_at_k, 0.25)
        self.assertEqual(report.aggregate.mean_reciprocal_rank_at_k, 0.25)

        serialized = retrieval_evaluation_to_dict(report)
        self.assertEqual(serialized["case_count"], 2)
        self.assertEqual(serialized["cases"][0]["first_relevant_rank"], 2)
        table = format_retrieval_evaluation_table(report)
        self.assertIn("Retrieval evaluation: policies-v1", table)
        self.assertIn("MACRO", table)
        self.assertIn("P@3", table)

    def test_rejects_invalid_schema_duplicate_labels_and_unknown_fields(
        self,
    ) -> None:
        from rag_pipeline.exceptions import (
            InvalidRetrievalEvaluationDatasetError,
        )
        from rag_pipeline.retrieval_evaluation import (
            load_retrieval_evaluation_dataset,
        )

        invalid_payloads = (
            {
                "schema_version": 2,
                "name": "wrong-version",
                "cases": [],
            },
            {
                "schema_version": 1,
                "name": "unknown-field",
                "cases": [],
                "top_k": 4,
            },
            {
                "schema_version": 1,
                "name": "duplicate-label",
                "cases": [
                    {
                        "id": "case-1",
                        "query": "Question",
                        "relevant": [
                            {"chunk_id": "same"},
                            {"chunk_id": "same"},
                        ],
                    }
                ],
            },
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            for index, payload in enumerate(invalid_payloads):
                with self.subTest(payload=payload):
                    dataset_path = Path(temp_dir) / f"invalid-{index}.json"
                    dataset_path.write_text(
                        json.dumps(payload),
                        encoding="utf-8",
                    )
                    with self.assertRaises(
                        InvalidRetrievalEvaluationDatasetError
                    ):
                        load_retrieval_evaluation_dataset(dataset_path)

    def test_rejects_invalid_cutoff_and_noncontiguous_ranking(self) -> None:
        from rag_pipeline.exceptions import RetrievalEvaluationInputError
        from rag_pipeline.retrieval_evaluation import (
            RelevantDocument,
            RetrievalEvaluationCase,
            RetrievalEvaluationDataset,
            evaluate_retrieval,
        )

        dataset = RetrievalEvaluationDataset(
            name="invalid-ranking",
            cases=(
                RetrievalEvaluationCase(
                    case_id="case-1",
                    query="Question",
                    relevant_documents=(
                        RelevantDocument({"document_id": "expected"}),
                    ),
                ),
            ),
        )

        with self.assertRaisesRegex(
            RetrievalEvaluationInputError,
            "positive integer",
        ):
            evaluate_retrieval(dataset, lambda _: [], top_k=0)
        with self.assertRaisesRegex(
            RetrievalEvaluationInputError,
            "contiguous one-based ranks",
        ):
            evaluate_retrieval(
                dataset,
                lambda _: [
                    make_result(2, document_id="expected", chunk_index=0)
                ],
                top_k=1,
            )


class RetrievalEvaluationCliTests(unittest.TestCase):
    """Exercise the command against local Qdrant without model downloads."""

    def test_cli_evaluates_multiple_queries_with_one_embedding_service(
        self,
    ) -> None:
        from rag_pipeline.__main__ import main
        from rag_pipeline.embeddings import EmbeddingService
        from rag_pipeline.vector_store import LocalVectorStore, VectorStoreConfig

        class PolicyEmbeddings(Embeddings):
            """Map two policy topics to deterministic orthogonal vectors."""

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

        embedding_service = EmbeddingService(
            PolicyEmbeddings(),
            model_name="evaluation-test-model",
        )
        documents = [
            Document(
                page_content="Expense claims require itemized receipts.",
                metadata={
                    "source": "expenses.txt",
                    "file_name": "expenses.txt",
                    "chunk_index": 0,
                },
            ),
            Document(
                page_content="Annual leave requests use the HR portal.",
                metadata={
                    "source": "leave.txt",
                    "file_name": "leave.txt",
                    "chunk_index": 0,
                },
            ),
        ]
        dataset_payload = {
            "schema_version": 1,
            "name": "cli-policies-v1",
            "cases": [
                {
                    "id": "expenses",
                    "query": "Which receipt is required?",
                    "relevant": [
                        {"file_name": "expenses.txt", "chunk_index": 0}
                    ],
                },
                {
                    "id": "leave",
                    "query": "Where are leave requests submitted?",
                    "relevant": [
                        {"file_name": "leave.txt", "chunk_index": 0}
                    ],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            store_config = VectorStoreConfig(
                path=Path(temp_dir) / "qdrant",
                collection_name="evaluation-policies",
            )
            with LocalVectorStore(store_config) as store:
                store.index(
                    embedding_service.embed_documents(documents),
                    model_identifier=embedding_service.model_identifier,
                )

            dataset_path = Path(temp_dir) / "retrieval-evaluation.json"
            dataset_path.write_text(
                json.dumps(dataset_payload),
                encoding="utf-8",
            )
            output = io.StringIO()
            with patch(
                "rag_pipeline.embeddings.create_local_embedding_service",
                return_value=embedding_service,
            ) as embedding_factory:
                with redirect_stdout(output):
                    exit_code = main(
                        [
                            "evaluate-retrieval",
                            str(dataset_path),
                            "--store-path",
                            str(store_config.resolved_path),
                            "--collection-name",
                            store_config.collection_name,
                            "--top-k",
                            "1",
                            "--output-format",
                            "json",
                        ]
                    )

        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(embedding_factory.call_count, 1)
        self.assertEqual(report["dataset_name"], "cli-policies-v1")
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["metrics"]["hit_rate_at_k"], 1.0)
        self.assertEqual(report["metrics"]["mean_recall_at_k"], 1.0)
        self.assertEqual(
            report["metrics"]["mean_reciprocal_rank_at_k"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
