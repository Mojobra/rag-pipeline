"""Test answer dataset validation, deterministic metrics, and CLI execution."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake import FakeListLLM


class CharacterTokenizer:
    """Provide deterministic character-level prompt lengths for tests."""

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


def make_prediction(
    answer: str,
    *,
    citation_count: int = 1,
    generated: bool = True,
) -> object:
    """Build a compact GeneratedAnswer with optional citation records."""
    from rag_pipeline.citations import Citation
    from rag_pipeline.generation import GeneratedAnswer

    citations = tuple(
        Citation(
            number=index + 1,
            source=f"source-{index}.txt",
            page_number=None,
            chunk_index=index,
            start_index=None,
            end_index=None,
            chunk_id=f"chunk-{index}",
            retrieval_rank=index + 1,
            retrieval_score=1.0,
            excerpt="Supporting evidence.",
        )
        for index in range(citation_count)
    )
    return GeneratedAnswer(
        answer=answer,
        model_identifier="test-model",
        prompt_identifier="test-prompt",
        used_context=(),
        citations=citations,
        context_characters=0,
        context_was_truncated=False,
        prompt_tokens=10,
        prompt_token_limit=512,
        generated=generated,
    )


class AnswerEvaluationTests(unittest.TestCase):
    """Verify schema rules, lexical scores, and abstention diagnostics."""

    def test_loads_answerable_and_abstention_cases(self) -> None:
        from rag_pipeline.answer_evaluation import (
            load_answer_evaluation_dataset,
        )

        payload = {
            "schema_version": 1,
            "name": "policy-answers-v1",
            "cases": [
                {
                    "id": "receipts",
                    "query": "What is required?",
                    "should_abstain": False,
                    "reference_answers": [
                        "Itemized receipts are required.",
                        "Claims need itemized receipts.",
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

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "answers.json"
            dataset_path.write_text(json.dumps(payload), encoding="utf-8")
            dataset = load_answer_evaluation_dataset(dataset_path)

        self.assertEqual(dataset.name, "policy-answers-v1")
        self.assertEqual(len(dataset.cases), 2)
        self.assertFalse(dataset.cases[0].should_abstain)
        self.assertEqual(len(dataset.cases[0].reference_answers), 2)
        self.assertTrue(dataset.cases[1].should_abstain)
        self.assertEqual(dataset.cases[1].reference_answers, ())

    def test_rejects_invalid_or_contradictory_dataset_cases(self) -> None:
        from rag_pipeline.answer_evaluation import (
            load_answer_evaluation_dataset,
        )
        from rag_pipeline.exceptions import InvalidAnswerEvaluationDatasetError

        base_case = {
            "id": "case-1",
            "query": "Question",
            "should_abstain": False,
            "reference_answers": ["Answer"],
        }
        invalid_payloads = (
            {
                "schema_version": 2,
                "name": "wrong-version",
                "cases": [base_case],
            },
            {
                "schema_version": 1,
                "name": "unknown-field",
                "cases": [base_case],
                "model": "judge",
            },
            {
                "schema_version": 1,
                "name": "missing-reference",
                "cases": [{**base_case, "reference_answers": []}],
            },
            {
                "schema_version": 1,
                "name": "contradictory",
                "cases": [{**base_case, "should_abstain": True}],
            },
            {
                "schema_version": 1,
                "name": "wrong-boolean",
                "cases": [{**base_case, "should_abstain": "false"}],
            },
            {
                "schema_version": 1,
                "name": "duplicate-reference",
                "cases": [
                    {
                        **base_case,
                        "reference_answers": ["Answer!", "answer"],
                    }
                ],
            },
            {
                "schema_version": 1,
                "name": "empty-normalized-reference",
                "cases": [{**base_case, "reference_answers": ["!!!"]}],
            },
            {
                "schema_version": 1,
                "name": "references-not-list",
                "cases": [{**base_case, "reference_answers": "Answer"}],
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
                    with self.assertRaises(InvalidAnswerEvaluationDatasetError):
                        load_answer_evaluation_dataset(dataset_path)

    def test_calculates_reference_abstention_and_citation_metrics(self) -> None:
        from rag_pipeline.answer_evaluation import (
            AnswerEvaluationCase,
            AnswerEvaluationDataset,
            answer_evaluation_to_dict,
            evaluate_answers,
            format_answer_evaluation_table,
        )
        from rag_pipeline.generation import INSUFFICIENT_CONTEXT_ANSWER

        dataset = AnswerEvaluationDataset(
            name="metrics-v1",
            cases=(
                AnswerEvaluationCase(
                    case_id="exact",
                    query="Exact query",
                    should_abstain=False,
                    reference_answers=("Itemized receipts are required.",),
                ),
                AnswerEvaluationCase(
                    case_id="partial",
                    query="Partial query",
                    should_abstain=False,
                    reference_answers=("An itemized receipt is required.",),
                ),
                AnswerEvaluationCase(
                    case_id="abstain",
                    query="Unsupported query",
                    should_abstain=True,
                    reference_answers=(),
                ),
            ),
        )
        predictions = {
            "Exact query": make_prediction("ITEMIZED receipts are required!"),
            "Partial query": make_prediction("Itemized receipt required."),
            "Unsupported query": make_prediction(
                INSUFFICIENT_CONTEXT_ANSWER,
                citation_count=0,
                generated=False,
            ),
        }

        report = evaluate_answers(dataset, predictions.__getitem__)

        self.assertEqual(report.cases[0].exact_match, 1.0)
        self.assertEqual(report.cases[0].token_f1, 1.0)
        self.assertEqual(report.cases[1].exact_match, 0.0)
        self.assertAlmostEqual(report.cases[1].token_f1, 0.75)
        self.assertIsNone(report.cases[2].exact_match)
        self.assertIsNone(report.cases[2].token_f1)
        self.assertTrue(report.cases[2].predicted_abstention)
        self.assertAlmostEqual(report.aggregate.exact_match_rate or 0.0, 0.5)
        self.assertAlmostEqual(report.aggregate.mean_token_f1 or 0.0, 0.875)
        self.assertEqual(report.aggregate.abstention_accuracy, 1.0)
        self.assertEqual(report.aggregate.abstention_precision, 1.0)
        self.assertEqual(report.aggregate.abstention_recall, 1.0)
        self.assertEqual(report.aggregate.answerable_response_rate, 1.0)
        self.assertEqual(report.aggregate.citation_behavior_rate, 1.0)

        serialized = answer_evaluation_to_dict(report)
        self.assertEqual(serialized["case_count"], 3)
        self.assertEqual(serialized["cases"][2]["exact_match"], None)
        table = format_answer_evaluation_table(report)
        self.assertIn("Answer evaluation: metrics-v1", table)
        self.assertIn("Abstention: accuracy=1.000", table)
        self.assertIn("MACRO", table)

    def test_exposes_always_abstain_and_always_answer_failures(self) -> None:
        from rag_pipeline.answer_evaluation import (
            AnswerEvaluationCase,
            AnswerEvaluationDataset,
            evaluate_answers,
        )
        from rag_pipeline.generation import INSUFFICIENT_CONTEXT_ANSWER

        dataset = AnswerEvaluationDataset(
            name="abstention-errors",
            cases=(
                AnswerEvaluationCase(
                    case_id="answerable",
                    query="Answerable",
                    should_abstain=False,
                    reference_answers=("Supported answer",),
                ),
                AnswerEvaluationCase(
                    case_id="unanswerable",
                    query="Unanswerable",
                    should_abstain=True,
                    reference_answers=(),
                ),
            ),
        )
        predictions = {
            "Answerable": make_prediction(
                INSUFFICIENT_CONTEXT_ANSWER,
                citation_count=0,
            ),
            "Unanswerable": make_prediction("Unsupported answer"),
        }

        report = evaluate_answers(dataset, predictions.__getitem__)

        self.assertEqual(report.aggregate.abstention_accuracy, 0.0)
        self.assertEqual(report.aggregate.abstention_precision, 0.0)
        self.assertEqual(report.aggregate.abstention_recall, 0.0)
        self.assertEqual(report.aggregate.answerable_response_rate, 0.0)
        self.assertEqual(report.aggregate.exact_match_rate, 0.0)
        self.assertEqual(report.aggregate.mean_token_f1, 0.0)
        self.assertEqual(report.aggregate.citation_behavior_rate, 1.0)

    def test_uses_null_for_undefined_abstention_metrics(self) -> None:
        from rag_pipeline.answer_evaluation import (
            AnswerEvaluationCase,
            AnswerEvaluationDataset,
            evaluate_answers,
        )

        dataset = AnswerEvaluationDataset(
            name="answerable-only",
            cases=(
                AnswerEvaluationCase(
                    case_id="answerable",
                    query="Answerable",
                    should_abstain=False,
                    reference_answers=("Answer",),
                ),
            ),
        )

        report = evaluate_answers(
            dataset,
            lambda _: make_prediction("Answer"),
        )

        self.assertIsNone(report.aggregate.abstention_precision)
        self.assertIsNone(report.aggregate.abstention_recall)
        self.assertEqual(report.aggregate.answerable_response_rate, 1.0)

    def test_rejects_invalid_generation_results(self) -> None:
        from rag_pipeline.answer_evaluation import (
            AnswerEvaluationCase,
            AnswerEvaluationDataset,
            evaluate_answers,
        )
        from rag_pipeline.exceptions import AnswerEvaluationInputError
        from rag_pipeline.generation import GeneratedAnswer

        dataset = AnswerEvaluationDataset(
            name="invalid-result",
            cases=(
                AnswerEvaluationCase(
                    case_id="case-1",
                    query="Question",
                    should_abstain=False,
                    reference_answers=("Answer",),
                ),
            ),
        )

        with self.assertRaisesRegex(
            AnswerEvaluationInputError,
            "must return a GeneratedAnswer",
        ):
            evaluate_answers(dataset, lambda _: "Answer")  # type: ignore[arg-type]

        invalid_prediction = GeneratedAnswer(
            answer=" ",
            model_identifier="test-model",
            prompt_identifier="test-prompt",
            used_context=(),
            citations=(),
            context_characters=0,
            context_was_truncated=False,
            prompt_tokens=0,
            prompt_token_limit=512,
            generated=True,
        )
        with self.assertRaisesRegex(
            AnswerEvaluationInputError,
            "invalid answer",
        ):
            evaluate_answers(dataset, lambda _: invalid_prediction)


class AnswerEvaluationCliTests(unittest.TestCase):
    """Exercise full answer evaluation with local provider test doubles."""

    def test_cli_reuses_services_and_scores_answer_and_abstention(self) -> None:
        from rag_pipeline.__main__ import main
        from rag_pipeline.embeddings import EmbeddingService
        from rag_pipeline.generation import AnswerGenerator
        from rag_pipeline.vector_store import LocalVectorStore, VectorStoreConfig

        class PolicyEmbeddings(Embeddings):
            """Map supported and unsupported queries to orthogonal vectors."""

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0] for _ in texts]

            def embed_query(self, text: str) -> list[float]:
                return [1.0, 0.0] if "receipt" in text.lower() else [0.0, 1.0]

        embedding_service = EmbeddingService(
            PolicyEmbeddings(),
            model_name="answer-evaluation-test-model",
        )
        answer_generator = AnswerGenerator(
            FakeListLLM(responses=["Itemized receipts are required."]),
            model_identifier="answer-evaluation-test-llm",
            tokenizer=CharacterTokenizer(),
        )
        document = Document(
            page_content="Expense claims require itemized receipts.",
            metadata={
                "source": "expenses.txt",
                "file_name": "expenses.txt",
                "chunk_index": 0,
            },
        )
        dataset_payload = {
            "schema_version": 1,
            "name": "cli-answer-evaluation-v1",
            "cases": [
                {
                    "id": "receipts",
                    "query": "Which receipt is required?",
                    "should_abstain": False,
                    "reference_answers": ["Itemized receipts are required."],
                },
                {
                    "id": "unsupported",
                    "query": "Who founded the company?",
                    "should_abstain": True,
                    "reference_answers": [],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            store_config = VectorStoreConfig(
                path=Path(temp_dir) / "qdrant",
                collection_name="answer-evaluation-policies",
            )
            with LocalVectorStore(store_config) as store:
                store.index(
                    embedding_service.embed_documents([document]),
                    model_identifier=embedding_service.model_identifier,
                )

            dataset_path = Path(temp_dir) / "answer-evaluation.json"
            dataset_path.write_text(
                json.dumps(dataset_payload),
                encoding="utf-8",
            )
            output = io.StringIO()
            with patch(
                "rag_pipeline.embeddings.create_local_embedding_service",
                return_value=embedding_service,
            ) as embedding_factory:
                with patch(
                    "rag_pipeline.generation.create_local_answer_generator",
                    return_value=answer_generator,
                ) as generation_factory:
                    with redirect_stdout(output):
                        exit_code = main(
                            [
                                "evaluate-answer",
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
        self.assertEqual(generation_factory.call_count, 1)
        self.assertEqual(report["dataset_name"], "cli-answer-evaluation-v1")
        self.assertEqual(report["case_count"], 2)
        self.assertEqual(report["metrics"]["exact_match_rate"], 1.0)
        self.assertEqual(report["metrics"]["mean_token_f1"], 1.0)
        self.assertEqual(report["metrics"]["abstention_accuracy"], 1.0)
        self.assertEqual(report["metrics"]["citation_behavior_rate"], 1.0)
        self.assertFalse(report["cases"][1]["generated"])


if __name__ == "__main__":
    unittest.main()
