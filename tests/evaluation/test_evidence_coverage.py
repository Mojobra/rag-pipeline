"""Test source-anchor survival through retrieval and exact prompt prefixes."""

from __future__ import annotations

import unittest
from dataclasses import replace

from langchain_core.documents import Document

from rag_pipeline.evaluation.answers import (
    AnswerEvaluationCase,
    AnswerEvaluationDataset,
    evaluate_answers,
)
from rag_pipeline.evaluation.evidence import (
    EvidenceObservation,
    evaluate_evidence_coverage,
    evidence_coverage_to_dict,
    supports_evidence_coverage,
)
from rag_pipeline.evaluation.retrieval import (
    RelevantDocument,
    RetrievalEvaluationCase,
    RetrievalEvaluationDataset,
)
from rag_pipeline.exceptions import AnswerEvaluationInputError
from rag_pipeline.generation import INSUFFICIENT_CONTEXT_ANSWER, GeneratedAnswer
from rag_pipeline.generation.citations import build_citation
from rag_pipeline.retrieval import RetrievalResult


def result(text: str, file_name: str, rank: int) -> RetrievalResult:
    """Create one ranked hit with stable source metadata for selector matching."""
    return RetrievalResult(
        Document(
            page_content=text, metadata={"source": file_name, "file_name": file_name}
        ),
        score=1.0 / rank,
        rank=rank,
    )


def prediction(
    answer: str,
    packed: tuple[tuple[RetrievalResult, str], ...],
    *,
    truncated: bool = False,
) -> GeneratedAnswer:
    """Build an aligned generated answer without invoking any model provider."""
    abstained = answer == INSUFFICIENT_CONTEXT_ANSWER
    citations = (
        ()
        if abstained
        else tuple(
            build_citation(item, number=number, evidence_text=text)
            for number, (item, text) in enumerate(packed, start=1)
        )
    )
    return GeneratedAnswer(
        answer=answer,
        model_identifier="test-model",
        prompt_identifier="grounded-v2",
        used_context=tuple(item for item, _ in packed),
        citations=citations,
        context_characters=sum(len(text) for _, text in packed),
        context_was_truncated=truncated,
        prompt_tokens=100 if packed else 0,
        prompt_token_limit=512,
        generated=bool(packed),
        packed_evidence_texts=tuple(text for _, text in packed),
    )


class EvidenceCoverageTests(unittest.TestCase):
    """Keep selector states separate from answer correctness and duplicate hits."""

    def setUp(self) -> None:
        self.receipt = result("Receipts above EUR 25 are required.", "expense.md", 1)
        self.approval = result(
            "Director approval above EUR 500 is required.", "travel.md", 2
        )
        self.duplicate = result("Receipts above EUR 25 are required.", "expense.md", 2)
        self.unrelated = result("The office is closed on Sunday.", "office.md", 1)
        self.retrieval = RetrievalEvaluationDataset(
            name="paired-retrieval",
            schema_version=2,
            cases=(
                RetrievalEvaluationCase(
                    "multi",
                    "Which receipts and approvals are needed?",
                    (
                        RelevantDocument(
                            {"file_name": "expense.md"}, "Receipts above EUR 25"
                        ),
                        RelevantDocument(
                            {"file_name": "travel.md"},
                            "Director approval above EUR 500",
                        ),
                    ),
                ),
                RetrievalEvaluationCase(
                    "missing",
                    "What is the receipt rule?",
                    (
                        RelevantDocument(
                            {"file_name": "expense.md"}, "Receipts above EUR 25"
                        ),
                    ),
                ),
                RetrievalEvaluationCase(
                    "repeat",
                    "Does the policy mention receipts?",
                    (
                        RelevantDocument(
                            {"file_name": "expense.md"}, "Receipts above EUR 25"
                        ),
                    ),
                ),
                RetrievalEvaluationCase(
                    "truncated",
                    "What approval is needed?",
                    (
                        RelevantDocument(
                            {"file_name": "travel.md"},
                            "Director approval above EUR 500",
                        ),
                    ),
                ),
            ),
        )
        self.answers = AnswerEvaluationDataset(
            name="paired-answers",
            cases=(
                AnswerEvaluationCase(
                    "multi",
                    "Which receipts and approvals are needed?",
                    False,
                    ("Receipts above EUR 25 are required.",),
                ),
                AnswerEvaluationCase(
                    "missing",
                    "What is the receipt rule?",
                    False,
                    ("Receipts above EUR 25 are required.",),
                ),
                AnswerEvaluationCase(
                    "repeat",
                    "Does the policy mention receipts?",
                    False,
                    ("Receipts above EUR 25 are required.",),
                ),
                AnswerEvaluationCase(
                    "truncated",
                    "What approval is needed?",
                    False,
                    ("Director approval above EUR 500 is required.",),
                ),
                AnswerEvaluationCase(
                    "unsupported", "Who founded the company?", True, ()
                ),
            ),
        )

    def test_distinguishes_all_stages_with_duplicates_and_multi_source(self) -> None:
        observations = (
            EvidenceObservation(
                (self.receipt, self.approval),
                prediction(
                    "Receipts above EUR 25 are required.",
                    ((self.receipt, self.receipt.document.page_content),),
                ),
            ),
            EvidenceObservation(
                (self.unrelated,), prediction(INSUFFICIENT_CONTEXT_ANSWER, ())
            ),
            EvidenceObservation(
                (self.receipt, self.duplicate),
                prediction(
                    INSUFFICIENT_CONTEXT_ANSWER,
                    ((self.duplicate, self.duplicate.document.page_content),),
                ),
            ),
            EvidenceObservation(
                (self.approval,),
                prediction(
                    "Approval is needed.",
                    ((self.approval, "Director"),),
                    truncated=True,
                ),
            ),
            EvidenceObservation((), prediction(INSUFFICIENT_CONTEXT_ANSWER, ())),
        )
        responses = iter(item.answer for item in observations)
        answer_report = evaluate_answers(self.answers, lambda _query: next(responses))

        report = evaluate_evidence_coverage(
            self.retrieval, self.answers, answer_report, observations
        )
        serialized = evidence_coverage_to_dict(report)

        self.assertTrue(supports_evidence_coverage(self.retrieval, self.answers))
        self.assertEqual(report.selector_count, 5)
        self.assertEqual(report.not_retrieved_count, 1)
        self.assertEqual(report.retrieved_not_packed_count, 2)
        self.assertEqual(report.packed_but_not_answered_count, 1)
        self.assertEqual(report.packed_and_answered_count, 1)
        self.assertEqual(report.retrieval_coverage_rate, 0.8)
        self.assertEqual(report.packed_coverage_rate, 0.4)
        self.assertEqual(report.fully_packed_case_count, 1)
        self.assertEqual(
            [selector.state for selector in report.cases[0].selectors],
            ["packed_and_answered", "retrieved_not_packed"],
        )
        self.assertEqual(report.cases[2].selectors[0].retrieved_ranks, (1, 2))
        self.assertEqual(report.cases[2].selectors[0].packed_evidence_numbers, (1,))
        self.assertTrue(report.cases[3].context_was_truncated)
        self.assertEqual(serialized["metrics"]["packed_coverage_rate"], 0.4)
        self.assertEqual(len(serialized["cases"]), 4)

    def test_requires_paired_source_anchors_without_changing_legacy_evaluation(
        self,
    ) -> None:
        legacy = RetrievalEvaluationDataset(
            name="legacy",
            cases=(
                RetrievalEvaluationCase(
                    "multi",
                    "Which receipts and approvals are needed?",
                    (RelevantDocument({"file_name": "expense.md"}),),
                ),
            ),
        )
        self.assertFalse(supports_evidence_coverage(legacy, self.answers))
        mismatched = AnswerEvaluationDataset(
            name="mismatched",
            cases=(
                AnswerEvaluationCase(
                    "multi", "A different question?", False, ("Yes.",)
                ),
            ),
        )
        self.assertFalse(supports_evidence_coverage(self.retrieval, mismatched))

    def test_rejects_misaligned_packed_prefix_and_citation(self) -> None:
        observations = (
            EvidenceObservation(
                (self.receipt,),
                prediction(
                    "Answered.", ((self.receipt, self.receipt.document.page_content),)
                ),
            ),
            *(
                EvidenceObservation((), prediction(INSUFFICIENT_CONTEXT_ANSWER, ()))
                for _ in self.answers.cases[1:]
            ),
        )
        responses = iter(item.answer for item in observations)
        answer_report = evaluate_answers(self.answers, lambda _query: next(responses))
        with self.assertRaisesRegex(
            AnswerEvaluationInputError, "align with answer cases"
        ):
            evaluate_evidence_coverage(
                self.retrieval, self.answers, answer_report, observations[:-1]
            )

        for changed_answer, message in (
            (
                replace(
                    observations[0].answer,
                    packed_evidence_texts=("Not a source prefix",),
                ),
                "packed evidence prefixes",
            ),
            (replace(observations[0].answer, citations=()), "citations must align"),
        ):
            with self.subTest(message=message):
                changed = (
                    EvidenceObservation((self.receipt,), changed_answer),
                    *observations[1:],
                )
                with self.assertRaisesRegex(AnswerEvaluationInputError, message):
                    evaluate_evidence_coverage(
                        self.retrieval, self.answers, answer_report, changed
                    )
        wrong_retrieval = (
            EvidenceObservation((self.unrelated,), observations[0].answer),
            *observations[1:],
        )
        with self.assertRaisesRegex(AnswerEvaluationInputError, "retrieval trace"):
            evaluate_evidence_coverage(
                self.retrieval, self.answers, answer_report, wrong_retrieval
            )


if __name__ == "__main__":
    unittest.main()
