"""Validate the committed evaluation corpus and labels as one contract."""

from __future__ import annotations

import re
import unicodedata
import unittest
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "evaluation" / "datasets" / "asteria-policies-v1"
DOCUMENTS_ROOT = DATASET_ROOT / "documents"
RETRIEVAL_DATASET_PATH = DATASET_ROOT / "retrieval-v1.json"
ANSWER_DATASET_PATH = DATASET_ROOT / "answers-v1.json"

_WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return set(_WORD_PATTERN.findall(normalized))


class CommittedEvaluationDatasetTests(unittest.TestCase):
    """Protect the versioned corpus, selectors, and answer labels from drift."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load schemas and reproduce the documented default chunking policy."""
        from rag_pipeline.evaluation.answers import (
            load_answer_evaluation_dataset,
        )
        from rag_pipeline.evaluation.retrieval import (
            load_retrieval_evaluation_dataset,
        )
        from rag_pipeline.ingestion import load_documents
        from rag_pipeline.ingestion.chunking import chunk_documents

        cls.documents = load_documents([DOCUMENTS_ROOT])
        cls.chunks = chunk_documents(cls.documents)
        cls.retrieval_dataset = load_retrieval_evaluation_dataset(
            RETRIEVAL_DATASET_PATH
        )
        cls.answer_dataset = load_answer_evaluation_dataset(ANSWER_DATASET_PATH)

    def test_corpus_and_dataset_versions_are_explicit(self) -> None:
        """Keep the published v1 asset counts and identities intentional."""
        self.assertEqual(len(self.documents), 5)
        self.assertEqual(len(self.chunks), 10)
        self.assertEqual(
            Counter(chunk.metadata["file_name"] for chunk in self.chunks),
            {
                "business-travel.md": 2,
                "expense-reimbursement.md": 2,
                "information-security.md": 2,
                "records-retention.md": 2,
                "remote-work.md": 2,
            },
        )
        self.assertEqual(
            self.retrieval_dataset.name,
            "asteria-policies-retrieval-v1",
        )
        self.assertEqual(
            self.answer_dataset.name,
            "asteria-policies-answers-v1",
        )
        self.assertEqual(self.retrieval_dataset.schema_version, 1)
        self.assertEqual(self.answer_dataset.schema_version, 1)

    def test_retrieval_selectors_resolve_to_one_default_chunk(self) -> None:
        """Fail when corpus or splitter changes make a relevance label ambiguous."""
        for case in self.retrieval_dataset.cases:
            for selector in case.relevant_documents:
                with self.subTest(
                    case_id=case.case_id,
                    selector=dict(selector.metadata),
                ):
                    self.assertEqual(
                        set(selector.metadata),
                        {"file_name", "chunk_index"},
                    )
                    matches = [
                        chunk for chunk in self.chunks if selector.matches(chunk)
                    ]
                    self.assertEqual(len(matches), 1)

    def test_answerable_cases_align_with_retrieval_judgments(self) -> None:
        """Ensure answer scoring uses the same questions as retrieval scoring."""
        retrieval_cases = {case.case_id: case for case in self.retrieval_dataset.cases}
        answerable_cases = {
            case.case_id: case
            for case in self.answer_dataset.cases
            if not case.should_abstain
        }
        abstention_cases = tuple(
            case for case in self.answer_dataset.cases if case.should_abstain
        )

        self.assertEqual(set(answerable_cases), set(retrieval_cases))
        self.assertEqual(len(answerable_cases), 17)
        self.assertEqual(len(abstention_cases), 4)
        for case_id, answer_case in answerable_cases.items():
            with self.subTest(case_id=case_id):
                self.assertEqual(
                    answer_case.query,
                    retrieval_cases[case_id].query,
                )

    def test_reference_vocabulary_occurs_in_labeled_evidence(self) -> None:
        """Catch references that introduce facts absent from relevant chunks."""
        retrieval_cases = {case.case_id: case for case in self.retrieval_dataset.cases}
        for answer_case in self.answer_dataset.cases:
            if answer_case.should_abstain:
                continue

            retrieval_case = retrieval_cases[answer_case.case_id]
            evidence_chunks = [
                chunk
                for chunk in self.chunks
                if any(
                    selector.matches(chunk)
                    for selector in retrieval_case.relevant_documents
                )
            ]
            evidence_tokens = _tokens(
                " ".join(chunk.page_content for chunk in evidence_chunks)
            )
            for reference in answer_case.reference_answers:
                with self.subTest(
                    case_id=answer_case.case_id,
                    reference=reference,
                ):
                    self.assertLessEqual(_tokens(reference), evidence_tokens)


if __name__ == "__main__":
    unittest.main()
