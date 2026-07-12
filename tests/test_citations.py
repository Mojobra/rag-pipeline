from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from rag_pipeline.citations import (  # noqa: E402
    CitationConfig,
    build_citation,
    build_citations,
    format_citation,
)
from rag_pipeline.exceptions import (  # noqa: E402
    CitationInputError,
    InvalidCitationConfigurationError,
)
from rag_pipeline.retrieval import RetrievalResult  # noqa: E402


def make_result(
    *,
    content: str = "Expense claims require receipts and manager approval.",
    metadata: dict[str, object] | None = None,
    score: float = 0.82,
    rank: int = 1,
) -> RetrievalResult:
    document_metadata = (
        {
            "source": "policy.pdf",
            "page": 1,
            "chunk_index": 2,
            "start_index": 100,
            "end_index": 154,
            "chunk_id": "stable-chunk-id",
        }
        if metadata is None
        else metadata
    )
    return RetrievalResult(
        document=Document(
            page_content=content,
            metadata=document_metadata,
        ),
        score=score,
        rank=rank,
    )


class CitationTests(unittest.TestCase):
    def test_builds_structured_citation_from_retrieval_metadata(self) -> None:
        citation = build_citation(
            make_result(rank=2),
            number=1,
            config=CitationConfig(max_excerpt_characters=32),
        )

        self.assertEqual(citation.label, "[1]")
        self.assertEqual(citation.source, "policy.pdf")
        self.assertEqual(citation.page_number, 2)
        self.assertEqual(citation.chunk_index, 2)
        self.assertEqual(citation.start_index, 100)
        self.assertEqual(citation.end_index, 154)
        self.assertEqual(citation.chunk_id, "stable-chunk-id")
        self.assertEqual(citation.retrieval_rank, 2)
        self.assertEqual(citation.retrieval_score, 0.82)
        self.assertLessEqual(len(citation.excerpt), 32)
        self.assertTrue(citation.excerpt.endswith("..."))

        rendered = format_citation(citation)
        self.assertIn(
            "[1] policy.pdf (page 2, chunk 3, characters 100-154)",
            rendered,
        )
        self.assertIn(citation.excerpt, rendered)

    def test_builds_ordered_citations_with_optional_locations(self) -> None:
        results = [
            make_result(metadata={"source": "first.txt"}),
            make_result(metadata={"source": "second.txt"}, rank=2),
        ]

        citations = build_citations(results)

        self.assertEqual([item.number for item in citations], [1, 2])
        self.assertEqual(
            [item.source for item in citations],
            ["first.txt", "second.txt"],
        )
        self.assertIsNone(citations[0].page_number)
        self.assertEqual(
            format_citation(citations[0]).splitlines()[0],
            "[1] first.txt",
        )

    def test_rejects_invalid_configuration(self) -> None:
        for value in (True, 3):
            with self.subTest(value=value):
                with self.assertRaises(InvalidCitationConfigurationError):
                    CitationConfig(max_excerpt_characters=value)

    def test_rejects_untraceable_or_malformed_evidence(self) -> None:
        invalid_results = [
            (make_result(metadata={}), "source metadata"),
            (
                make_result(metadata={"source": "policy.pdf", "page": True}),
                "page metadata",
            ),
            (
                make_result(
                    metadata={"source": "policy.pdf", "start_index": 10}
                ),
                "provided together",
            ),
            (make_result(score=math.nan), "score must be finite"),
        ]

        for result, message in invalid_results:
            with self.subTest(message=message):
                with self.assertRaisesRegex(CitationInputError, message):
                    build_citation(result, number=1)

        with self.assertRaisesRegex(CitationInputError, "evidence text cannot be empty"):
            build_citation(make_result(), number=1, evidence_text=" ")

        with self.assertRaisesRegex(CitationInputError, "must be a prefix"):
            build_citation(
                make_result(),
                number=1,
                evidence_text="manager approval",
            )


if __name__ == "__main__":
    unittest.main()
