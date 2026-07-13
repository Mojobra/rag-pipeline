from __future__ import annotations

import sys
import unittest
from pathlib import Path

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from rag_pipeline.chunking import ChunkingConfig  # noqa: E402
from rag_pipeline.chunking_experiments import (  # noqa: E402
    chunking_experiment_to_dict,
    format_chunking_experiment_table,
    parse_chunking_candidate,
    run_chunking_experiment,
)
from rag_pipeline.exceptions import (  # noqa: E402
    InvalidChunkingExperimentError,
)


class ChunkingExperimentTests(unittest.TestCase):
    def test_compares_candidates_with_exact_overlap_metrics(self) -> None:
        original_metadata = {"source": "alphabet.txt"}
        document = Document(
            page_content="abcdefghijklmnopqrstuvwxyz",
            metadata=original_metadata.copy(),
        )

        report = run_chunking_experiment(
            (item for item in [document]),
            candidates=(
                candidate
                for candidate in [
                    ChunkingConfig(chunk_size=10, chunk_overlap=3),
                    ChunkingConfig(chunk_size=13, chunk_overlap=0),
                ]
            ),
        )

        self.assertEqual(report.input_document_count, 1)
        self.assertEqual(report.chunked_document_count, 1)
        self.assertEqual(report.source_character_count, 26)

        overlapping = report.results[0].metrics
        self.assertEqual(overlapping.chunk_count, 4)
        self.assertEqual(overlapping.total_chunk_characters, 35)
        self.assertEqual(overlapping.min_chunk_characters, 5)
        self.assertEqual(overlapping.mean_chunk_characters, 8.75)
        self.assertEqual(overlapping.p95_chunk_characters, 10)
        self.assertEqual(overlapping.max_chunk_characters, 10)
        self.assertEqual(overlapping.duplicated_characters, 9)
        self.assertEqual(overlapping.duplication_percentage, 25.71)

        non_overlapping = report.results[1].metrics
        self.assertEqual(non_overlapping.chunk_count, 2)
        self.assertEqual(non_overlapping.total_chunk_characters, 26)
        self.assertEqual(non_overlapping.duplicated_characters, 0)
        self.assertEqual(non_overlapping.duplication_percentage, 0.0)
        self.assertEqual(document.metadata, original_metadata)

    def test_reports_zero_metrics_for_blank_documents(self) -> None:
        report = run_chunking_experiment(
            [Document(page_content="   ")],
            candidates=[ChunkingConfig(chunk_size=10, chunk_overlap=2)],
        )

        self.assertEqual(report.input_document_count, 1)
        self.assertEqual(report.chunked_document_count, 0)
        self.assertEqual(report.source_character_count, 0)
        self.assertEqual(report.results[0].metrics.chunk_count, 0)
        self.assertEqual(report.results[0].metrics.mean_chunk_characters, 0.0)

    def test_rejects_missing_duplicate_and_invalid_candidate_types(self) -> None:
        document = Document(page_content="Content")
        duplicate = ChunkingConfig(chunk_size=10, chunk_overlap=2)

        with self.assertRaisesRegex(
            InvalidChunkingExperimentError,
            "at least one chunking candidate",
        ):
            run_chunking_experiment([document], candidates=[])

        with self.assertRaisesRegex(
            InvalidChunkingExperimentError,
            "duplicate chunking candidate 10:2",
        ):
            run_chunking_experiment(
                [document],
                candidates=[duplicate, duplicate],
            )

        with self.assertRaisesRegex(TypeError, "ChunkingConfig"):
            run_chunking_experiment(
                [document],
                candidates=["10:2"],  # type: ignore[list-item]
            )

    def test_rejects_non_document_inputs(self) -> None:
        with self.assertRaisesRegex(TypeError, "LangChain Document"):
            run_chunking_experiment(
                ["plain text"],  # type: ignore[list-item]
                candidates=[ChunkingConfig(chunk_size=10, chunk_overlap=2)],
            )

    def test_parses_candidate_and_explains_invalid_values(self) -> None:
        self.assertEqual(
            parse_chunking_candidate(" 1000 : 200 "),
            ChunkingConfig(chunk_size=1000, chunk_overlap=200),
        )

        invalid_values = [
            ("1000", "SIZE:OVERLAP"),
            ("large:200", "must be integers"),
            ("100:100", "overlap must be smaller"),
        ]
        for value, message in invalid_values:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    InvalidChunkingExperimentError,
                    message,
                ):
                    parse_chunking_candidate(value)

    def test_formats_stable_table_and_json_compatible_data(self) -> None:
        report = run_chunking_experiment(
            [Document(page_content="abcdefghijklmnopqrstuvwxyz")],
            candidates=[ChunkingConfig(chunk_size=10, chunk_overlap=3)],
        )

        data = chunking_experiment_to_dict(report)
        table = format_chunking_experiment_table(report)

        self.assertEqual(data["input_document_count"], 1)
        self.assertEqual(data["candidates"][0]["chunk_size"], 10)  # type: ignore[index]
        self.assertEqual(
            data["candidates"][0]["duplicated_characters"],  # type: ignore[index]
            9,
        )
        self.assertIn("Chunking experiment", table)
        self.assertIn("Documents: 1 input, 1 non-blank", table)
        self.assertIn("9 (25.7%)", table)


if __name__ == "__main__":
    unittest.main()
