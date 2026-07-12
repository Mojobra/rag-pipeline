from __future__ import annotations

import sys
import unittest
from pathlib import Path

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from rag_pipeline.chunking import ChunkingConfig, chunk_documents  # noqa: E402
from rag_pipeline.exceptions import (  # noqa: E402
    InvalidChunkingConfigurationError,
)


class DocumentChunkingTests(unittest.TestCase):
    def test_splits_with_overlap_and_adds_position_metadata(self) -> None:
        original_metadata = {"source": "alphabet.txt", "page": 2}
        document = Document(
            page_content="abcdefghijklmnopqrstuvwxyz",
            metadata=original_metadata.copy(),
        )

        chunks = chunk_documents(
            [document],
            config=ChunkingConfig(chunk_size=10, chunk_overlap=3),
        )

        self.assertEqual(
            [chunk.page_content for chunk in chunks],
            ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"],
        )
        self.assertEqual(
            [chunk.metadata["start_index"] for chunk in chunks],
            [0, 7, 14, 21],
        )
        self.assertEqual(
            [chunk.metadata["end_index"] for chunk in chunks],
            [10, 17, 24, 26],
        )

        for index, chunk in enumerate(chunks):
            self.assertEqual(chunk.metadata["source"], "alphabet.txt")
            self.assertEqual(chunk.metadata["page"], 2)
            self.assertEqual(chunk.metadata["chunk_index"], index)
            self.assertEqual(chunk.metadata["chunk_count"], 4)
            self.assertEqual(
                chunk.metadata["chunk_char_count"], len(chunk.page_content)
            )

        self.assertEqual(document.metadata, original_metadata)

    def test_preserves_document_order_and_skips_blank_content(self) -> None:
        documents = [
            Document(page_content="   ", metadata={"source": "blank.pdf"}),
            Document(page_content="First", metadata={"source": "first.txt"}),
            Document(page_content="Second", metadata={"source": "second.txt"}),
        ]

        chunks = chunk_documents(
            documents,
            config=ChunkingConfig(chunk_size=20, chunk_overlap=0),
        )

        self.assertEqual(
            [chunk.metadata["source"] for chunk in chunks],
            ["first.txt", "second.txt"],
        )
        self.assertEqual([chunk.metadata["chunk_index"] for chunk in chunks], [0, 0])

    def test_rejects_invalid_configuration(self) -> None:
        invalid_settings = [
            ({"chunk_size": 0}, "chunk_size must be greater than zero"),
            ({"chunk_overlap": -1}, "chunk_overlap cannot be negative"),
            (
                {"chunk_size": 10, "chunk_overlap": 10},
                "chunk_overlap must be smaller than chunk_size",
            ),
            ({"chunk_size": True}, "chunk_size must be an integer"),
        ]

        for settings, message in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(
                    InvalidChunkingConfigurationError, message
                ):
                    ChunkingConfig(**settings)

    def test_rejects_non_document_inputs(self) -> None:
        with self.assertRaisesRegex(TypeError, "LangChain Document"):
            chunk_documents(["plain text"])  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()
