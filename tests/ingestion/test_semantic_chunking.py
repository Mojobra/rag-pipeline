"""Test bounded semantic grouping with deterministic LangChain embeddings."""

from __future__ import annotations

import unittest

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_pipeline.exceptions import ChunkingError, InvalidChunkingConfigurationError
from rag_pipeline.ingestion.semantic_chunking import (
    SemanticChunkingConfig,
    chunk_documents_semantically,
)


class TopicEmbeddings(Embeddings):
    """Map two synthetic sentence topics to deterministic vector directions."""

    def __init__(self) -> None:
        self.requests: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.requests.append(texts)
        return [
            [1.0, 0.0]
            if "cat" in text.casefold() or "kitten" in text.casefold()
            else [0.0, 1.0]
            for text in texts
        ]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class SemanticChunkingTests(unittest.TestCase):
    """Verify semantic boundaries, hard limits, provenance, and validation."""

    def test_splits_at_topic_change_and_preserves_exact_offsets(self) -> None:
        text = (
            "Cats chase mice. Kittens sleep nearby. "
            "Quarterly invoices are approved. Finance audits payments."
        )
        embeddings = TopicEmbeddings()
        document = Document(
            page_content=text,
            metadata={"source": "topics.txt", "file_name": "topics.txt"},
        )

        chunks = chunk_documents_semantically(
            [document],
            embeddings=embeddings,
            config=SemanticChunkingConfig(
                max_chunk_size=200,
                min_chunk_size=1,
                breakpoint_percentile=50,
                buffer_size=0,
            ),
        )

        self.assertEqual(
            [chunk.page_content for chunk in chunks],
            [
                "Cats chase mice. Kittens sleep nearby.",
                "Quarterly invoices are approved. Finance audits payments.",
            ],
        )
        self.assertEqual(len(embeddings.requests), 1)
        self.assertEqual(len(embeddings.requests[0]), 4)
        self.assertEqual([chunk.metadata["chunk_index"] for chunk in chunks], [0, 1])
        self.assertEqual([chunk.metadata["chunk_count"] for chunk in chunks], [2, 2])
        self.assertEqual(chunks[0].metadata["start_index"], 0)
        self.assertEqual(chunks[0].metadata["end_index"], len(chunks[0].page_content))
        self.assertEqual(
            chunks[1].metadata["start_index"],
            text.index("Quarterly"),
        )
        for chunk in chunks:
            start = chunk.metadata["start_index"]
            end = chunk.metadata["end_index"]
            self.assertEqual(chunk.page_content, text[start:end])
            self.assertEqual(chunk.metadata["chunking_strategy"], "semantic")
        self.assertNotIn("chunk_index", document.metadata)

    def test_hard_splits_one_oversized_sentence(self) -> None:
        embeddings = TopicEmbeddings()

        chunks = chunk_documents_semantically(
            [Document(page_content="abcdefghijklmnopqrstuvwxyz")],
            embeddings=embeddings,
            config=SemanticChunkingConfig(
                max_chunk_size=10,
                min_chunk_size=1,
                buffer_size=0,
            ),
        )

        self.assertEqual([len(chunk.page_content) for chunk in chunks], [10, 10, 6])
        self.assertEqual(len(embeddings.requests), 1)
        self.assertEqual(
            [chunk.metadata["start_index"] for chunk in chunks],
            [0, 10, 20],
        )

    def test_rejects_invalid_configuration(self) -> None:
        invalid_settings = (
            ({"max_chunk_size": 0}, "greater than zero"),
            (
                {"max_chunk_size": 10, "min_chunk_size": 11},
                "cannot exceed",
            ),
            ({"buffer_size": -1}, "cannot be negative"),
            ({"breakpoint_percentile": 0}, "greater than 0"),
            ({"breakpoint_percentile": float("nan")}, "greater than 0"),
        )

        for settings, message in invalid_settings:
            with (
                self.subTest(settings=settings),
                self.assertRaisesRegex(InvalidChunkingConfigurationError, message),
            ):
                SemanticChunkingConfig(**settings)

    def test_rejects_malformed_provider_vector_count(self) -> None:
        class MissingVectorEmbeddings(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0]]

            def embed_query(self, text: str) -> list[float]:
                return [1.0, 0.0]

        with self.assertRaisesRegex(ChunkingError, "1 vector.*2 text unit"):
            chunk_documents_semantically(
                [Document(page_content="First sentence. Second sentence.")],
                embeddings=MissingVectorEmbeddings(),
                config=SemanticChunkingConfig(
                    max_chunk_size=100,
                    min_chunk_size=1,
                    buffer_size=0,
                ),
            )

    def test_rejects_unsafe_semantic_vectors(self) -> None:
        invalid_vectors = (
            ([[0.0, 0.0], [1.0, 0.0]], "zero magnitude"),
            ([[1.0], [1.0, 0.0]], "dimensions are inconsistent"),
            ([[1.0, float("inf")], [1.0, 0.0]], "not finite"),
            ([[1.0, "invalid"], [1.0, 0.0]], "non-numeric"),
        )

        for vectors, message in invalid_vectors:

            class FixedEmbeddings(Embeddings):
                def __init__(self, configured_vectors: object) -> None:
                    self.configured_vectors = configured_vectors

                def embed_documents(self, texts: list[str]) -> list[list[float]]:
                    return self.configured_vectors  # type: ignore[return-value]

                def embed_query(self, text: str) -> list[float]:
                    return [1.0, 0.0]

            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ChunkingError, message),
            ):
                chunk_documents_semantically(
                    [Document(page_content="First sentence. Second sentence.")],
                    embeddings=FixedEmbeddings(vectors),
                    config=SemanticChunkingConfig(
                        max_chunk_size=100,
                        min_chunk_size=1,
                        buffer_size=0,
                    ),
                )

    def test_handles_large_finite_vectors_without_cosine_overflow(self) -> None:
        class LargeVectorEmbeddings(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[1e308, 1e308] for _ in texts]

            def embed_query(self, text: str) -> list[float]:
                return [1e308, 1e308]

        chunks = chunk_documents_semantically(
            [Document(page_content="First sentence. Second sentence.")],
            embeddings=LargeVectorEmbeddings(),
            config=SemanticChunkingConfig(
                max_chunk_size=100,
                min_chunk_size=1,
                buffer_size=0,
            ),
        )

        self.assertEqual(len(chunks), 1)


if __name__ == "__main__":
    unittest.main()
