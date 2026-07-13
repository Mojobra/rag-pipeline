from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_qdrant import SparseEmbeddings, SparseVector


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from rag_pipeline.exceptions import (  # noqa: E402
    EmbeddingInputError,
    EmbeddingProviderError,
    InvalidEmbeddingConfigurationError,
)
from rag_pipeline.sparse_embeddings import (  # noqa: E402
    LocalSparseEmbeddingConfig,
    SparseEmbeddingService,
    SparseEmbeddingVector,
    create_local_sparse_embedding_service,
)


class FakeSparseEmbeddings(SparseEmbeddings):
    def __init__(
        self,
        document_vectors: list[SparseVector] | None = None,
        query_vector: SparseVector | None = None,
    ) -> None:
        self.document_vectors = document_vectors or []
        self.query_vector = query_vector or SparseVector(indices=[], values=[])
        self.document_requests: list[list[str]] = []
        self.query_requests: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        self.document_requests.append(texts)
        return self.document_vectors

    def embed_query(self, text: str) -> SparseVector:
        self.query_requests.append(text)
        return self.query_vector


class SparseEmbeddingServiceTests(unittest.TestCase):
    def test_embeds_documents_and_query_with_validated_sorted_vectors(self) -> None:
        provider = FakeSparseEmbeddings(
            document_vectors=[
                SparseVector(indices=[8, 2], values=[0.8, 0.2]),
                SparseVector(indices=[], values=[]),
            ],
            query_vector=SparseVector(indices=[5], values=[1.5]),
        )
        service = SparseEmbeddingService(provider, model_name="test-bm25")
        documents = [
            Document(page_content="Policy ZX-42"),
            Document(page_content="Stop words only"),
        ]

        vectors = service.embed_documents(documents)
        query_vector = service.embed_query("ZX-42")

        self.assertEqual(
            provider.document_requests,
            [["Policy ZX-42", "Stop words only"]],
        )
        self.assertEqual(provider.query_requests, ["ZX-42"])
        self.assertEqual(
            vectors,
            [
                SparseEmbeddingVector(indices=(2, 8), values=(0.2, 0.8)),
                SparseEmbeddingVector(indices=(), values=()),
            ],
        )
        self.assertEqual(
            query_vector,
            SparseEmbeddingVector(indices=(5,), values=(1.5,)),
        )
        self.assertTrue(vectors[1].is_empty)
        self.assertEqual(service.model_identifier, "test-bm25")

    def test_empty_document_input_does_not_call_provider(self) -> None:
        provider = FakeSparseEmbeddings()
        service = SparseEmbeddingService(provider, model_name="test-bm25")

        self.assertEqual(service.embed_documents([]), [])
        self.assertEqual(provider.document_requests, [])

    def test_rejects_blank_document_and_query_content(self) -> None:
        provider = FakeSparseEmbeddings()
        service = SparseEmbeddingService(provider, model_name="test-bm25")

        with self.assertRaisesRegex(EmbeddingInputError, "empty page_content"):
            service.embed_documents([Document(page_content="   ")])
        with self.assertRaisesRegex(EmbeddingInputError, "query cannot be empty"):
            service.embed_query("   ")

        self.assertEqual(provider.document_requests, [])
        self.assertEqual(provider.query_requests, [])

    def test_rejects_invalid_provider_responses(self) -> None:
        invalid_vectors = [
            (
                SparseVector(indices=[1], values=[]),
                "different index and value counts",
            ),
            (SparseVector(indices=[-1], values=[1.0]), "invalid index"),
            (SparseVector(indices=[1, 1], values=[1.0, 2.0]), "duplicate index"),
            (SparseVector(indices=[1], values=[math.nan]), "non-finite"),
        ]

        for vector, message in invalid_vectors:
            with self.subTest(message=message):
                service = SparseEmbeddingService(
                    FakeSparseEmbeddings(document_vectors=[vector]),
                    model_name="test-bm25",
                )
                with self.assertRaisesRegex(EmbeddingProviderError, message):
                    service.embed_documents([Document(page_content="Content")])

        wrong_count = SparseEmbeddingService(
            FakeSparseEmbeddings(document_vectors=[]),
            model_name="test-bm25",
        )
        with self.assertRaisesRegex(EmbeddingProviderError, "returned 0 vector"):
            wrong_count.embed_documents([Document(page_content="Content")])

    def test_rejects_invalid_configuration(self) -> None:
        invalid_settings = [
            ({"model_name": " "}, "model_name must be a non-empty string"),
            ({"cache_dir": ""}, "cache_dir must be non-empty"),
            ({"batch_size": 0}, "batch_size must be greater than zero"),
            ({"batch_size": True}, "batch_size must be an integer"),
            ({"threads": 0}, "threads must be greater than zero"),
        ]

        for settings, message in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(
                    InvalidEmbeddingConfigurationError,
                    message,
                ):
                    LocalSparseEmbeddingConfig(**settings)

    def test_local_factory_configures_langchain_fastembed(self) -> None:
        provider = FakeSparseEmbeddings()

        with tempfile.TemporaryDirectory() as temp_dir:
            config = LocalSparseEmbeddingConfig(
                model_name="test-bm25",
                cache_dir=temp_dir,
                batch_size=64,
                threads=2,
            )
            with patch(
                "langchain_qdrant.FastEmbedSparse",
                return_value=provider,
            ) as factory:
                service = create_local_sparse_embedding_service(config)

        factory.assert_called_once_with(
            model_name="test-bm25",
            batch_size=64,
            cache_dir=str(Path(temp_dir).resolve()),
            threads=2,
        )
        self.assertEqual(service.model_identifier, "test-bm25")


if __name__ == "__main__":
    unittest.main()
