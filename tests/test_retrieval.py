from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from rag_pipeline.embeddings import EmbeddedDocument, EmbeddingService  # noqa: E402
from rag_pipeline.exceptions import (  # noqa: E402
    InvalidRetrievalConfigurationError,
    RetrievalInputError,
    RetrievalProviderError,
    VectorStoreCollectionNotFoundError,
    VectorStoreCompatibilityError,
)
from rag_pipeline.retrieval import (  # noqa: E402
    RetrievalConfig,
    RetrieverService,
)
from rag_pipeline.vector_store import (  # noqa: E402
    LocalVectorStore,
    VectorStoreConfig,
)


class QueryEmbeddings(Embeddings):
    def __init__(self, query_vector: list[float]) -> None:
        self.query_vector = query_vector
        self.queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("retrieval must not embed documents")

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self.query_vector


def make_record(
    content: str,
    *,
    source: str,
    chunk_index: int,
    vector: tuple[float, ...],
) -> EmbeddedDocument:
    return EmbeddedDocument(
        document=Document(
            page_content=content,
            metadata={"source": source, "chunk_index": chunk_index},
        ),
        embedding=vector,
    )


class SemanticRetrievalTests(unittest.TestCase):
    def test_returns_ranked_cosine_results_with_metadata(self) -> None:
        records = [
            make_record(
                "Expense claims require receipts.",
                source="expenses.txt",
                chunk_index=0,
                vector=(1.0, 0.0),
            ),
            make_record(
                "Managers review unusual expenses.",
                source="expenses.txt",
                chunk_index=1,
                vector=(0.8, 0.6),
            ),
            make_record(
                "Annual leave requests use the HR portal.",
                source="leave.txt",
                chunk_index=0,
                vector=(0.0, 1.0),
            ),
        ]
        query_model = QueryEmbeddings([1.0, 0.0])
        embedding_service = EmbeddingService(
            query_model,
            model_name="test-model",
        )

        with LocalVectorStore(
            VectorStoreConfig(path=None, collection_name="retrieval")
        ) as store:
            store.index(records, model_identifier="test-model")
            results = RetrieverService(embedding_service, store).retrieve(
                "What evidence is needed for an expense claim?",
                config=RetrievalConfig(top_k=2),
            )

        self.assertEqual(query_model.queries, ["What evidence is needed for an expense claim?"])
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertEqual(
            [result.document.page_content for result in results],
            [
                "Expense claims require receipts.",
                "Managers review unusual expenses.",
            ],
        )
        self.assertAlmostEqual(results[0].score, 1.0, places=6)
        self.assertAlmostEqual(results[1].score, 0.8, places=6)
        self.assertEqual(results[0].document.metadata["source"], "expenses.txt")
        self.assertIn("chunk_id", results[0].document.metadata)

    def test_applies_score_threshold(self) -> None:
        records = [
            make_record(
                "Highly relevant",
                source="a.txt",
                chunk_index=0,
                vector=(1.0, 0.0),
            ),
            make_record(
                "Moderately relevant",
                source="b.txt",
                chunk_index=0,
                vector=(0.8, 0.6),
            ),
        ]
        service = EmbeddingService(
            QueryEmbeddings([1.0, 0.0]),
            model_name="test-model",
        )

        with LocalVectorStore(
            VectorStoreConfig(path=None, collection_name="threshold")
        ) as store:
            store.index(records, model_identifier="test-model")
            results = RetrieverService(service, store).retrieve(
                "query",
                config=RetrievalConfig(top_k=2, score_threshold=0.9),
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].document.page_content, "Highly relevant")

    def test_rejects_blank_query_before_embedding(self) -> None:
        query_model = QueryEmbeddings([1.0, 0.0])
        service = EmbeddingService(query_model, model_name="test-model")

        with LocalVectorStore(
            VectorStoreConfig(path=None, collection_name="blank-query")
        ) as store:
            with self.assertRaisesRegex(RetrievalInputError, "query cannot be empty"):
                RetrieverService(service, store).retrieve("   ")

        self.assertEqual(query_model.queries, [])

    def test_rejects_missing_collection(self) -> None:
        service = EmbeddingService(
            QueryEmbeddings([1.0, 0.0]),
            model_name="test-model",
        )

        with LocalVectorStore(
            VectorStoreConfig(path=None, collection_name="missing")
        ) as store:
            with self.assertRaisesRegex(
                VectorStoreCollectionNotFoundError, "index documents"
            ):
                RetrieverService(service, store).retrieve("query")

    def test_rejects_model_and_dimension_mismatch(self) -> None:
        record = make_record(
            "Indexed content",
            source="indexed.txt",
            chunk_index=0,
            vector=(1.0, 0.0),
        )

        with LocalVectorStore(
            VectorStoreConfig(path=None, collection_name="compatibility")
        ) as store:
            store.index([record], model_identifier="model-a")

            with self.assertRaisesRegex(
                VectorStoreCompatibilityError, "embedding_model"
            ):
                RetrieverService(
                    EmbeddingService(
                        QueryEmbeddings([1.0, 0.0]),
                        model_name="model-b",
                    ),
                    store,
                ).retrieve("query")

            with self.assertRaisesRegex(
                VectorStoreCompatibilityError, "dimension is 2.*use 3"
            ):
                RetrieverService(
                    EmbeddingService(
                        QueryEmbeddings([1.0, 0.0, 0.0]),
                        model_name="model-a",
                    ),
                    store,
                ).retrieve("query")

    def test_rejects_malformed_provider_results(self) -> None:
        record = make_record(
            "Indexed content",
            source="indexed.txt",
            chunk_index=0,
            vector=(1.0, 0.0),
        )
        service = EmbeddingService(
            QueryEmbeddings([1.0, 0.0]),
            model_name="test-model",
        )

        class MalformedLangChainStore:
            def similarity_search_with_score_by_vector(
                self, **kwargs: object
            ) -> list[tuple[Document, float, str]]:
                return [(Document(page_content="Bad row"), 1.0, "extra")]

        with LocalVectorStore(
            VectorStoreConfig(path=None, collection_name="malformed")
        ) as store:
            store.index([record], model_identifier="test-model")
            with patch.object(
                store,
                "as_langchain_vector_store",
                return_value=MalformedLangChainStore(),
            ):
                with self.assertRaisesRegex(
                    RetrievalProviderError, "document-score pair"
                ):
                    RetrieverService(service, store).retrieve("query")

    def test_rejects_invalid_configuration(self) -> None:
        invalid_settings = [
            ({"top_k": 0}, "top_k must be greater than zero"),
            ({"top_k": True}, "top_k must be an integer"),
            ({"score_threshold": math.nan}, "score_threshold must be finite"),
            ({"score_threshold": 1.1}, "score_threshold must be between"),
            ({"score_threshold": True}, "score_threshold must be a number"),
        ]

        for settings, message in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(
                    InvalidRetrievalConfigurationError, message
                ):
                    RetrievalConfig(**settings)


if __name__ == "__main__":
    unittest.main()
