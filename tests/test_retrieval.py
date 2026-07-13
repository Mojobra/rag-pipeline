from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import SparseEmbeddings, SparseVector


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
    MetadataFilter,
    RetrievalConfig,
    RetrieverService,
    parse_metadata_filter,
)
from rag_pipeline.sparse_embeddings import SparseEmbeddingService  # noqa: E402
from rag_pipeline.vector_store import (  # noqa: E402
    LocalVectorStore,
    SearchMode,
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


class KeywordSparseEmbeddings(SparseEmbeddings):
    def __init__(self) -> None:
        self.document_requests: list[list[str]] = []
        self.query_requests: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        self.document_requests.append(texts)
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> SparseVector:
        self.query_requests.append(text)
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> SparseVector:
        if "zx-42" in text.lower():
            return SparseVector(indices=[42], values=[1.0])
        return SparseVector(indices=[], values=[])


def make_record(
    content: str,
    *,
    source: str,
    chunk_index: int,
    vector: tuple[float, ...],
    metadata: dict[str, object] | None = None,
) -> EmbeddedDocument:
    return EmbeddedDocument(
        document=Document(
            page_content=content,
            metadata={
                "source": source,
                "chunk_index": chunk_index,
                **(metadata or {}),
            },
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

    def test_pushes_metadata_filters_into_qdrant_before_top_k(self) -> None:
        records = [
            make_record(
                "Globally closest but belongs to HR.",
                source="hr.txt",
                chunk_index=0,
                vector=(1.0, 0.0),
                metadata={
                    "department": "hr",
                    "active": True,
                    "policy_version": 2026,
                    "policy": {"region": "global"},
                },
            ),
            make_record(
                "Current finance policy.",
                source="finance.txt",
                chunk_index=0,
                vector=(0.8, 0.6),
                metadata={
                    "department": "finance",
                    "active": True,
                    "policy_version": 2026,
                    "policy": {"region": "eu"},
                },
            ),
            make_record(
                "Archived finance policy.",
                source="finance-archive.txt",
                chunk_index=0,
                vector=(0.9, 0.435889894),
                metadata={
                    "department": "finance",
                    "active": False,
                    "policy_version": 2025,
                    "policy": {"region": "eu"},
                },
            ),
        ]
        service = EmbeddingService(
            QueryEmbeddings([1.0, 0.0]),
            model_name="test-model",
        )

        with LocalVectorStore(
            VectorStoreConfig(path=None, collection_name="metadata-filter")
        ) as store:
            store.index(records, model_identifier="test-model")
            results = RetrieverService(service, store).retrieve(
                "current finance policy",
                config=RetrievalConfig(
                    top_k=1,
                    metadata_filters=(
                        MetadataFilter("department", "finance"),
                        MetadataFilter("active", True),
                        MetadataFilter("policy_version", 2026),
                        MetadataFilter("policy.region", "eu"),
                    ),
                ),
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].document.page_content, "Current finance policy.")
        self.assertEqual(results[0].rank, 1)

    def test_hybrid_rrf_promotes_exact_terms_and_preserves_filters(self) -> None:
        records = [
            make_record(
                "Conceptually related equipment policy.",
                source="semantic.txt",
                chunk_index=0,
                vector=(1.0, 0.0),
                metadata={"department": "finance"},
            ),
            make_record(
                "Repair code ZX-42 requires manager approval.",
                source="exact.txt",
                chunk_index=0,
                vector=(0.0, 1.0),
                metadata={"department": "finance"},
            ),
            make_record(
                "HR procedure ZX-42.",
                source="hr.txt",
                chunk_index=0,
                vector=(0.9, 0.435889894),
                metadata={"department": "hr"},
            ),
        ]
        dense_provider = QueryEmbeddings([1.0, 0.0])
        dense_service = EmbeddingService(dense_provider, model_name="dense-model")
        sparse_provider = KeywordSparseEmbeddings()
        sparse_service = SparseEmbeddingService(
            sparse_provider,
            model_name="sparse-model",
        )
        sparse_vectors = sparse_service.embed_documents(
            [record.document for record in records]
        )

        with LocalVectorStore(
            VectorStoreConfig(
                path=None,
                collection_name="hybrid-retrieval",
                search_mode=SearchMode.HYBRID,
            )
        ) as store:
            store.index(
                records,
                model_identifier="dense-model",
                sparse_vectors=sparse_vectors,
                sparse_model_identifier="sparse-model",
            )
            results = RetrieverService(
                dense_service,
                store,
                sparse_service,
            ).retrieve(
                "What does ZX-42 require?",
                config=RetrievalConfig(
                    top_k=2,
                    metadata_filters=(
                        MetadataFilter("department", "finance"),
                    ),
                ),
            )

        self.assertEqual(dense_provider.queries, ["What does ZX-42 require?"])
        self.assertEqual(
            sparse_provider.query_requests,
            ["What does ZX-42 require?"],
        )
        self.assertEqual(results[0].document.metadata["source"], "exact.txt")
        self.assertEqual(results[0].score_kind, "rrf")
        self.assertNotIn(
            "hr.txt",
            [result.document.metadata["source"] for result in results],
        )

    def test_hybrid_retrieval_falls_back_to_dense_for_empty_sparse_query(self) -> None:
        record = make_record(
            "Conceptually related policy.",
            source="semantic.txt",
            chunk_index=0,
            vector=(1.0, 0.0),
        )
        dense_service = EmbeddingService(
            QueryEmbeddings([1.0, 0.0]),
            model_name="dense-model",
        )
        sparse_service = SparseEmbeddingService(
            KeywordSparseEmbeddings(),
            model_name="sparse-model",
        )

        with LocalVectorStore(
            VectorStoreConfig(
                path=None,
                collection_name="hybrid-dense-fallback",
                search_mode=SearchMode.HYBRID,
            )
        ) as store:
            store.index(
                [record],
                model_identifier="dense-model",
                sparse_vectors=[
                    sparse_service.embed_documents([record.document])[0]
                ],
                sparse_model_identifier="sparse-model",
            )
            results = RetrieverService(
                dense_service,
                store,
                sparse_service,
            ).retrieve("general policy")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].document.metadata["source"], "semantic.txt")
        self.assertEqual(results[0].score_kind, "cosine")

    def test_returns_empty_results_when_metadata_does_not_match(self) -> None:
        record = make_record(
            "Finance policy.",
            source="finance.txt",
            chunk_index=0,
            vector=(1.0, 0.0),
            metadata={"department": "finance"},
        )
        service = EmbeddingService(
            QueryEmbeddings([1.0, 0.0]),
            model_name="test-model",
        )

        with LocalVectorStore(
            VectorStoreConfig(path=None, collection_name="metadata-no-match")
        ) as store:
            store.index([record], model_identifier="test-model")
            results = RetrieverService(service, store).retrieve(
                "policy",
                config=RetrievalConfig(
                    metadata_filters=(MetadataFilter("department", "legal"),)
                ),
            )

        self.assertEqual(results, [])

    def test_parses_string_integer_boolean_and_nested_metadata_filters(self) -> None:
        parsed_filters = [
            parse_metadata_filter("file_extension=.pdf"),
            parse_metadata_filter("page=0"),
            parse_metadata_filter("active=true"),
            parse_metadata_filter('policy.version="001"'),
            parse_metadata_filter("source=archive=2026.txt"),
        ]

        self.assertEqual(
            parsed_filters,
            [
                MetadataFilter("file_extension", ".pdf"),
                MetadataFilter("page", 0),
                MetadataFilter("active", True),
                MetadataFilter("policy.version", "001"),
                MetadataFilter("source", "archive=2026.txt"),
            ],
        )

    def test_rejects_invalid_metadata_filter_expressions(self) -> None:
        invalid_expressions = [
            ("department", "KEY=VALUE"),
            ("department=", "KEY=VALUE"),
            ("bad field=value", "dot-separated"),
            ("page=1.5", "string, integer, or boolean"),
            ("page=null", "string, integer, or boolean"),
            ("tags=[\"finance\"]", "string, integer, or boolean"),
        ]

        for expression, message in invalid_expressions:
            with self.subTest(expression=expression):
                with self.assertRaisesRegex(
                    InvalidRetrievalConfigurationError,
                    message,
                ):
                    parse_metadata_filter(expression)

        with self.assertRaisesRegex(
            InvalidRetrievalConfigurationError,
            "signed 64-bit",
        ):
            MetadataFilter("version", 2**63)

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
            ({"metadata_filters": "department=finance"}, "MetadataFilter objects"),
            ({"metadata_filters": [object()]}, "MetadataFilter objects"),
            (
                {
                    "metadata_filters": [
                        MetadataFilter("department", "finance"),
                        MetadataFilter("department", "finance"),
                    ]
                },
                "duplicate metadata filter",
            ),
        ]

        for settings, message in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(
                    InvalidRetrievalConfigurationError, message
                ):
                    RetrievalConfig(**settings)


if __name__ == "__main__":
    unittest.main()
