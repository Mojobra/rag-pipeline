"""Test transport-neutral indexing and retrieval application contracts."""

from __future__ import annotations

import unittest

from langchain_core.documents import Document

from rag_pipeline.application.indexing import IndexingPipelineConfig
from rag_pipeline.application.retrieval import (
    RetrievalPipeline,
    RetrievalPipelineConfig,
)
from rag_pipeline.chunking import ChunkingConfig
from rag_pipeline.embeddings import LocalEmbeddingConfig
from rag_pipeline.exceptions import InvalidPipelineConfigurationError
from rag_pipeline.reranking import LocalRerankerConfig, RerankingConfig
from rag_pipeline.retrieval import RetrievalConfig, RetrievalResult
from rag_pipeline.sparse_embeddings import LocalSparseEmbeddingConfig
from rag_pipeline.vector_store import SearchMode, VectorStoreConfig


def _result(rank: int = 1) -> RetrievalResult:
    """Build one valid retrieval result for application orchestration tests."""
    return RetrievalResult(
        document=Document(
            page_content="Expense claims require itemized receipts.",
            metadata={"source": "expenses.txt", "chunk_index": 0},
        ),
        score=0.9,
        rank=rank,
    )


class _RecordingRetriever:
    """Return configured results while recording query and config reuse."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, RetrievalConfig | None]] = []

    def retrieve(
        self,
        query: str,
        *,
        config: RetrievalConfig | None = None,
    ) -> list[RetrievalResult]:
        """Record one retrieval request and return a defensive list copy."""
        self.calls.append((query, config))
        return list(self.results)


class _RecordingReranker:
    """Record second-stage calls and return the configured result order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[RetrievalResult], RerankingConfig | None]] = []

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        *,
        config: RerankingConfig | None = None,
    ) -> list[RetrievalResult]:
        """Record a reranking request without altering its candidates."""
        self.calls.append((query, list(results), config))
        return list(results)


class ApplicationConfigurationTests(unittest.TestCase):
    """Verify cross-stage settings fail before provider side effects."""

    def test_requires_sparse_settings_exactly_for_hybrid_workflows(self) -> None:
        embedding = LocalEmbeddingConfig(model_name="dense-test")
        sparse = LocalSparseEmbeddingConfig(model_name="sparse-test")

        invalid_indexing_settings = (
            (SearchMode.HYBRID, None),
            (SearchMode.DENSE, sparse),
        )
        for search_mode, sparse_config in invalid_indexing_settings:
            with (
                self.subTest(stage="indexing", search_mode=search_mode),
                self.assertRaisesRegex(
                    InvalidPipelineConfigurationError,
                    "Hybrid indexing requires sparse",
                ),
            ):
                IndexingPipelineConfig(
                    chunking=ChunkingConfig(),
                    embedding=embedding,
                    vector_store=VectorStoreConfig(search_mode=search_mode),
                    sparse_embedding=sparse_config,
                )

        invalid_retrieval_settings = (
            (SearchMode.HYBRID, None),
            (SearchMode.DENSE, sparse),
        )
        for search_mode, sparse_config in invalid_retrieval_settings:
            with (
                self.subTest(stage="retrieval", search_mode=search_mode),
                self.assertRaisesRegex(
                    InvalidPipelineConfigurationError,
                    "Hybrid retrieval requires sparse",
                ),
            ):
                RetrievalPipelineConfig(
                    embedding=embedding,
                    vector_store=VectorStoreConfig(search_mode=search_mode),
                    sparse_embedding=sparse_config,
                    retrieval=RetrievalConfig(),
                    local_reranker=None,
                    reranking=None,
                )

    def test_requires_reranker_model_and_result_settings_together(self) -> None:
        invalid_reranker_settings = (
            (LocalRerankerConfig(model_name="reranker-test"), None),
            (None, RerankingConfig()),
        )
        for local_reranker, reranking in invalid_reranker_settings:
            with (
                self.subTest(local_reranker=local_reranker),
                self.assertRaisesRegex(
                    InvalidPipelineConfigurationError,
                    "configured together",
                ),
            ):
                RetrievalPipelineConfig(
                    embedding=LocalEmbeddingConfig(model_name="dense-test"),
                    vector_store=VectorStoreConfig(),
                    sparse_embedding=None,
                    retrieval=RetrievalConfig(),
                    local_reranker=local_reranker,
                    reranking=reranking,
                )


class RetrievalPipelineTests(unittest.TestCase):
    """Verify reusable retrieval and lazy reranking orchestration."""

    def test_reuses_config_and_skips_reranker_factory_for_empty_results(
        self,
    ) -> None:
        retrieval_config = RetrievalConfig(top_k=3)
        retriever = _RecordingRetriever([])
        factory_calls = 0

        def create_reranker() -> _RecordingReranker:
            nonlocal factory_calls
            factory_calls += 1
            return _RecordingReranker()

        pipeline = RetrievalPipeline(
            retriever,
            retrieval_config=retrieval_config,
            reranker_factory=create_reranker,
            reranking_config=RerankingConfig(top_n=1),
        )

        self.assertEqual(pipeline.retrieve("unsupported question"), [])
        self.assertEqual(factory_calls, 0)
        self.assertEqual(
            retriever.calls,
            [("unsupported question", retrieval_config)],
        )

    def test_initializes_reranker_once_and_reuses_it_across_queries(self) -> None:
        retrieval_config = RetrievalConfig(top_k=2)
        reranking_config = RerankingConfig(top_n=1)
        retriever = _RecordingRetriever([_result()])
        reranker = _RecordingReranker()
        factory_calls = 0

        def create_reranker() -> _RecordingReranker:
            nonlocal factory_calls
            factory_calls += 1
            return reranker

        pipeline = RetrievalPipeline(
            retriever,
            retrieval_config=retrieval_config,
            reranker_factory=create_reranker,
            reranking_config=reranking_config,
        )

        first = pipeline.retrieve("first")
        second = pipeline.retrieve("second")

        self.assertEqual(first, [_result()])
        self.assertEqual(second, [_result()])
        self.assertEqual(factory_calls, 1)
        self.assertEqual(
            [call[0] for call in reranker.calls],
            ["first", "second"],
        )
        self.assertTrue(all(call[2] is reranking_config for call in reranker.calls))

    def test_rejects_invalid_lazy_reranker_before_invocation(self) -> None:
        pipeline = RetrievalPipeline(
            _RecordingRetriever([_result()]),
            retrieval_config=RetrievalConfig(),
            reranker_factory=lambda: object(),
            reranking_config=RerankingConfig(),
        )

        with self.assertRaisesRegex(
            TypeError,
            "reranker_factory must return",
        ):
            pipeline.retrieve("expense policy")


if __name__ == "__main__":
    unittest.main()
