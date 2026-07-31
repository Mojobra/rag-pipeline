"""Test benchmark configuration normalization and cross-stage invariants."""

from __future__ import annotations

import unittest

from rag_pipeline.benchmarking.config import BenchmarkConfig
from rag_pipeline.exceptions import InvalidBenchmarkConfigurationError
from rag_pipeline.infrastructure.sparse_embeddings import LocalSparseEmbeddingConfig
from rag_pipeline.infrastructure.vector_store import SearchMode
from rag_pipeline.retrieval import RetrievalConfig
from rag_pipeline.retrieval.reranking import LocalRerankerConfig, RerankingConfig


class BenchmarkConfigTests(unittest.TestCase):
    """Verify benchmark wiring fails before corpus or provider side effects."""

    def test_normalizes_public_values_and_resolves_final_cutoff(self) -> None:
        config = BenchmarkConfig(
            name="  local baseline  ",
            search_mode="dense",
            retrieval=RetrievalConfig(top_k=8),
            local_reranker=LocalRerankerConfig(model_name="reranker-test"),
            reranking=RerankingConfig(top_n=3),
        )

        self.assertEqual(config.name, "local baseline")
        self.assertIs(config.search_mode, SearchMode.DENSE)
        self.assertEqual(config.final_top_k, 3)

    def test_requires_sparse_settings_exactly_for_hybrid_mode(self) -> None:
        sparse = LocalSparseEmbeddingConfig(model_name="sparse-test")
        invalid_settings = (
            ("hybrid", None, "require sparse"),
            ("dense", sparse, "only valid for hybrid"),
        )

        for search_mode, sparse_embedding, message in invalid_settings:
            with (
                self.subTest(search_mode=search_mode),
                self.assertRaisesRegex(
                    InvalidBenchmarkConfigurationError,
                    message,
                ),
            ):
                BenchmarkConfig(
                    name="invalid-search-mode-pair",
                    search_mode=search_mode,
                    sparse_embedding=sparse_embedding,
                )

    def test_requires_reranker_settings_as_a_valid_width_pair(self) -> None:
        local_reranker = LocalRerankerConfig(model_name="reranker-test")
        invalid_settings = (
            (local_reranker, None, RetrievalConfig(), "enabled together"),
            (None, RerankingConfig(), RetrievalConfig(), "enabled together"),
            (
                local_reranker,
                RerankingConfig(top_n=3),
                RetrievalConfig(top_k=2),
                "at least",
            ),
        )

        for local_config, reranking, retrieval, message in invalid_settings:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(
                    InvalidBenchmarkConfigurationError,
                    message,
                ),
            ):
                BenchmarkConfig(
                    name="invalid-reranking-pair",
                    retrieval=retrieval,
                    local_reranker=local_config,
                    reranking=reranking,
                )

    def test_rejects_invalid_scalar_and_component_values(self) -> None:
        invalid_settings = (
            ({"name": " "}, "non-empty"),
            ({"name": "run", "search_mode": "lexical"}, "dense"),
            ({"name": "run", "write_batch_size": True}, "integer"),
            ({"name": "run", "write_batch_size": 0}, "greater than zero"),
            ({"name": "run", "work_directory": " "}, "cannot be empty"),
            ({"name": "run", "chunking": object()}, "ChunkingConfig"),
        )

        for settings, message in invalid_settings:
            with (
                self.subTest(settings=settings),
                self.assertRaisesRegex(
                    (InvalidBenchmarkConfigurationError, TypeError),
                    message,
                ),
            ):
                BenchmarkConfig(**settings)


if __name__ == "__main__":
    unittest.main()
