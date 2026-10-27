"""Verify local API resource reuse and no-evidence behavior without providers."""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from rag_pipeline.api.runtime import LocalApiRuntime
from rag_pipeline.api.settings import load_api_settings
from rag_pipeline.application.retrieval import RetrievalPipelineConfig
from rag_pipeline.exceptions import RetrievalProviderError
from rag_pipeline.generation import AnswerGenerator, GeneratedAnswer
from rag_pipeline.retrieval import RetrievalConfig, RetrievalResult


class RecordingPipeline:
    """Track query policy passed to a lifespan-owned retrieval session."""

    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int, float | None]] = []

    def retrieve(
        self, query: str, *, retrieval_config: RetrievalConfig
    ) -> list[RetrievalResult]:
        self.calls.append(
            (
                query,
                retrieval_config.top_k,
                retrieval_config.score_threshold,
            )
        )
        return self.results


class LocalApiRuntimeTests(unittest.TestCase):
    """Ensure expensive services are lazy, shared, and closed exactly once."""

    def test_retrieval_and_generation_reuse_one_open_session(self) -> None:
        hit = RetrievalResult(
            Document(page_content="Receipt required.", metadata={"source": "a.md"}),
            score=0.8,
            rank=1,
        )
        pipeline = RecordingPipeline([hit])
        lifecycle: list[str] = []

        @contextmanager
        def open_pipeline(_config: RetrievalPipelineConfig):
            lifecycle.append("open")
            try:
                yield pipeline
            finally:
                lifecycle.append("close")

        generated = GeneratedAnswer(
            answer="A receipt is required.",
            model_identifier="test-model",
            prompt_identifier="grounded-v2",
            used_context=(hit,),
            citations=(),
            context_characters=17,
            context_was_truncated=False,
            prompt_tokens=42,
            prompt_token_limit=512,
            generated=True,
            packed_evidence_texts=("Receipt required.",),
        )
        generator = Mock(spec=AnswerGenerator)
        generator.generate.return_value = generated

        with (
            patch(
                "rag_pipeline.api.runtime.open_local_retrieval_pipeline",
                side_effect=open_pipeline,
            ),
            patch(
                "rag_pipeline.api.runtime.create_local_answer_generator",
                return_value=generator,
            ) as factory,
        ):
            runtime = LocalApiRuntime(load_api_settings())
            self.assertEqual(lifecycle, [])
            self.assertEqual(
                runtime.retrieve("What?", top_k=2, score_threshold=None), [hit]
            )
            factory.assert_not_called()
            self.assertEqual(
                runtime.answer("What?", top_k=4, score_threshold=0.2).answer,
                "A receipt is required.",
            )
            self.assertEqual(
                runtime.answer("Again?", top_k=3, score_threshold=None).answer,
                "A receipt is required.",
            )
            factory.assert_called_once()
            self.assertEqual(lifecycle, ["open"])
            runtime.close()
            runtime.close()

        self.assertEqual(lifecycle, ["open", "close"])
        self.assertEqual(
            pipeline.calls,
            [("What?", 2, None), ("What?", 4, 0.2), ("Again?", 3, None)],
        )

    def test_no_evidence_skips_generator(self) -> None:
        pipeline = RecordingPipeline([])

        @contextmanager
        def open_pipeline(_config: RetrievalPipelineConfig):
            yield pipeline

        with (
            patch(
                "rag_pipeline.api.runtime.open_local_retrieval_pipeline",
                side_effect=open_pipeline,
            ),
            patch("rag_pipeline.api.runtime.create_local_answer_generator") as factory,
        ):
            runtime = LocalApiRuntime(load_api_settings())
            outcome = runtime.answer("Unknown?", top_k=4, score_threshold=0.2)
            runtime.close()

        self.assertIsNone(outcome.generated_answer)
        factory.assert_not_called()

    def test_empty_source_scope_skips_store_and_generator(self) -> None:
        """A valid caller with no grants cannot trigger unfiltered retrieval."""
        with (
            patch(
                "rag_pipeline.api.runtime.open_local_retrieval_pipeline"
            ) as open_store,
            patch("rag_pipeline.api.runtime.create_local_answer_generator") as factory,
        ):
            runtime = LocalApiRuntime(load_api_settings())
            self.assertEqual(
                runtime.retrieve(
                    "Unknown?", top_k=4, score_threshold=None, source_allowlist=()
                ),
                [],
            )
            outcome = runtime.answer(
                "Unknown?", top_k=4, score_threshold=0.2, source_allowlist=()
            )
            runtime.close()
        self.assertIsNone(outcome.generated_answer)
        open_store.assert_not_called()
        factory.assert_not_called()

    def test_provider_scope_violation_fails_before_answer_generation(self) -> None:
        """Treat an incorrectly scoped provider result as a service failure."""
        hit = RetrievalResult(
            Document(page_content="Secret", metadata={"source": "secret.md"}),
            score=0.9,
            rank=1,
        )
        pipeline = RecordingPipeline([hit])

        @contextmanager
        def open_pipeline(_config: RetrievalPipelineConfig):
            yield pipeline

        with (
            patch(
                "rag_pipeline.api.runtime.open_local_retrieval_pipeline",
                side_effect=open_pipeline,
            ),
            patch("rag_pipeline.api.runtime.create_local_answer_generator") as factory,
        ):
            runtime = LocalApiRuntime(load_api_settings())
            with self.assertRaises(RetrievalProviderError):
                runtime.retrieve(
                    "Secret?",
                    top_k=2,
                    score_threshold=None,
                    source_allowlist=("public.md",),
                )
            with self.assertRaises(RetrievalProviderError):
                runtime.answer(
                    "Secret?",
                    top_k=2,
                    score_threshold=None,
                    source_allowlist=("public.md",),
                )
            runtime.close()
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
