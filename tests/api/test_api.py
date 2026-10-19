"""Exercise HTTP validation, provenance, errors, and lifecycle with doubles."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document

from rag_pipeline.api import create_app
from rag_pipeline.api.settings import load_api_settings
from rag_pipeline.application.answering import AnswerOutcome
from rag_pipeline.exceptions import (
    GenerationProviderError,
    RetrievalInputError,
    VectorStoreCollectionNotFoundError,
    VectorStoreCompatibilityError,
)
from rag_pipeline.generation import INSUFFICIENT_CONTEXT_ANSWER, GeneratedAnswer
from rag_pipeline.generation.citations import Citation
from rag_pipeline.infrastructure.vector_store import SearchMode
from rag_pipeline.retrieval import RetrievalResult


def sample_hit() -> RetrievalResult:
    """Return a ranked policy chunk with stable citation provenance."""
    return RetrievalResult(
        Document(
            page_content="Expense claims require an itemized receipt.",
            metadata={
                "source": "expense.md",
                "page": 0,
                "chunk_index": 2,
                "chunk_id": "chunk-2",
                "private_field": "do not expose",
            },
        ),
        score=0.8,
        rank=1,
    )


def sample_answer(hit: RetrievalResult) -> AnswerOutcome:
    """Build a deterministic cited answer without a language-model provider."""
    citation = Citation(
        number=1,
        source="expense.md",
        page_number=1,
        chunk_index=2,
        start_index=0,
        end_index=42,
        chunk_id="chunk-2",
        retrieval_rank=1,
        retrieval_score=0.8,
        excerpt="Expense claims require an itemized receipt.",
    )
    generated = GeneratedAnswer(
        answer="An itemized receipt is required.",
        model_identifier="local-test-model",
        prompt_identifier="grounded-v2",
        used_context=(hit,),
        citations=(citation,),
        context_characters=42,
        context_was_truncated=False,
        prompt_tokens=64,
        prompt_token_limit=512,
        generated=True,
        packed_evidence_texts=(hit.document.page_content,),
    )
    return AnswerOutcome(generated.answer, generated)


class FakeRuntime:
    """Record HTTP-to-application calls and controlled failure modes."""

    def __init__(self) -> None:
        self.hit = sample_hit()
        self.result = sample_answer(self.hit)
        self.calls: list[tuple[str, str, int, float | None]] = []
        self.failure: Exception | None = None
        self.closed = False

    def retrieve(
        self, query: str, *, top_k: int, score_threshold: float | None
    ) -> list[RetrievalResult]:
        self.calls.append(("retrieve", query, top_k, score_threshold))
        if self.failure is not None:
            raise self.failure
        return [self.hit]

    def answer(
        self, query: str, *, top_k: int, score_threshold: float | None
    ) -> AnswerOutcome:
        self.calls.append(("answer", query, top_k, score_threshold))
        if self.failure is not None:
            raise self.failure
        return self.result

    def close(self) -> None:
        self.closed = True


class ApiContractTests(unittest.TestCase):
    """Keep the first HTTP surface read-only, bounded, and source-cited."""

    def test_health_and_openapi_do_not_query_runtime(self) -> None:
        runtime = FakeRuntime()
        with TestClient(create_app(lambda: runtime)) as client:
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            schema = client.get("/openapi.json").json()
            self.assertIn("/v1/retrieve", schema["paths"])
            self.assertIn("/v1/answer", schema["paths"])
            self.assertNotIn("/v1/index", schema["paths"])
            self.assertEqual(runtime.calls, [])
        self.assertTrue(runtime.closed)

    def test_retrieval_returns_ranked_text_and_selected_provenance(self) -> None:
        runtime = FakeRuntime()
        with TestClient(create_app(lambda: runtime)) as client:
            response = client.post(
                "/v1/retrieve",
                json={"query": "  What receipt?  ", "top_k": 2},
            )
        self.assertEqual(response.status_code, 200)
        hit = response.json()["results"][0]
        self.assertEqual(hit["text"], runtime.hit.document.page_content)
        self.assertEqual(hit["source"], "expense.md")
        self.assertEqual(hit["page_number"], 1)
        self.assertEqual(hit["chunk_index"], 2)
        self.assertNotIn("private_field", str(response.json()))
        self.assertEqual(runtime.calls, [("retrieve", "What receipt?", 2, None)])

    def test_answer_preserves_citations_and_default_gate(self) -> None:
        runtime = FakeRuntime()
        with TestClient(create_app(lambda: runtime)) as client:
            response = client.post("/v1/answer", json={"query": "What receipt?"})
            disabled = client.post(
                "/v1/answer",
                json={"query": "What receipt?", "score_threshold": None},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], runtime.result.answer)
        self.assertTrue(response.json()["generated"])
        self.assertEqual(response.json()["citations"][0]["source"], "expense.md")
        self.assertEqual(response.json()["citations"][0]["number"], 1)
        self.assertEqual(runtime.calls[0], ("answer", "What receipt?", 4, 0.2))
        self.assertEqual(disabled.status_code, 200)
        self.assertEqual(runtime.calls[1], ("answer", "What receipt?", 4, None))

    def test_no_evidence_abstains_without_citations(self) -> None:
        runtime = FakeRuntime()
        runtime.result = AnswerOutcome(INSUFFICIENT_CONTEXT_ANSWER, None)
        with TestClient(create_app(lambda: runtime)) as client:
            response = client.post("/v1/answer", json={"query": "Unknown?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], INSUFFICIENT_CONTEXT_ANSWER)
        self.assertFalse(response.json()["generated"])
        self.assertEqual(response.json()["citations"], [])
        self.assertIsNone(response.json()["model_identifier"])

    def test_request_validation_rejects_unbounded_or_server_owned_options(self) -> None:
        runtime = FakeRuntime()
        invalid = (
            {"query": "   "},
            {"query": "x" * 2001},
            {"query": "valid", "top_k": 0},
            {"query": "valid", "top_k": True},
            {"query": "valid", "score_threshold": 1.5},
            {"query": "valid", "score_threshold": True},
            {"query": "valid", "store_path": "C:/private"},
        )
        with TestClient(create_app(lambda: runtime)) as client:
            for body in invalid:
                with self.subTest(body=body):
                    self.assertEqual(
                        client.post("/v1/retrieve", json=body).status_code, 422
                    )
                    self.assertEqual(
                        client.post("/v1/answer", json=body).status_code, 422
                    )
        self.assertEqual(runtime.calls, [])

    def test_domain_errors_are_stable_and_provider_details_are_hidden(self) -> None:
        runtime = FakeRuntime()
        with TestClient(create_app(lambda: runtime)) as client:
            for failure, status in (
                (RetrievalInputError("query is invalid"), 422),
                (VectorStoreCollectionNotFoundError("missing collection"), 404),
                (VectorStoreCompatibilityError("incompatible collection"), 409),
                (GenerationProviderError("secret provider token"), 503),
            ):
                with self.subTest(status=status):
                    runtime.failure = failure
                    response = client.post("/v1/answer", json={"query": "What?"})
                    self.assertEqual(response.status_code, status)
                    self.assertNotIn("secret provider token", response.text)
        self.assertTrue(runtime.closed)

    def test_environment_selects_collection_and_mode_without_model_loading(
        self,
    ) -> None:
        values = {
            "RAG_API_STORE_PATH": ".rag_data/test-api",
            "RAG_API_COLLECTION_NAME": "policy_hybrid",
            "RAG_API_SEARCH_MODE": "hybrid",
            "RAG_API_EMBEDDING_MODEL": "example/dense",
            "RAG_API_SPARSE_MODEL": "example/sparse",
            "RAG_API_GENERATION_MODEL": "example/generator",
        }
        with patch.dict("os.environ", values):
            settings = load_api_settings()
        self.assertEqual(
            settings.retrieval.vector_store.collection_name, "policy_hybrid"
        )
        self.assertEqual(settings.retrieval.vector_store.search_mode, SearchMode.HYBRID)
        self.assertEqual(settings.retrieval.embedding.model_name, "example/dense")
        self.assertEqual(
            settings.retrieval.sparse_embedding.model_name, "example/sparse"
        )
        self.assertEqual(settings.generation_model.model_name, "example/generator")


if __name__ == "__main__":
    unittest.main()
