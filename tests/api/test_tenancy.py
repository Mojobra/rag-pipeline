"""Exercise tenant-bound HTTP retrieval against separate local Qdrant stores."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_pipeline.api import create_app
from rag_pipeline.api.authorization import load_access_policy
from rag_pipeline.api.runtime import LocalApiRuntime, TenantApiRuntime
from rag_pipeline.api.settings import load_api_settings
from rag_pipeline.application.retrieval import (
    RetrievalPipeline,
    RetrievalPipelineConfig,
)
from rag_pipeline.exceptions import RetrievalProviderError
from rag_pipeline.generation import GeneratedAnswer
from rag_pipeline.generation.citations import Citation
from rag_pipeline.infrastructure.embeddings import EmbeddedDocument, EmbeddingService
from rag_pipeline.infrastructure.vector_store import LocalVectorStore, VectorStoreConfig
from rag_pipeline.retrieval import RetrievalResult, RetrieverService

_ADMIN = "admin-" + "a" * 32
_ALPHA = "alpha-" + "b" * 32
_BETA = "beta-" + "c" * 32
_UNGRANTED = "ungranted-" + "d" * 32


class _FixedEmbeddings(Embeddings):
    """Return one deterministic query vector without provider I/O."""

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("This test indexes precomputed vectors.")


class _EchoGenerator:
    """Expose the retrieved tenant-specific text through answer and citation."""

    def generate(self, question: str, results: list[RetrievalResult], *, config):
        hit = results[0]
        content = hit.document.page_content
        citation = Citation(
            number=1,
            source="policy.md",
            page_number=None,
            chunk_index=0,
            start_index=None,
            end_index=None,
            chunk_id=hit.document.metadata.get("chunk_id"),
            retrieval_rank=1,
            retrieval_score=hit.score,
            excerpt=content,
        )
        return GeneratedAnswer(
            answer=content,
            model_identifier="offline-test-model",
            prompt_identifier="offline-test-prompt",
            used_context=(hit,),
            citations=(citation,),
            context_characters=len(content),
            context_was_truncated=False,
            prompt_tokens=16,
            prompt_token_limit=512,
            generated=True,
            packed_evidence_texts=(content,),
        )


class TenantIsolationTests(unittest.TestCase):
    """Keep tenant identity, grants, storage, and answers bound together."""

    def setUp(self) -> None:
        """Create private v2 policy and distinct temporary tenant directories."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.alpha_path = root / "alpha"
        self.beta_path = root / "beta"
        self.policy_path = root / "policy.json"
        self.policy = {
            "version": 2,
            "tenants": [
                {
                    "id": "alpha",
                    "store_path": str(self.alpha_path),
                    "collection_name": "documents",
                },
                {
                    "id": "beta",
                    "store_path": str(self.beta_path),
                    "collection_name": "documents",
                },
            ],
            "sources": [
                {"tenant": "alpha", "source": "policy.md", "groups": ["readers"]},
                {"tenant": "beta", "source": "policy.md", "groups": ["readers"]},
            ],
        }
        self._write_policy()
        principals = {
            "principals": [
                {
                    "id": "alice",
                    "token": _ALPHA,
                    "tenant": "alpha",
                    "groups": ["readers"],
                },
                {
                    "id": "bob",
                    "token": _BETA,
                    "tenant": "beta",
                    "groups": ["readers"],
                },
                {"id": "carol", "token": _UNGRANTED, "tenant": "alpha"},
            ]
        }
        environment = patch.dict(
            "os.environ",
            {
                "RAG_API_BEARER_TOKEN": _ADMIN,
                "RAG_API_PRINCIPALS_JSON": json.dumps(principals),
                "RAG_API_ACCESS_POLICY_PATH": str(self.policy_path),
                "RAG_API_EMBEDDING_MODEL": "test-model",
            },
            clear=True,
        )
        environment.start()
        self.addCleanup(environment.stop)

    def _write_policy(self) -> None:
        self.policy_path.write_text(json.dumps(self.policy), encoding="utf-8")

    def test_policy_scopes_equal_sources_and_groups_by_tenant(self) -> None:
        """The same source/group labels grant access only within their tenant."""
        policy = load_access_policy(sha256(_ADMIN.encode("ascii")).digest())
        self.assertIsNotNone(policy)
        assert policy is not None
        alpha = policy.access_for_token(_ALPHA)
        beta = policy.access_for_token(_BETA)
        self.assertEqual((alpha.tenant_id, alpha.sources), ("alpha", ("policy.md",)))
        self.assertEqual((beta.tenant_id, beta.sources), ("beta", ("policy.md",)))
        self.assertEqual(policy.access_for_token(_UNGRANTED).sources, ())
        self.assertEqual(len(policy.tenants), 2)

    def test_invalid_tenant_relationships_fail_startup(self) -> None:
        """Reject storage overlap, unscoped grants, and cross-tenant principals."""
        original = json.loads(json.dumps(self.policy))
        invalid_policies = (
            {**original, "tenants": []},
            {**original, "tenants": [original["tenants"][0]] * 2},
            {
                **original,
                "tenants": [
                    original["tenants"][0],
                    {
                        **original["tenants"][1],
                        "store_path": str(self.alpha_path / "sub"),
                    },
                ],
            },
            {**original, "sources": [{"source": "policy.md", "groups": ["readers"]}]},
            {
                **original,
                "sources": [
                    {"tenant": "beta", "source": "policy.md", "principals": ["alice"]}
                ],
            },
            {
                **original,
                "sources": [original["sources"][0], original["sources"][0]],
            },
        )
        for invalid in invalid_policies:
            with self.subTest(invalid=invalid):
                self.policy = invalid
                self._write_policy()
                with self.assertRaises(ValueError):
                    load_access_policy(sha256(_ADMIN.encode("ascii")).digest())

    def test_v2_principals_require_known_tenant(self) -> None:
        """A token without a valid tenant binding cannot query any collection."""
        for tenant in (None, "unknown"):
            principal = {"id": "mallory", "token": _ALPHA, "groups": ["readers"]}
            if tenant is not None:
                principal["tenant"] = tenant
            with self.subTest(tenant=tenant):
                with patch.dict(
                    "os.environ",
                    {
                        "RAG_API_PRINCIPALS_JSON": json.dumps(
                            {"principals": [principal]}
                        )
                    },
                ):
                    with self.assertRaises(ValueError):
                        load_access_policy(sha256(_ADMIN.encode("ascii")).digest())

    def test_missing_tenant_or_source_scope_never_uses_default_store(self) -> None:
        """Internal routing mistakes fail closed before local model or store I/O."""
        policy = load_access_policy(sha256(_ADMIN.encode("ascii")).digest())
        assert policy is not None
        runtime = TenantApiRuntime(load_api_settings(), policy.tenants)
        for tenant, sources in ((None, ("policy.md",)), ("alpha", None), ("other", ())):
            with self.subTest(tenant=tenant, sources=sources):
                with self.assertRaises(RetrievalProviderError):
                    runtime.retrieve(
                        "query",
                        top_k=1,
                        score_threshold=None,
                        source_allowlist=sources,
                        tenant_id=tenant,
                    )
        default_runtime = LocalApiRuntime(load_api_settings())
        with self.assertRaises(RetrievalProviderError):
            default_runtime.retrieve(
                "query",
                top_k=1,
                score_threshold=None,
                source_allowlist=("policy.md",),
                tenant_id="alpha",
            )
        default_runtime.close()
        runtime.close()

    def test_no_grant_skips_store_and_model_initialization(self) -> None:
        """An authenticated tenant principal without grants gets no evidence."""
        with (
            patch("rag_pipeline.api.runtime.open_local_retrieval_pipeline") as opened,
            patch("rag_pipeline.api.runtime.create_local_answer_generator") as model,
            TestClient(create_app()) as client,
        ):
            headers = {"Authorization": f"Bearer {_UNGRANTED}"}
            self.assertEqual(
                client.post(
                    "/v1/retrieve", headers=headers, json={"query": "policy"}
                ).json(),
                {"results": []},
            )
            answer = client.post(
                "/v1/answer", headers=headers, json={"query": "policy"}
            )
            self.assertEqual(answer.status_code, 200)
            self.assertFalse(answer.json()["generated"])
            self.assertEqual(answer.json()["citations"], [])
            opened.assert_not_called()
            model.assert_not_called()

    def test_http_queries_and_citations_use_separate_qdrant_files(self) -> None:
        """Same collection/source IDs cannot cross the tenant storage boundary."""
        for path, content in (
            (self.alpha_path, "Alpha confidential policy."),
            (self.beta_path, "Beta confidential policy."),
        ):
            with LocalVectorStore(
                VectorStoreConfig(path=path, collection_name="documents")
            ) as store:
                store.index(
                    [
                        EmbeddedDocument(
                            Document(
                                page_content=content,
                                metadata={"source": "policy.md", "chunk_index": 0},
                            ),
                            embedding=(1.0, 0.0),
                        )
                    ],
                    model_identifier="test-model",
                )

        @contextmanager
        def open_test_pipeline(config: RetrievalPipelineConfig):
            with LocalVectorStore(config.vector_store) as store:
                retriever = RetrieverService(
                    EmbeddingService(_FixedEmbeddings(), model_name="test-model"),
                    store,
                )
                yield RetrievalPipeline(retriever, retrieval_config=config.retrieval)

        with (
            patch(
                "rag_pipeline.api.runtime.open_local_retrieval_pipeline",
                side_effect=open_test_pipeline,
            ),
            patch(
                "rag_pipeline.api.runtime.create_local_answer_generator",
                return_value=_EchoGenerator(),
            ),
            TestClient(create_app()) as client,
        ):
            for token, expected, forbidden in (
                (_ALPHA, "Alpha confidential policy.", "Beta confidential policy."),
                (_BETA, "Beta confidential policy.", "Alpha confidential policy."),
            ):
                headers = {"Authorization": f"Bearer {token}"}
                retrieval = client.post(
                    "/v1/retrieve",
                    headers=headers,
                    json={"query": "policy", "top_k": 1},
                )
                self.assertEqual(retrieval.status_code, 200)
                self.assertEqual(retrieval.json()["results"][0]["text"], expected)
                self.assertNotIn(forbidden, retrieval.text)
                answer = client.post(
                    "/v1/answer",
                    headers=headers,
                    json={"query": "policy", "top_k": 1},
                )
                self.assertEqual(answer.status_code, 200)
                self.assertEqual(answer.json()["answer"], expected)
                self.assertEqual(answer.json()["citations"][0]["excerpt"], expected)
                self.assertNotIn(forbidden, answer.text)
                self.assertEqual(
                    client.post(
                        "/v1/retrieve",
                        headers=headers,
                        json={"query": "policy", "tenant_id": "beta"},
                    ).status_code,
                    422,
                )
            self.assertEqual(
                client.post(
                    "/v1/retrieve",
                    headers={"Authorization": f"Bearer {_ADMIN}"},
                    json={"query": "policy"},
                ).status_code,
                401,
            )
            self.assertEqual(
                client.post(
                    "/v1/retrieve",
                    headers={"Authorization": f"Bearer {_UNGRANTED}"},
                    json={"query": "policy"},
                ).json(),
                {"results": []},
            )


if __name__ == "__main__":
    unittest.main()
