from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from rag_pipeline.embeddings import (  # noqa: E402
    EmbeddingService,
    LocalEmbeddingConfig,
    create_local_embedding_service,
    create_profile_embedding_service,
)
from rag_pipeline.exceptions import (  # noqa: E402
    EmbeddingInputError,
    EmbeddingProviderError,
    InvalidEmbeddingConfigurationError,
)
from rag_pipeline.model_profiles import (  # noqa: E402
    ModelProvider,
    ProviderModelProfile,
)


class StubEmbeddings(Embeddings):
    def __init__(
        self,
        document_vectors: list[list[float]],
        query_vector: list[float] | None = None,
    ) -> None:
        self.document_vectors = document_vectors
        self.query_vector = query_vector or [0.5, 0.5]
        self.document_requests: list[list[str]] = []
        self.query_requests: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_requests.append(texts)
        return self.document_vectors

    def embed_query(self, text: str) -> list[float]:
        self.query_requests.append(text)
        return self.query_vector


class EmbeddingServiceTests(unittest.TestCase):
    def test_embeds_documents_and_query_with_one_dimension_contract(self) -> None:
        model = StubEmbeddings(
            document_vectors=[[1, 0], [0, 1]],
            query_vector=[0.25, 0.75],
        )
        service = EmbeddingService(
            model,
            model_name="test-model",
            model_revision="abc123",
        )
        documents = [
            Document(page_content="First chunk", metadata={"chunk_index": 0}),
            Document(page_content="Second chunk", metadata={"chunk_index": 1}),
        ]

        embedded_documents = service.embed_documents(documents)
        query_embedding = service.embed_query("Which chunk is first?")

        self.assertEqual(model.document_requests, [["First chunk", "Second chunk"]])
        self.assertEqual(model.query_requests, ["Which chunk is first?"])
        self.assertIs(embedded_documents[0].document, documents[0])
        self.assertEqual(embedded_documents[0].embedding, (1.0, 0.0))
        self.assertEqual(embedded_documents[1].dimension, 2)
        self.assertEqual(query_embedding, (0.25, 0.75))
        self.assertEqual(service.dimension, 2)
        self.assertEqual(service.model_identifier, "test-model@abc123")
        self.assertEqual(documents[0].metadata, {"chunk_index": 0})

    def test_empty_input_does_not_call_provider(self) -> None:
        model = StubEmbeddings(document_vectors=[])
        service = EmbeddingService(model, model_name="test-model")

        result = service.embed_documents([])

        self.assertEqual(result, [])
        self.assertEqual(model.document_requests, [])
        self.assertIsNone(service.dimension)

    def test_rejects_blank_document_and_query_content(self) -> None:
        service = EmbeddingService(
            StubEmbeddings(document_vectors=[]),
            model_name="test-model",
        )

        with self.assertRaisesRegex(EmbeddingInputError, "empty page_content"):
            service.embed_documents([Document(page_content="   ")])

        with self.assertRaisesRegex(EmbeddingInputError, "query cannot be empty"):
            service.embed_query("\n")

    def test_rejects_invalid_provider_document_responses(self) -> None:
        documents = [
            Document(page_content="First"),
            Document(page_content="Second"),
        ]
        invalid_responses = [
            ([[1.0, 2.0]], "1 vector.*2 document"),
            ([[1.0, 2.0], [3.0]], "inconsistent document dimensions"),
            ([[1.0, float("nan")], [3.0, 4.0]], "non-finite value"),
            ([[], []], "document vector 0 is empty"),
        ]

        for vectors, message in invalid_responses:
            with self.subTest(message=message):
                service = EmbeddingService(
                    StubEmbeddings(document_vectors=vectors),
                    model_name="test-model",
                )
                with self.assertRaisesRegex(EmbeddingProviderError, message):
                    service.embed_documents(documents)

    def test_rejects_query_dimension_changes(self) -> None:
        service = EmbeddingService(
            StubEmbeddings(
                document_vectors=[[1.0, 2.0]],
                query_vector=[1.0, 2.0, 3.0],
            ),
            model_name="test-model",
        )
        service.embed_documents([Document(page_content="Chunk")])

        with self.assertRaisesRegex(
            EmbeddingProviderError, "dimension changed from 2 to 3"
        ):
            service.embed_query("Question")

    def test_rejects_invalid_local_configuration(self) -> None:
        invalid_settings = [
            ({"model_name": " "}, "model_name must be a non-empty string"),
            ({"device": ""}, "device must be a non-empty string"),
            ({"model_revision": " "}, "model_revision must be a non-empty string"),
            ({"batch_size": 0}, "batch_size must be greater than zero"),
            ({"batch_size": True}, "batch_size must be an integer"),
            (
                {"normalize_embeddings": 1},
                "normalize_embeddings must be a boolean",
            ),
        ]

        for settings, message in invalid_settings:
            with self.subTest(settings=settings):
                with self.assertRaisesRegex(
                    InvalidEmbeddingConfigurationError, message
                ):
                    LocalEmbeddingConfig(**settings)

    def test_local_factory_configures_langchain_huggingface(self) -> None:
        captured: dict[str, object] = {}

        class FakeHuggingFaceEmbeddings(StubEmbeddings):
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)
                super().__init__(document_vectors=[[1.0, 0.0]])

        fake_module = ModuleType("langchain_huggingface")
        fake_module.HuggingFaceEmbeddings = FakeHuggingFaceEmbeddings  # type: ignore[attr-defined]
        config = LocalEmbeddingConfig(
            model_name="organization/model",
            model_revision="revision-sha",
            device="cuda:0",
            batch_size=8,
            normalize_embeddings=False,
        )

        with patch.dict(sys.modules, {"langchain_huggingface": fake_module}):
            service = create_local_embedding_service(config)

        self.assertEqual(service.model_identifier, "organization/model@revision-sha")
        self.assertEqual(captured["model_name"], "organization/model")
        self.assertEqual(
            captured["model_kwargs"],
            {"device": "cuda:0", "revision": "revision-sha"},
        )
        self.assertEqual(
            captured["encode_kwargs"],
            {"batch_size": 8, "normalize_embeddings": False},
        )
        self.assertEqual(
            captured["query_encode_kwargs"],
            {"normalize_embeddings": False},
        )

    def test_profile_factory_configures_hosted_langchain_embeddings(self) -> None:
        provider_cases = (
            (
                ModelProvider.GEMINI,
                "langchain_google_genai",
                "GoogleGenerativeAIEmbeddings",
            ),
            (ModelProvider.OPENAI, "langchain_openai", "OpenAIEmbeddings"),
        )

        for provider, module_name, class_name in provider_cases:
            with self.subTest(provider=provider.value):
                captured: dict[str, object] = {}

                class FakeProviderEmbeddings(StubEmbeddings):
                    def __init__(self, **kwargs: object) -> None:
                        captured.update(kwargs)
                        super().__init__(document_vectors=[[1.0, 0.0]])

                fake_module = ModuleType(module_name)
                setattr(fake_module, class_name, FakeProviderEmbeddings)
                profile = ProviderModelProfile(
                    provider=provider,
                    api_key="private-key",
                    generation_model="generation-model",
                    embedding_model="embedding-model",
                )

                with patch.dict(sys.modules, {module_name: fake_module}):
                    service = create_profile_embedding_service(profile)

                self.assertEqual(service.model_identifier, "embedding-model")
                self.assertEqual(captured["model"], "embedding-model")
                self.assertEqual(
                    captured["api_key"].get_secret_value(),  # type: ignore[union-attr]
                    "private-key",
                )

    def test_claude_profile_uses_configured_local_embedding_model(self) -> None:
        profile = ProviderModelProfile(
            provider=ModelProvider.CLAUDE,
            api_key="anthropic-key",
            generation_model="claude-model",
            embedding_model="organization/local-embedding-model",
        )
        expected_service = EmbeddingService(
            StubEmbeddings(document_vectors=[[1.0, 0.0]]),
            model_name="organization/local-embedding-model",
        )

        with patch(
            "rag_pipeline.embeddings.create_local_embedding_service",
            return_value=expected_service,
        ) as local_factory:
            service = create_profile_embedding_service(
                profile,
                local_device="cuda:1",
                local_batch_size=12,
                local_model_revision="revision-sha",
            )

        self.assertIs(service, expected_service)
        local_config = local_factory.call_args.args[0]
        self.assertEqual(local_config.model_name, profile.embedding_model)
        self.assertEqual(local_config.device, "cuda:1")
        self.assertEqual(local_config.batch_size, 12)
        self.assertEqual(local_config.model_revision, "revision-sha")


if __name__ == "__main__":
    unittest.main()
