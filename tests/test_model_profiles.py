"""Verify model-profile resolution, precedence, validation, and secret safety."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from rag_pipeline.exceptions import (  # noqa: E402
    InvalidModelProviderConfigurationError,
)
from rag_pipeline.model_profiles import (  # noqa: E402
    ModelProvider,
    ProviderModelProfile,
    is_provider_alias,
    load_provider_model_profile,
)


class ModelProfileTests(unittest.TestCase):
    def test_recognizes_provider_aliases_without_treating_model_ids_as_aliases(
        self,
    ) -> None:
        self.assertTrue(is_provider_alias("gemini"))
        self.assertTrue(is_provider_alias(" OpenAI "))
        self.assertTrue(is_provider_alias("CLAUDE"))
        self.assertFalse(is_provider_alias("text-embedding-3-small"))
        self.assertFalse(is_provider_alias(None))

    def test_loads_dotenv_profile_with_process_environment_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "GOOGLE_API_KEY=file-secret\n"
                "GEMINI=gemini-from-file\n"
                "GEMINI_EMBED=embedding-from-file\n",
                encoding="utf-8",
            )

            profile = load_provider_model_profile(
                "GEMINI",
                env_file=env_file,
                environ={"GEMINI": "gemini-from-environment"},
            )

        self.assertEqual(profile.provider, ModelProvider.GEMINI)
        self.assertEqual(profile.api_key, "file-secret")
        self.assertEqual(profile.generation_model, "gemini-from-environment")
        self.assertEqual(profile.embedding_model, "embedding-from-file")

    def test_reports_missing_variable_names_without_exposing_secrets(self) -> None:
        with self.assertRaisesRegex(
            InvalidModelProviderConfigurationError,
            "OPENAI, OPENAI_EMBED",
        ) as raised:
            load_provider_model_profile(
                ModelProvider.OPENAI,
                env_file=None,
                environ={"OPENAI_API_KEY": "private-secret"},
            )

        self.assertNotIn("private-secret", str(raised.exception))

        with self.assertRaisesRegex(
            InvalidModelProviderConfigurationError,
            "GOOGLE_API_KEY",
        ):
            load_provider_model_profile(
                ModelProvider.GEMINI,
                env_file=None,
                environ={
                    "GOOGLE_API_KEY": "<API-KEY>",
                    "GEMINI": "gemini-model",
                    "GEMINI_EMBED": "embedding-model",
                },
            )

    def test_profile_repr_redacts_api_key_and_marks_claude_embeddings_local(
        self,
    ) -> None:
        profile = ProviderModelProfile(
            provider=ModelProvider.CLAUDE,
            api_key="private-secret",
            generation_model="claude-sonnet",
            embedding_model="sentence-transformers/test-model",
        )

        self.assertNotIn("private-secret", repr(profile))
        self.assertTrue(profile.uses_local_embeddings)

    def test_rejects_unknown_provider_and_incomplete_direct_profiles(self) -> None:
        with self.assertRaisesRegex(
            InvalidModelProviderConfigurationError,
            "Unsupported model provider",
        ):
            load_provider_model_profile(
                "cohere",
                env_file=None,
                environ={},
            )

        with self.assertRaisesRegex(
            InvalidModelProviderConfigurationError,
            "generation_model must be a non-empty string",
        ):
            ProviderModelProfile(
                provider=ModelProvider.GEMINI,
                api_key="secret",
                generation_model=" ",
                embedding_model="embedding-model",
            )


if __name__ == "__main__":
    unittest.main()
