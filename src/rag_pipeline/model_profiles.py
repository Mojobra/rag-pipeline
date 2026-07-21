"""Load secure provider-specific model profiles from environment settings.

The module maps stable CLI aliases to API credentials plus generation and
embedding model names. It performs no provider initialization and keeps secret
values out of dataclass representations and validation messages.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import cast

from dotenv import dotenv_values

from rag_pipeline.exceptions import InvalidModelProviderConfigurationError


DEFAULT_ENV_FILE = Path(".env")


class ModelProvider(str, Enum):
    """Identify a hosted generation provider selectable from the CLI.

    The enum values are deliberately stable user-facing aliases. Provider model
    IDs remain external configuration so upgrades do not require code changes.
    """

    GEMINI = "gemini"
    OPENAI = "openai"
    CLAUDE = "claude"


@dataclass(frozen=True, slots=True)
class ProviderModelProfile:
    """Represent one provider's credential and paired RAG model choices.

    The profile is the configuration handoff to embedding and generation
    factories. Its API key is excluded from ``repr`` so logs and tracebacks do
    not reveal secrets; provider clients receive the key explicitly later.
    """

    provider: ModelProvider
    api_key: str = field(repr=False)
    generation_model: str
    embedding_model: str

    def __post_init__(self) -> None:
        """Reject incomplete profiles before any provider client is created."""
        if not isinstance(self.provider, ModelProvider):
            raise InvalidModelProviderConfigurationError(
                "provider must be a ModelProvider."
            )
        for name, value in (
            ("api_key", self.api_key),
            ("generation_model", self.generation_model),
            ("embedding_model", self.embedding_model),
        ):
            if not isinstance(value, str) or not value.strip():
                raise InvalidModelProviderConfigurationError(
                    f"{name} must be a non-empty string."
                )

    @property
    def uses_local_embeddings(self) -> bool:
        """Report whether this profile uses the local embedding adapter.

        Anthropic exposes Claude generation but no embedding API. Its profile
        therefore treats ``CLAUDE_EMBED`` as a Hugging Face model name while
        still using the Anthropic API for answer generation.
        """
        return self.provider == ModelProvider.CLAUDE


_PROFILE_VARIABLES: dict[ModelProvider, tuple[str, str, str]] = {
    ModelProvider.GEMINI: ("GOOGLE_API_KEY", "GEMINI", "GEMINI_EMBED"),
    ModelProvider.OPENAI: ("OPENAI_API_KEY", "OPENAI", "OPENAI_EMBED"),
    ModelProvider.CLAUDE: ("ANTHROPIC_API_KEY", "CLAUDE", "CLAUDE_EMBED"),
}
_API_KEY_PLACEHOLDERS = frozenset({"<api-key>", "replace-me", "your-api-key"})


def is_provider_alias(value: object) -> bool:
    """Return whether a CLI value selects one of the hosted model profiles."""
    if not isinstance(value, str):
        return False
    return value.strip().lower() in {provider.value for provider in ModelProvider}


def load_provider_model_profile(
    provider: ModelProvider | str,
    *,
    env_file: str | Path | None = DEFAULT_ENV_FILE,
    environ: Mapping[str, str] | None = None,
) -> ProviderModelProfile:
    """Resolve one complete model profile from a dotenv file and environment.

    Values from ``environ`` (or ``os.environ``) override the dotenv file, which
    matches production secret-injection conventions. The function reads at
    most one local file, does not mutate process environment state, and reports
    missing variable names without including any secret values.

    Args:
        provider: Stable provider alias or ``ModelProvider`` value.
        env_file: Dotenv file to read, or ``None`` to disable file loading.
        environ: Environment mapping used for overrides; defaults to the process.

    Returns:
        A validated profile containing both model IDs and the provider key.

    Raises:
        InvalidModelProviderConfigurationError: If the alias is unsupported or
            any required variable is absent, blank, or still an API-key
            placeholder.
    """
    selected_provider = _parse_provider(provider)
    values: dict[str, str | None] = {}
    if env_file is not None:
        values.update(dotenv_values(Path(env_file)))
    values.update(os.environ if environ is None else environ)

    key_name, generation_name, embedding_name = _PROFILE_VARIABLES[
        selected_provider
    ]
    required_names = (key_name, generation_name, embedding_name)
    missing_names = [
        name
        for name in required_names
        if _is_unconfigured_value(name, values.get(name))
    ]
    if missing_names:
        missing = ", ".join(missing_names)
        raise InvalidModelProviderConfigurationError(
            f"Model profile '{selected_provider.value}' requires configured "
            f"environment variable(s): {missing}."
        )

    return ProviderModelProfile(
        provider=selected_provider,
        api_key=cast("str", values[key_name]).strip(),
        generation_model=cast("str", values[generation_name]).strip(),
        embedding_model=cast("str", values[embedding_name]).strip(),
    )


def _parse_provider(provider: ModelProvider | str) -> ModelProvider:
    if isinstance(provider, ModelProvider):
        return provider
    if not isinstance(provider, str) or not provider.strip():
        raise InvalidModelProviderConfigurationError(
            "provider must be one of: gemini, openai, claude."
        )
    try:
        return ModelProvider(provider.strip().lower())
    except ValueError as exc:
        raise InvalidModelProviderConfigurationError(
            f"Unsupported model provider {provider!r}; choose gemini, openai, or "
            "claude."
        ) from exc


def _is_unconfigured_value(name: str, value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return True
    return name.endswith("_API_KEY") and value.strip().lower() in _API_KEY_PLACEHOLDERS
