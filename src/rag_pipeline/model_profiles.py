"""Load secure, role-specific model profiles from environment settings.

The module maps stable CLI aliases to generation or embedding configuration.
It performs no provider initialization, validates only the selected role's
settings, and keeps secret values out of representations and error messages.
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
    """Identify a hosted model provider selectable for a CLI model role.

    The enum values are deliberately stable user-facing aliases. Provider model
    IDs remain external configuration so upgrades do not require code changes.
    """

    GEMINI = "gemini"
    OPENAI = "openai"
    CLAUDE = "claude"


@dataclass(frozen=True, slots=True)
class ProviderGenerationProfile:
    """Represent one hosted provider's generation model and credential.

    Answer generation consumes this narrow profile so changing the LLM does
    not select or require an embedding model. The API key is omitted from the
    dataclass representation to reduce accidental secret disclosure.
    """

    provider: ModelProvider
    api_key: str = field(repr=False)
    generation_model: str

    def __post_init__(self) -> None:
        """Reject incomplete generation settings before client construction."""
        _validate_provider(self.provider)
        _validate_profile_value("api_key", self.api_key)
        _validate_profile_value("generation_model", self.generation_model)


@dataclass(frozen=True, slots=True)
class ProviderEmbeddingProfile:
    """Represent one provider-backed embedding selection.

    Gemini and OpenAI use the credential with hosted embedding APIs. Claude's
    configured embedding model is local because Anthropic has no embedding
    endpoint, so its profile intentionally carries no API key.
    """

    provider: ModelProvider
    api_key: str | None = field(repr=False)
    embedding_model: str

    def __post_init__(self) -> None:
        """Require a model and credentials only for hosted embeddings."""
        _validate_provider(self.provider)
        _validate_profile_value("embedding_model", self.embedding_model)
        if self.uses_local_embeddings:
            if self.api_key is not None:
                _validate_profile_value("api_key", self.api_key)
        else:
            _validate_profile_value("api_key", self.api_key)

    @property
    def uses_local_embeddings(self) -> bool:
        """Return whether the configured model runs through the local adapter."""
        return self.provider == ModelProvider.CLAUDE


@dataclass(frozen=True, slots=True)
class ProviderModelProfile:
    """Represent a complete provider profile for compatibility integrations.

    New CLI paths use the narrower generation and embedding profiles so each
    selection is independent. This complete form remains available to callers
    that deliberately need both model roles from one provider.
    """

    provider: ModelProvider
    api_key: str = field(repr=False)
    generation_model: str
    embedding_model: str

    def __post_init__(self) -> None:
        """Reject incomplete profiles before any provider client is created."""
        _validate_provider(self.provider)
        for name, value in (
            ("api_key", self.api_key),
            ("generation_model", self.generation_model),
            ("embedding_model", self.embedding_model),
        ):
            _validate_profile_value(name, value)

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
    values = _load_configuration_values(env_file=env_file, environ=environ)
    key_name, generation_name, embedding_name = _PROFILE_VARIABLES[
        selected_provider
    ]
    required_names = (key_name, generation_name, embedding_name)
    _require_configured_values(selected_provider, required_names, values)

    return ProviderModelProfile(
        provider=selected_provider,
        api_key=_configured_value(values, key_name),
        generation_model=_configured_value(values, generation_name),
        embedding_model=_configured_value(values, embedding_name),
    )


def load_provider_generation_profile(
    provider: ModelProvider | str,
    *,
    env_file: str | Path | None = DEFAULT_ENV_FILE,
    environ: Mapping[str, str] | None = None,
) -> ProviderGenerationProfile:
    """Load only the credential and model required for hosted generation.

    Environment variables override the optional dotenv file. Embedding model
    configuration is deliberately ignored, allowing ``--model`` to change the
    LLM without changing or validating the retrieval embedding selection.

    Args:
        provider: Stable provider alias or enum value.
        env_file: Dotenv file to read, or ``None`` to disable file loading.
        environ: Environment overrides; defaults to the current process.

    Returns:
        A secret-safe profile for the hosted generation factory.

    Raises:
        InvalidModelProviderConfigurationError: If the alias, API key, or
            generation model setting is invalid.
    """
    selected_provider = _parse_provider(provider)
    values = _load_configuration_values(env_file=env_file, environ=environ)
    key_name, generation_name, _ = _PROFILE_VARIABLES[selected_provider]
    _require_configured_values(
        selected_provider,
        (key_name, generation_name),
        values,
    )
    return ProviderGenerationProfile(
        provider=selected_provider,
        api_key=_configured_value(values, key_name),
        generation_model=_configured_value(values, generation_name),
    )


def load_provider_embedding_profile(
    provider: ModelProvider | str,
    *,
    env_file: str | Path | None = DEFAULT_ENV_FILE,
    environ: Mapping[str, str] | None = None,
) -> ProviderEmbeddingProfile:
    """Load only the settings required by the selected embedding adapter.

    Gemini and OpenAI require their API key plus the configured embedding model.
    Claude embeddings are local, so only ``CLAUDE_EMBED`` is required and the
    Anthropic generation credential is not loaded into the profile.

    Args:
        provider: Stable provider alias or enum value.
        env_file: Dotenv file to read, or ``None`` to disable file loading.
        environ: Environment overrides; defaults to the current process.

    Returns:
        A profile for either the hosted or configured local embedding factory.

    Raises:
        InvalidModelProviderConfigurationError: If the alias or a setting used
            by the selected embedding adapter is invalid.
    """
    selected_provider = _parse_provider(provider)
    values = _load_configuration_values(env_file=env_file, environ=environ)
    key_name, _, embedding_name = _PROFILE_VARIABLES[selected_provider]
    required_names = (
        (embedding_name,)
        if selected_provider == ModelProvider.CLAUDE
        else (key_name, embedding_name)
    )
    _require_configured_values(selected_provider, required_names, values)
    return ProviderEmbeddingProfile(
        provider=selected_provider,
        api_key=(
            None
            if selected_provider == ModelProvider.CLAUDE
            else _configured_value(values, key_name)
        ),
        embedding_model=_configured_value(values, embedding_name),
    )


def _load_configuration_values(
    *,
    env_file: str | Path | None,
    environ: Mapping[str, str] | None,
) -> dict[str, str | None]:
    """Read dotenv values and overlay environment-provided configuration."""
    values: dict[str, str | None] = {}
    if env_file is not None:
        values.update(dotenv_values(Path(env_file)))
    values.update(os.environ if environ is None else environ)
    return values


def _require_configured_values(
    provider: ModelProvider,
    required_names: tuple[str, ...],
    values: Mapping[str, str | None],
) -> None:
    """Raise a secret-safe error listing missing provider variable names."""
    missing_names = [
        name
        for name in required_names
        if _is_unconfigured_value(name, values.get(name))
    ]
    if missing_names:
        missing = ", ".join(missing_names)
        raise InvalidModelProviderConfigurationError(
            f"Model profile '{provider.value}' requires configured "
            f"environment variable(s): {missing}."
        )


def _configured_value(values: Mapping[str, str | None], name: str) -> str:
    """Return a value already checked by ``_require_configured_values``."""
    return cast("str", values[name]).strip()


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


def _validate_provider(provider: object) -> None:
    if not isinstance(provider, ModelProvider):
        raise InvalidModelProviderConfigurationError(
            "provider must be a ModelProvider."
        )


def _validate_profile_value(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidModelProviderConfigurationError(
            f"{name} must be a non-empty string."
        )


def _is_unconfigured_value(name: str, value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return True
    return name.endswith("_API_KEY") and value.strip().lower() in _API_KEY_PLACEHOLDERS
