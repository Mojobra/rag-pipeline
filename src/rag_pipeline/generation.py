"""Execute grounded answer generation from validated, bounded prompt evidence.

The module owns local model construction, LangChain invocation, deterministic
abstention, and citation assembly. Prompt definition and evidence packing live
in :mod:`rag_pipeline.prompting`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser

from rag_pipeline.citations import Citation, build_citation
from rag_pipeline.exceptions import (
    GenerationInputError,
    GenerationProviderError,
    InvalidGenerationConfigurationError,
)
from rag_pipeline.prompting import (
    DEFAULT_TOKEN_SAFETY_MARGIN,
    GROUNDED_ANSWER_PROMPT,
    GROUNDED_ANSWER_PROMPT_ID,
    INSUFFICIENT_CONTEXT_ANSWER,
    PromptTokenizer,
    pack_retrieval_context,
    resolve_input_token_limit,
    validate_prompt_tokenizer,
)
from rag_pipeline.retrieval import RetrievalResult

__all__ = [
    "DEFAULT_LOCAL_GENERATION_MODEL",
    "DEFAULT_TOKEN_SAFETY_MARGIN",
    "GROUNDED_ANSWER_PROMPT",
    "GROUNDED_ANSWER_PROMPT_ID",
    "INSUFFICIENT_CONTEXT_ANSWER",
    "AnswerGenerator",
    "GeneratedAnswer",
    "GenerationConfig",
    "LocalGenerationConfig",
    "PromptTokenizer",
    "create_local_answer_generator",
]

DEFAULT_LOCAL_GENERATION_MODEL = "google/flan-t5-small"


@dataclass(frozen=True, slots=True)
class LocalGenerationConfig:
    """Validated settings for the local Hugging Face generation pipeline.

    The configuration controls reproducible model identity, inference device,
    output length, and deterministic versus sampled generation behavior.
    """

    model_name: str = DEFAULT_LOCAL_GENERATION_MODEL
    model_revision: str | None = None
    device: str = "cpu"
    max_new_tokens: int = 128
    temperature: float = 0.0

    def __post_init__(self) -> None:
        """Validate local model and decoding settings before initialization."""
        _validate_non_empty_string("model_name", self.model_name)
        _validate_non_empty_string("device", self.device)
        if self.model_revision is not None:
            _validate_non_empty_string("model_revision", self.model_revision)
        _pipeline_device(self.device)

        if isinstance(self.max_new_tokens, bool) or not isinstance(
            self.max_new_tokens, int
        ):
            raise InvalidGenerationConfigurationError(
                "max_new_tokens must be an integer."
            )
        if self.max_new_tokens <= 0:
            raise InvalidGenerationConfigurationError(
                "max_new_tokens must be greater than zero."
            )
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, Real):
            raise InvalidGenerationConfigurationError("temperature must be a number.")
        temperature = float(self.temperature)
        if not isfinite(temperature):
            raise InvalidGenerationConfigurationError("temperature must be finite.")
        if not 0.0 <= temperature <= 2.0:
            raise InvalidGenerationConfigurationError(
                "temperature must be between 0 and 2."
            )

    @property
    def model_identifier(self) -> str:
        if self.model_revision is None:
            return self.model_name
        return f"{self.model_name}@{self.model_revision}"


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Validated input-budget policy for one answer-generation request.

    The character cap bounds rendered evidence as a secondary guard. The token
    cap defaults to the tokenizer's model limit and always reserves a safety
    margin before provider inference.
    """

    max_context_characters: int = 1200
    max_input_tokens: int | None = None
    token_safety_margin: int = DEFAULT_TOKEN_SAFETY_MARGIN

    def __post_init__(self) -> None:
        """Reject prompt limits that cannot leave a positive safe token budget."""
        if isinstance(self.max_context_characters, bool) or not isinstance(
            self.max_context_characters, int
        ):
            raise InvalidGenerationConfigurationError(
                "max_context_characters must be an integer."
            )
        if self.max_context_characters <= 0:
            raise InvalidGenerationConfigurationError(
                "max_context_characters must be greater than zero."
            )
        if self.max_input_tokens is not None:
            if isinstance(self.max_input_tokens, bool) or not isinstance(
                self.max_input_tokens, int
            ):
                raise InvalidGenerationConfigurationError(
                    "max_input_tokens must be an integer."
                )
            if self.max_input_tokens <= 0:
                raise InvalidGenerationConfigurationError(
                    "max_input_tokens must be greater than zero."
                )
        if isinstance(self.token_safety_margin, bool) or not isinstance(
            self.token_safety_margin, int
        ):
            raise InvalidGenerationConfigurationError(
                "token_safety_margin must be an integer."
            )
        if self.token_safety_margin < 0:
            raise InvalidGenerationConfigurationError(
                "token_safety_margin cannot be negative."
            )
        if (
            self.max_input_tokens is not None
            and self.token_safety_margin >= self.max_input_tokens
        ):
            raise InvalidGenerationConfigurationError(
                "token_safety_margin must be smaller than max_input_tokens."
            )


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """Grounded generation result with reproducibility and evidence metadata.

    The record identifies model and prompt versions, accepted retrieval records,
    prefix-aligned citations, budget usage, truncation, and whether a model was
    invoked or the no-evidence fallback was returned.
    """

    answer: str
    model_identifier: str
    prompt_identifier: str
    used_context: tuple[RetrievalResult, ...]
    citations: tuple[Citation, ...]
    context_characters: int
    context_was_truncated: bool
    prompt_tokens: int
    prompt_token_limit: int
    generated: bool


class AnswerGenerator:
    """Coordinate guarded prompt construction and one LangChain language model.

    The service packs ranked evidence under exact tokenizer limits, creates
    citations from the same accepted prefixes, invokes the model only when
    evidence remains, and normalizes provider failures for the application.
    """

    def __init__(
        self,
        language_model: BaseLanguageModel[Any],
        *,
        model_identifier: str,
        tokenizer: PromptTokenizer,
    ) -> None:
        if not isinstance(language_model, BaseLanguageModel):
            raise TypeError(
                "language_model must implement LangChain's BaseLanguageModel."
            )
        _validate_non_empty_string("model_identifier", model_identifier)
        validate_prompt_tokenizer(tokenizer)

        self._model_identifier = model_identifier
        self._tokenizer = tokenizer
        self._chain = GROUNDED_ANSWER_PROMPT | language_model | StrOutputParser()

    @property
    def model_identifier(self) -> str:
        return self._model_identifier

    @property
    def prompt_identifier(self) -> str:
        return GROUNDED_ANSWER_PROMPT_ID

    def generate(
        self,
        question: str,
        retrieval_results: Iterable[RetrievalResult],
        *,
        config: GenerationConfig | None = None,
    ) -> GeneratedAnswer:
        """Answer one question from bounded retrieval evidence.

        Input and budget validation happen before model inference. No accepted
        evidence returns the deterministic fallback without calling the model;
        otherwise the method performs generation and attaches citations only to
        non-abstaining output. Retrieved documents are not mutated.
        """
        if not isinstance(question, str):
            raise TypeError("question must be a string.")
        if not question.strip():
            raise GenerationInputError("question cannot be empty.")
        if config is not None and not isinstance(config, GenerationConfig):
            raise TypeError("config must be a GenerationConfig.")

        settings = config or GenerationConfig()
        input_token_limit = resolve_input_token_limit(
            self._tokenizer,
            configured_limit=settings.max_input_tokens,
        )
        if settings.token_safety_margin >= input_token_limit:
            raise InvalidGenerationConfigurationError(
                "token_safety_margin must be smaller than the tokenizer input limit."
            )
        packed_context = pack_retrieval_context(
            question.strip(),
            retrieval_results,
            tokenizer=self._tokenizer,
            max_characters=settings.max_context_characters,
            input_token_limit=input_token_limit,
            token_safety_margin=settings.token_safety_margin,
        )
        if not packed_context.evidence:
            return GeneratedAnswer(
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                model_identifier=self._model_identifier,
                prompt_identifier=GROUNDED_ANSWER_PROMPT_ID,
                used_context=(),
                citations=(),
                context_characters=0,
                context_was_truncated=packed_context.was_truncated,
                prompt_tokens=0,
                prompt_token_limit=input_token_limit,
                generated=False,
            )

        used_context = tuple(item.retrieval_result for item in packed_context.evidence)
        citations = tuple(
            build_citation(
                item.retrieval_result,
                number=number,
                evidence_text=item.evidence_text,
            )
            for number, item in enumerate(packed_context.evidence, start=1)
        )

        try:
            answer = self._chain.invoke(
                {
                    "question": question.strip(),
                    "context": packed_context.rendered_context,
                    "insufficient_answer": INSUFFICIENT_CONTEXT_ANSWER,
                }
            )
        except Exception as exc:
            raise GenerationProviderError(
                f"Generation model {self._model_identifier} failed."
            ) from exc

        if not isinstance(answer, str) or not answer.strip():
            raise GenerationProviderError(
                f"Generation model {self._model_identifier} returned an empty answer."
            )
        normalized_answer = answer.strip()
        answer_citations = (
            () if normalized_answer == INSUFFICIENT_CONTEXT_ANSWER else citations
        )

        return GeneratedAnswer(
            answer=normalized_answer,
            model_identifier=self._model_identifier,
            prompt_identifier=GROUNDED_ANSWER_PROMPT_ID,
            used_context=used_context,
            citations=answer_citations,
            context_characters=len(packed_context.rendered_context),
            context_was_truncated=packed_context.was_truncated,
            prompt_tokens=packed_context.prompt_tokens,
            prompt_token_limit=input_token_limit,
            generated=True,
        )


def create_local_answer_generator(
    config: LocalGenerationConfig | None = None,
) -> AnswerGenerator:
    """Initialize the local Hugging Face text-to-text generation service.

    Construction may download/cache model artifacts and allocate CPU/GPU
    resources. The factory also captures the provider tokenizer used for exact
    prompt budgeting; import, initialization, or tokenizer failures become
    ``GenerationProviderError``.
    """
    settings = config or LocalGenerationConfig()

    try:
        from langchain_huggingface import HuggingFacePipeline
    except ImportError as exc:
        raise GenerationProviderError(
            "Local generation requires langchain-huggingface and transformers."
        ) from exc

    model_kwargs: dict[str, str] = {}
    if settings.model_revision is not None:
        model_kwargs["revision"] = settings.model_revision

    pipeline_kwargs: dict[str, object] = {
        "max_new_tokens": settings.max_new_tokens,
        "do_sample": settings.temperature > 0,
        "truncation": True,
    }
    if settings.temperature > 0:
        pipeline_kwargs["temperature"] = float(settings.temperature)

    try:
        language_model = HuggingFacePipeline.from_model_id(
            model_id=settings.model_name,
            task="text2text-generation",
            device=_pipeline_device(settings.device),
            model_kwargs=model_kwargs,
            pipeline_kwargs=pipeline_kwargs,
            batch_size=1,
        )
    except Exception as exc:
        raise GenerationProviderError(
            f"Failed to initialize local generation model {settings.model_name}."
        ) from exc

    pipeline = getattr(language_model, "pipeline", None)
    tokenizer = getattr(pipeline, "tokenizer", None)
    if tokenizer is None:
        raise GenerationProviderError(
            f"Local generation model {settings.model_name} has no tokenizer."
        )

    return AnswerGenerator(
        language_model,
        model_identifier=settings.model_identifier,
        tokenizer=tokenizer,
    )


def _pipeline_device(device: str) -> int:
    """Map user-facing CPU/CUDA notation to Hugging Face pipeline indices.

    CPU maps to ``-1`` and bare CUDA to device zero; malformed or unsupported
    values fail before model initialization.
    """
    normalized = device.strip().lower()
    if normalized == "cpu":
        return -1
    if normalized == "cuda":
        return 0
    if normalized.startswith("cuda:"):
        index = normalized.removeprefix("cuda:")
        if index.isdigit():
            return int(index)
    raise InvalidGenerationConfigurationError(
        "device must be 'cpu', 'cuda', or 'cuda:<index>'."
    )


def _validate_non_empty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidGenerationConfigurationError(f"{name} must be a non-empty string.")
