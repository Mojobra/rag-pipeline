"""Keep CLI and HTTP answer orchestration consistent without model downloads."""

from __future__ import annotations

import unittest

from rag_pipeline.application.answering import answer_from_results
from rag_pipeline.exceptions import GenerationInputError
from rag_pipeline.generation import (
    INSUFFICIENT_CONTEXT_ANSWER,
    AnswerGenerator,
    GenerationConfig,
)


def fail_if_loaded() -> AnswerGenerator:
    """Make accidental generation-model startup fail the test immediately."""
    raise AssertionError("generator must not load")


class AnswerWorkflowTests(unittest.TestCase):
    """Verify no-evidence abstention precedes local model initialization."""

    def test_empty_results_skip_generator_factory(self) -> None:
        outcome = answer_from_results(
            "What is the policy?",
            [],
            generator_factory=fail_if_loaded,
            config=GenerationConfig(),
        )
        self.assertEqual(outcome.answer, INSUFFICIENT_CONTEXT_ANSWER)
        self.assertIsNone(outcome.generated_answer)

    def test_blank_question_is_rejected_even_without_results(self) -> None:
        with self.assertRaises(GenerationInputError):
            answer_from_results(
                "  ",
                [],
                generator_factory=fail_if_loaded,
                config=GenerationConfig(),
            )
