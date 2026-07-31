"""Protect the modular CLI's dispatch, help, and pure output contracts."""

from __future__ import annotations

import argparse
import io
import unittest
from contextlib import redirect_stdout

EXPECTED_COMMANDS = (
    "ingest",
    "chunk",
    "chunk-experiment",
    "embed",
    "index",
    "retrieve",
    "evaluate-retrieval",
    "evaluate-answer",
    "benchmark",
    "compare-benchmarks",
    "answer",
)


class CliArchitectureTests(unittest.TestCase):
    """Verify explicit command dispatch without executing providers or storage."""

    def test_entry_point_reexports_modular_cli_api(self) -> None:
        from rag_pipeline.__main__ import (
            build_parser as entry_build_parser,
        )
        from rag_pipeline.__main__ import (
            main as entry_main,
        )
        from rag_pipeline.cli import build_parser, main

        self.assertIs(entry_build_parser, build_parser)
        self.assertIs(entry_main, main)

    def test_every_command_has_one_explicit_handler_in_stable_order(self) -> None:
        from rag_pipeline.cli import build_parser

        parser = build_parser()
        subparser_actions = [
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        ]

        self.assertEqual(len(subparser_actions), 1)
        command_parsers = subparser_actions[0].choices
        self.assertEqual(tuple(command_parsers), EXPECTED_COMMANDS)
        for command, command_parser in command_parsers.items():
            with self.subTest(command=command):
                self.assertTrue(callable(command_parser.get_default("_handler")))

    def test_help_renders_for_root_and_every_subcommand(self) -> None:
        from rag_pipeline.cli import build_parser

        help_cases = [
            ["--help"],
            *[[command, "--help"] for command in EXPECTED_COMMANDS],
        ]
        for arguments in help_cases:
            with self.subTest(arguments=arguments):
                output = io.StringIO()
                with redirect_stdout(output):
                    with self.assertRaises(SystemExit) as raised:
                        build_parser().parse_args(arguments)

                self.assertEqual(raised.exception.code, 0)
                self.assertIn("usage:", output.getvalue())

    def test_benchmark_parser_builds_selected_chunking_contract(self) -> None:
        from rag_pipeline.cli import build_parser
        from rag_pipeline.cli.config import build_benchmark_config
        from rag_pipeline.ingestion.chunking import StructureAwareChunkingConfig
        from rag_pipeline.ingestion.semantic_chunking import SemanticChunkingConfig

        cases = (
            ("structure-aware", StructureAwareChunkingConfig),
            ("semantic", SemanticChunkingConfig),
        )
        for strategy, expected_type in cases:
            with self.subTest(strategy=strategy):
                args = build_parser().parse_args(
                    [
                        "benchmark",
                        "corpus",
                        "retrieval.json",
                        "answers.json",
                        "--output",
                        "artifact.json",
                        "--chunking-strategy",
                        strategy,
                    ]
                )

                config = build_benchmark_config(args)

                self.assertIsInstance(config.chunking, expected_type)

    def test_benchmark_config_builder_rejects_unknown_chunking_strategy(self) -> None:
        from rag_pipeline.cli import build_parser
        from rag_pipeline.cli.config import build_benchmark_chunking_config
        from rag_pipeline.exceptions import InvalidChunkingConfigurationError

        args = build_parser().parse_args(
            [
                "benchmark",
                "corpus",
                "retrieval.json",
                "answers.json",
                "--output",
                "artifact.json",
            ]
        )
        args.chunking_strategy = "unknown"

        with self.assertRaisesRegex(
            InvalidChunkingConfigurationError,
            "recursive, structure-aware, or semantic",
        ):
            build_benchmark_chunking_config(args)


class CliOutputTests(unittest.TestCase):
    """Verify terminal formatting independently from command orchestration."""

    def test_formats_retrieval_provenance_and_content_preview(self) -> None:
        from langchain_core.documents import Document

        from rag_pipeline.cli.output import format_retrieval_results
        from rag_pipeline.retrieval import RetrievalResult

        result = RetrievalResult(
            document=Document(
                page_content="Expense claims require itemized receipts.",
                metadata={
                    "source": "expenses.md",
                    "page": 0,
                    "chunk_index": 2,
                },
            ),
            score=0.91,
            rank=1,
            score_kind="cross_encoder",
            retrieval_score=0.75,
            retrieval_rank=3,
            retrieval_score_kind="cosine",
            reranker_model="test-reranker",
        )

        rendered = format_retrieval_results([result])

        self.assertIn(
            "1. score=0.9100 source=expenses.md page=1 chunk=2",
            rendered,
        )
        self.assertIn("retrieval_rank=3 retrieval_score=0.7500", rendered)
        self.assertIn(
            "Expense claims require itemized receipts.",
            rendered,
        )

    def test_formats_answer_with_structured_sources(self) -> None:
        from rag_pipeline.cli.output import format_generated_answer
        from rag_pipeline.generation import GeneratedAnswer
        from rag_pipeline.generation.citations import Citation

        answer = GeneratedAnswer(
            answer="Itemized receipts are required.",
            model_identifier="test-model",
            prompt_identifier="test-prompt",
            used_context=(),
            citations=(
                Citation(
                    number=1,
                    source="expenses.md",
                    page_number=None,
                    chunk_index=2,
                    start_index=0,
                    end_index=20,
                    chunk_id="chunk-2",
                    retrieval_rank=1,
                    retrieval_score=0.91,
                    excerpt="Itemized receipts.",
                ),
            ),
            context_characters=20,
            context_was_truncated=False,
            prompt_tokens=30,
            prompt_token_limit=256,
            generated=True,
        )

        rendered = format_generated_answer(answer)

        self.assertTrue(rendered.startswith("Answer:\nItemized receipts are required."))
        self.assertIn(
            "Sources:\n[1] expenses.md (chunk 3, characters 0-20)",
            rendered,
        )


if __name__ == "__main__":
    unittest.main()
