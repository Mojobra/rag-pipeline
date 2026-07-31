"""Assemble and dispatch the local RAG command-line application.

The root parser stays intentionally small: command modules own their arguments
and handlers, while this module preserves the package's public ``build_parser``
and ``main`` entry points.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import cast

from rag_pipeline import __version__
from rag_pipeline.cli.commands import register_commands

CommandHandler = Callable[[argparse.Namespace, argparse.ArgumentParser], int]


def build_parser() -> argparse.ArgumentParser:
    """Build the complete parser without initializing providers or storage."""
    parser = argparse.ArgumentParser(
        prog="rag_pipeline",
        description=(
            "Run individual stages of the local RAG pipeline, from document "
            "inspection through grounded answer generation."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"rag-pipeline {__version__}",
        help="Print the installed rag-pipeline version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")
    register_commands(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and execute the handler registered for one command.

    Handlers may read documents, initialize models, query or mutate Qdrant, and
    write terminal output. With no command, the historical readiness message is
    retained for backward compatibility.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(CommandHandler | None, getattr(args, "_handler", None))
    if handler is None:
        print("RAG Pipeline skeleton is ready.")
        return 0
    return handler(args, parser)
