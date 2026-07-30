"""Register all CLI commands in their stable user-facing help order."""

from __future__ import annotations

import argparse

from rag_pipeline.cli.commands.benchmarks import register_benchmark_commands
from rag_pipeline.cli.commands.documents import register_document_commands
from rag_pipeline.cli.commands.evaluation import register_evaluation_commands
from rag_pipeline.cli.commands.indexing import register_indexing_commands
from rag_pipeline.cli.commands.query import (
    register_answer_command,
    register_retrieve_command,
)


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Attach every command parser and its explicit execution handler.

    Registration order is intentionally stable because argparse exposes it in
    root help output and users scan commands in pipeline workflow order.
    """
    register_document_commands(subparsers)
    register_indexing_commands(subparsers)
    register_retrieve_command(subparsers)
    register_evaluation_commands(subparsers)
    register_benchmark_commands(subparsers)
    register_answer_command(subparsers)
