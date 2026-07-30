"""Expose the stable parser and entry point for the modular CLI adapter."""

from rag_pipeline.cli.app import build_parser, main


__all__ = ["build_parser", "main"]
