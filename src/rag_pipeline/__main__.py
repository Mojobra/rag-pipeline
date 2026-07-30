"""Run the modular command-line adapter for ``python -m rag_pipeline``."""

from rag_pipeline.cli import build_parser, main


__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
