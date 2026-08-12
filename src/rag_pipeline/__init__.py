"""Expose installed package metadata for the production-minded RAG pipeline.

Pipeline behavior lives in stage-specific modules; the package root intentionally
keeps a small public surface while the project evolves task by task.
"""

from importlib.metadata import version

__version__ = version("rag-pipeline")

__all__ = ["__version__"]
