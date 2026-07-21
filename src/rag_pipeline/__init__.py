"""Expose stable package metadata for the production-minded RAG pipeline.

Pipeline behavior lives in stage-specific modules; the package root intentionally
keeps a small public surface while the project evolves task by task.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
