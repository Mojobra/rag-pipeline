"""Expose the local read-only HTTP adapter without initializing model providers."""

from rag_pipeline.api.app import app, create_app

__all__ = ["app", "create_app"]
