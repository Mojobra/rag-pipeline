"""Own lazy local provider lifecycles behind serialized HTTP query operations."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from threading import Lock
from typing import Protocol

from rag_pipeline.api.authorization import TenantStorage
from rag_pipeline.api.settings import ApiSettings
from rag_pipeline.application.answering import AnswerOutcome, answer_from_results
from rag_pipeline.application.retrieval import (
    RetrievalPipeline,
    open_local_retrieval_pipeline,
)
from rag_pipeline.exceptions import RetrievalProviderError
from rag_pipeline.generation import AnswerGenerator, create_local_answer_generator
from rag_pipeline.retrieval import RetrievalConfig, RetrievalResult


class QueryRuntime(Protocol):
    """Read-only query contract consumed by the HTTP transport and test doubles."""

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        score_threshold: float | None,
        source_allowlist: tuple[str, ...] | None = None,
        tenant_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Return ranked evidence for one bounded query."""

    def answer(
        self,
        query: str,
        *,
        top_k: int,
        score_threshold: float | None,
        source_allowlist: tuple[str, ...] | None = None,
        tenant_id: str | None = None,
    ) -> AnswerOutcome:
        """Return a grounded answer or no-evidence abstention."""

    def close(self) -> None:
        """Release open model and store resources at application shutdown."""


class LocalApiRuntime:
    """Reuse one embedded-Qdrant session and lazily loaded local models.

    A lock serializes operations because embedded storage and local inference
    are process-owned resources, not a multi-worker service. The first query
    opens the collection; only an answer with evidence loads the generator.
    """

    def __init__(self, settings: ApiSettings) -> None:
        self._settings = settings
        self._resources = ExitStack()
        self._lock = Lock()
        self._retrieval: RetrievalPipeline | None = None
        self._generator: AnswerGenerator | None = None
        self._closed = False

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        score_threshold: float | None,
        source_allowlist: tuple[str, ...] | None = None,
        tenant_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Run retrieval under the process lock, reusing initialized providers."""
        if tenant_id is not None:
            raise RetrievalProviderError("Tenant scope cannot use the default store.")
        with self._lock:
            if source_allowlist == ():
                return []
            results = self._pipeline().retrieve(
                query,
                retrieval_config=RetrievalConfig(
                    top_k=top_k,
                    score_threshold=score_threshold,
                    source_allowlist=source_allowlist,
                ),
            )
            _assert_authorized_sources(results, source_allowlist)
            return results

    def answer(
        self,
        query: str,
        *,
        top_k: int,
        score_threshold: float | None,
        source_allowlist: tuple[str, ...] | None = None,
        tenant_id: str | None = None,
    ) -> AnswerOutcome:
        """Retrieve and generate atomically against one open collection view."""
        if tenant_id is not None:
            raise RetrievalProviderError("Tenant scope cannot use the default store.")
        with self._lock:
            results = (
                []
                if source_allowlist == ()
                else self._pipeline().retrieve(
                    query,
                    retrieval_config=RetrievalConfig(
                        top_k=top_k,
                        score_threshold=score_threshold,
                        source_allowlist=source_allowlist,
                    ),
                )
            )
            _assert_authorized_sources(results, source_allowlist)
            return answer_from_results(
                query,
                results,
                generator_factory=self._answer_generator,
                config=self._settings.generation,
            )

    def close(self) -> None:
        """Release the embedded database once; later requests are invalid."""
        with self._lock:
            if not self._closed:
                self._resources.close()
                self._retrieval = None
                self._generator = None
                self._closed = True

    def _pipeline(self) -> RetrievalPipeline:
        if self._closed:
            raise RuntimeError("API runtime is closed.")
        if self._retrieval is None:
            self._retrieval = self._resources.enter_context(
                open_local_retrieval_pipeline(self._settings.retrieval)
            )
        return self._retrieval

    def _answer_generator(self) -> AnswerGenerator:
        if self._generator is None:
            self._generator = create_local_answer_generator(
                self._settings.generation_model
            )
        return self._generator


class TenantApiRuntime:
    """Route scoped queries to distinct local stores under one process lock.

    Per-tenant runtimes remain lazy, so startup does not open stores or models.
    The global lock bounds inference and embedded-store activity to one query
    at a time; this favors predictable local memory over tenant throughput.
    """

    def __init__(
        self, settings: ApiSettings, tenants: tuple[TenantStorage, ...]
    ) -> None:
        self._lock = Lock()
        self._closed = False
        self._runtimes: dict[str, LocalApiRuntime] = {}
        for tenant in tenants:
            store = replace(
                settings.retrieval.vector_store,
                path=tenant.store_path,
                collection_name=tenant.collection_name,
            )
            retrieval = replace(settings.retrieval, vector_store=store)
            self._runtimes[tenant.id] = LocalApiRuntime(
                replace(settings, retrieval=retrieval)
            )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        score_threshold: float | None,
        source_allowlist: tuple[str, ...] | None = None,
        tenant_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Search only the credential-bound tenant's configured collection."""
        with self._lock:
            return self._tenant_runtime(tenant_id, source_allowlist).retrieve(
                query,
                top_k=top_k,
                score_threshold=score_threshold,
                source_allowlist=source_allowlist,
            )

    def answer(
        self,
        query: str,
        *,
        top_k: int,
        score_threshold: float | None,
        source_allowlist: tuple[str, ...] | None = None,
        tenant_id: str | None = None,
    ) -> AnswerOutcome:
        """Generate only from evidence read within the selected tenant store."""
        with self._lock:
            return self._tenant_runtime(tenant_id, source_allowlist).answer(
                query,
                top_k=top_k,
                score_threshold=score_threshold,
                source_allowlist=source_allowlist,
            )

    def close(self) -> None:
        """Release each lazy tenant runtime once when the API shuts down."""
        with self._lock:
            if not self._closed:
                for runtime in self._runtimes.values():
                    runtime.close()
                self._closed = True

    def _tenant_runtime(
        self, tenant_id: str | None, source_allowlist: tuple[str, ...] | None
    ) -> LocalApiRuntime:
        if self._closed or tenant_id is None or source_allowlist is None:
            raise RetrievalProviderError("Tenant query scope is unavailable.")
        runtime = self._runtimes.get(tenant_id)
        if runtime is None:
            raise RetrievalProviderError("Tenant query scope is unavailable.")
        return runtime


def _assert_authorized_sources(
    results: list[RetrievalResult], source_allowlist: tuple[str, ...] | None
) -> None:
    """Fail closed if a provider ever returns evidence outside its query scope."""
    if source_allowlist is not None and any(
        result.document.metadata.get("source") not in source_allowlist
        for result in results
    ):
        raise RetrievalProviderError("Retrieved evidence violated the source policy.")
