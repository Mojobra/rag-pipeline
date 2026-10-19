"""Expose authenticated local retrieval and grounded answers over HTTP.

The adapter has no authorization or document write endpoints. A shared bearer
credential protects query routes, but the service must remain on loopback.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from logging import Logger
from typing import cast

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from rag_pipeline import __version__
from rag_pipeline.api.auth import load_bearer_token_digest, require_bearer_token
from rag_pipeline.api.monitoring import ApiMetrics, RequestMetricsMiddleware
from rag_pipeline.api.request_logging import (
    RequestLoggingMiddleware,
    default_request_logger,
)
from rag_pipeline.api.runtime import LocalApiRuntime, QueryRuntime
from rag_pipeline.api.schemas import (
    AnswerRequest,
    AnswerResponse,
    HealthResponse,
    QueryRequest,
    RetrievalHit,
    RetrievalResponse,
)
from rag_pipeline.api.settings import load_api_settings
from rag_pipeline.exceptions import (
    GenerationInputError,
    RagPipelineError,
    RetrievalInputError,
    VectorStoreCollectionNotFoundError,
    VectorStoreCompatibilityError,
)


def create_app(
    runtime_factory: Callable[[], QueryRuntime] | None = None,
    *,
    request_logger: Logger | None = None,
) -> FastAPI:
    """Build an HTTP adapter with a replaceable, lifespan-owned query runtime.

    Importing the app never loads models or opens Qdrant. Startup requires a
    configured bearer secret; the real runtime initializes providers on the
    first authenticated query. Tests can inject an offline runtime and logger.
    """

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.bearer_token_digest = load_bearer_token_digest()
        runtime = (
            LocalApiRuntime(load_api_settings())
            if runtime_factory is None
            else runtime_factory()
        )
        application.state.query_runtime = runtime
        try:
            yield
        finally:
            runtime.close()

    application = FastAPI(
        title="RAG Pipeline API",
        version=__version__,
        lifespan=lifespan,
    )
    metrics = ApiMetrics()
    application.add_middleware(
        RequestLoggingMiddleware,
        logger=request_logger
        if request_logger is not None
        else default_request_logger(),
    )
    application.add_middleware(RequestMetricsMiddleware, metrics=metrics)

    @application.exception_handler(RagPipelineError)
    async def handle_pipeline_error(
        _request: Request, exc: RagPipelineError
    ) -> JSONResponse:
        """Map known input/storage failures without leaking provider internals."""
        if isinstance(exc, (RetrievalInputError, GenerationInputError)):
            return JSONResponse(status_code=422, content={"detail": str(exc)})
        if isinstance(exc, VectorStoreCollectionNotFoundError):
            return JSONResponse(status_code=404, content={"detail": str(exc)})
        if isinstance(exc, VectorStoreCompatibilityError):
            return JSONResponse(status_code=409, content={"detail": str(exc)})
        return JSONResponse(
            status_code=503,
            content={"detail": "Local RAG service is unavailable."},
        )

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Report process liveness without loading a model or querying Qdrant."""
        return HealthResponse()

    @application.get(
        "/metrics",
        response_class=PlainTextResponse,
        dependencies=[Depends(require_bearer_token)],
    )
    def scrape_metrics() -> Response:
        """Expose process-local aggregates to an authorized loopback scraper."""
        return Response(
            content=generate_latest(metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
            headers={"Cache-Control": "no-store"},
        )

    @application.post(
        "/v1/retrieve",
        response_model=RetrievalResponse,
        dependencies=[Depends(require_bearer_token)],
    )
    def retrieve(body: QueryRequest, request: Request) -> RetrievalResponse:
        """Search the configured collection and return ordered source evidence."""
        runtime = cast(QueryRuntime, request.app.state.query_runtime)
        results = runtime.retrieve(
            body.query,
            top_k=body.top_k,
            score_threshold=body.score_threshold,
        )
        response = RetrievalResponse(
            results=[RetrievalHit.from_result(result) for result in results]
        )
        metrics.record_retrieval(has_evidence=bool(results))
        return response

    @application.post(
        "/v1/answer",
        response_model=AnswerResponse,
        dependencies=[Depends(require_bearer_token)],
    )
    def answer(body: AnswerRequest, request: Request) -> AnswerResponse:
        """Return a local grounded answer with deterministic source citations."""
        runtime = cast(QueryRuntime, request.app.state.query_runtime)
        outcome = runtime.answer(
            body.query,
            top_k=body.top_k,
            score_threshold=body.score_threshold,
        )
        response = AnswerResponse.from_outcome(outcome)
        metrics.record_answer(
            generated=response.generated, cited=bool(response.citations)
        )
        return response

    return application


app = create_app()
