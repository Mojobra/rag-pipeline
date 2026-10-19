"""Record privacy-conscious HTTP request events at the API boundary."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def default_request_logger() -> logging.Logger:
    """Create an isolated stderr logger that emits one JSON object per event.

    This does not alter root or Uvicorn logging configuration. App factories
    may instead inject a logger, which keeps offline tests deterministic.
    """
    logger = logging.Logger("rag_pipeline.api.requests", level=logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class RequestLoggingMiddleware:
    """Correlate HTTP responses with bounded operational request events.

    Records an available route template, method, status, duration, and
    generated ID. Request bodies, headers, query strings, and raw paths are
    deliberately excluded because RAG questions and credentials are sensitive.
    """

    def __init__(self, app: ASGIApp, logger: logging.Logger) -> None:
        self.app = app
        self.logger = logger

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass through one ASGI request and log it after completion.

        A fresh ID is generated server-side, ignoring untrusted client IDs.
        Unhandled exceptions are re-raised; if no response has started, the
        event records 500 and the outer ASGI error handler owns the response.
        """
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid4().hex
        started_at = perf_counter()
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)["x-request-id"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            route_path = getattr(scope.get("route"), "path", None)
            event = {
                "timestamp": datetime.now(UTC).isoformat(),
                "request_id": request_id,
                "method": scope["method"],
                "route": route_path if isinstance(route_path, str) else "<unmatched>",
                "status_code": status_code,
                "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            }
            level = logging.ERROR if status_code >= 500 else logging.INFO
            self.logger.log(level, json.dumps(event, separators=(",", ":")))
