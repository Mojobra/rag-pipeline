"""Expose bounded, process-local HTTP and RAG outcome metrics."""

from __future__ import annotations

from time import perf_counter

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class ApiMetrics:
    """Own one API application's Prometheus registry and fixed metric schema.

    A per-app registry isolates test instances and avoids exporting default
    process collectors. Values are in memory and reset when the process exits.
    """

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "rag_api_http_requests_total",
            "Completed HTTP requests, grouped by bounded route and status.",
            ("method", "route", "status_code"),
            registry=self.registry,
        )
        self.duration = Histogram(
            "rag_api_http_request_duration_seconds",
            "Elapsed HTTP request time in seconds.",
            ("method", "route"),
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
            registry=self.registry,
        )
        self.in_flight = Gauge(
            "rag_api_http_in_flight_requests",
            "HTTP requests currently being processed, excluding metrics scrapes.",
            registry=self.registry,
        )
        self.retrieval_outcomes = Counter(
            "rag_api_retrieval_outcomes_total",
            "Completed retrieval calls with or without selected evidence.",
            ("outcome",),
            registry=self.registry,
        )
        self.answer_outcomes = Counter(
            "rag_api_answer_outcomes_total",
            "Completed answers grouped by model invocation and citation outcome.",
            ("outcome",),
            registry=self.registry,
        )

    def observe_request(
        self, *, method: str, route: str, status_code: int, seconds: float
    ) -> None:
        """Record a completed request without user-controlled metric labels.

        The route must be a registered template or the fixed unmatched marker.
        Unknown methods collapse to OTHER; status codes are numeric HTTP values.
        """
        safe_method = method if method in {"GET", "POST"} else "OTHER"
        self.requests.labels(safe_method, route, str(status_code)).inc()
        self.duration.labels(safe_method, route).observe(seconds)

    def record_retrieval(self, *, has_evidence: bool) -> None:
        """Count a successful query by whether any chunks survived retrieval."""
        self.retrieval_outcomes.labels("nonempty" if has_evidence else "empty").inc()

    def record_answer(self, *, generated: bool, cited: bool) -> None:
        """Distinguish no model call, cited output, and model abstention.

        A generated answer without citations is an abstention under the current
        answer contract; this does not measure factual correctness.
        """
        outcome = "no_evidence" if not generated else "cited" if cited else "abstained"
        self.answer_outcomes.labels(outcome).inc()


class RequestMetricsMiddleware:
    """Observe HTTP traffic without inspecting request or response content."""

    def __init__(self, app: ASGIApp, metrics: ApiMetrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Count elapsed request time and status, including handled failures.

        Scrapes bypass instrumentation, and unhandled exceptions count as 500
        if no response status has been sent. The outer server retains error
        handling; this middleware only observes traffic.
        """
        if scope["type"] != "http" or scope["path"] == "/metrics":
            await self.app(scope, receive, send)
            return

        started_at = perf_counter()
        status_code = 500
        self.metrics.in_flight.inc()

        async def capture_status(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, capture_status)
        finally:
            self.metrics.in_flight.dec()
            route_path = getattr(scope.get("route"), "path", None)
            self.metrics.observe_request(
                method=scope["method"],
                route=route_path if isinstance(route_path, str) else "<unmatched>",
                status_code=status_code,
                seconds=perf_counter() - started_at,
            )
