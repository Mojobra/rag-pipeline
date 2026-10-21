"""Test bounded API metrics and the protected scrape without external services."""

from __future__ import annotations

import logging
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families

from rag_pipeline.api import create_app
from rag_pipeline.api.monitoring import ApiMetrics
from rag_pipeline.application.answering import AnswerOutcome
from rag_pipeline.exceptions import GenerationProviderError
from rag_pipeline.generation import INSUFFICIENT_CONTEXT_ANSWER

_TEST_TOKEN = "test-only-" + "x" * 32
_AUTH_HEADERS = {"Authorization": f"Bearer {_TEST_TOKEN}"}


def sample_value(exposition: str, name: str, labels: dict[str, str]) -> float | None:
    """Find a precise sample in Prometheus' parsed exposition format."""
    for family in text_string_to_metric_families(exposition):
        for sample in family.samples:
            if sample.name == name and sample.labels == labels:
                return float(sample.value)
    return None


class ApiMonitoringTests(unittest.TestCase):
    """Verify operational visibility without exposing queries or credentials."""

    def setUp(self) -> None:
        """Use a valid dummy token and an isolated, silent request logger."""
        token_patch = patch.dict("os.environ", {"RAG_API_BEARER_TOKEN": _TEST_TOKEN})
        token_patch.start()
        self.addCleanup(token_patch.stop)
        self.runtime = Mock()
        self.runtime.retrieve.return_value = []
        self.runtime.answer.return_value = AnswerOutcome(
            INSUFFICIENT_CONTEXT_ANSWER, None
        )
        self.logger = logging.Logger("test.monitoring")
        self.logger.addHandler(logging.NullHandler())

    def app(self) -> FastAPI:
        """Create a fresh app so counter values never leak between tests."""
        return create_app(lambda: self.runtime, request_logger=self.logger)

    def test_scrape_requires_bearer_and_does_not_load_providers(self) -> None:
        """Keep the metrics surface private and free of model I/O."""
        with TestClient(self.app()) as client:
            missing = client.get("/metrics")
            wrong = client.get("/metrics", headers={"Authorization": "Bearer wrong"})
            allowed = client.get("/metrics", headers=_AUTH_HEADERS)
            again = client.get("/metrics", headers=_AUTH_HEADERS)
            schema = client.get("/openapi.json").json()

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(again.status_code, 200)
        self.assertIn("text/plain", allowed.headers["content-type"])
        self.assertEqual(allowed.headers["cache-control"], "no-store")
        self.assertIn("security", schema["paths"]["/metrics"]["get"])
        self.assertIn(
            "text/plain",
            schema["paths"]["/metrics"]["get"]["responses"]["200"]["content"],
        )
        self.assertEqual(len(allowed.headers["x-request-id"]), 32)
        self.assertNotEqual(
            allowed.headers["x-request-id"], again.headers["x-request-id"]
        )
        self.assertIsNone(
            sample_value(
                allowed.text,
                "rag_api_http_requests_total",
                {"method": "GET", "route": "/metrics", "status_code": "200"},
            )
        )
        self.assertNotIn(_TEST_TOKEN, allowed.text)
        self.runtime.retrieve.assert_not_called()
        self.runtime.answer.assert_not_called()

    def test_request_and_rag_outcome_samples_are_bounded(self) -> None:
        """Count real queries, failures, and abstention without content labels."""
        question = "private expense question"
        with TestClient(self.app()) as client:
            self.assertEqual(client.get("/health").status_code, 200)
            self.assertEqual(
                client.post("/v1/retrieve", json={"query": question}).status_code,
                401,
            )
            self.assertEqual(
                client.post(
                    "/v1/retrieve",
                    json={"query": " "},
                    headers=_AUTH_HEADERS,
                ).status_code,
                422,
            )
            self.assertEqual(
                client.post(
                    "/v1/retrieve?token=private-url-secret",
                    json={"query": question},
                    headers=_AUTH_HEADERS,
                ).status_code,
                200,
            )
            self.assertEqual(
                client.post(
                    "/v1/answer",
                    json={"query": question},
                    headers=_AUTH_HEADERS,
                ).status_code,
                200,
            )
            self.assertEqual(client.request("PATCH", "/private/path").status_code, 404)
            scrape = client.get("/metrics", headers=_AUTH_HEADERS)

        self.assertEqual(scrape.status_code, 200)
        self.assertEqual(
            sample_value(
                scrape.text,
                "rag_api_http_requests_total",
                {"method": "POST", "route": "/v1/retrieve", "status_code": "401"},
            ),
            1,
        )
        self.assertEqual(
            sample_value(
                scrape.text,
                "rag_api_http_requests_total",
                {"method": "POST", "route": "/v1/retrieve", "status_code": "422"},
            ),
            1,
        )
        self.assertEqual(
            sample_value(
                scrape.text,
                "rag_api_http_requests_total",
                {"method": "POST", "route": "/v1/retrieve", "status_code": "200"},
            ),
            1,
        )
        self.assertEqual(
            sample_value(
                scrape.text,
                "rag_api_http_requests_total",
                {"method": "OTHER", "route": "<unmatched>", "status_code": "404"},
            ),
            1,
        )
        self.assertEqual(
            sample_value(
                scrape.text,
                "rag_api_http_request_duration_seconds_count",
                {"method": "POST", "route": "/v1/retrieve"},
            ),
            3,
        )
        self.assertEqual(
            sample_value(scrape.text, "rag_api_http_in_flight_requests", {}), 0
        )
        self.assertEqual(
            sample_value(
                scrape.text, "rag_api_retrieval_outcomes_total", {"outcome": "empty"}
            ),
            1,
        )
        self.assertEqual(
            sample_value(
                scrape.text,
                "rag_api_answer_outcomes_total",
                {"outcome": "no_evidence"},
            ),
            1,
        )
        for secret in (question, _TEST_TOKEN, "private-url-secret", "/private/path"):
            self.assertNotIn(secret, scrape.text)
        self.runtime.retrieve.assert_called_once()
        self.runtime.answer.assert_called_once()

    def test_known_and_unhandled_failures_are_counted(self) -> None:
        """Distinguish 503 from unexpected 500 without exception text."""
        self.runtime.retrieve.side_effect = GenerationProviderError("private detail")
        with TestClient(self.app(), raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/retrieve", json={"query": "policy?"}, headers=_AUTH_HEADERS
            )
            self.runtime.retrieve.side_effect = ValueError("unhandled private detail")
            unexpected = client.post(
                "/v1/retrieve", json={"query": "policy?"}, headers=_AUTH_HEADERS
            )
            scrape = client.get("/metrics", headers=_AUTH_HEADERS)

        self.assertEqual(response.status_code, 503)
        self.assertEqual(unexpected.status_code, 500)
        for status in ("503", "500"):
            self.assertEqual(
                sample_value(
                    scrape.text,
                    "rag_api_http_requests_total",
                    {"method": "POST", "route": "/v1/retrieve", "status_code": status},
                ),
                1,
            )
        self.assertEqual(
            sample_value(scrape.text, "rag_api_http_in_flight_requests", {}), 0
        )
        self.assertNotIn("private detail", scrape.text)

    def test_outcome_counter_has_only_fixed_values(self) -> None:
        """Track cited and model-abstained answers without model identifiers."""
        metrics = ApiMetrics()
        metrics.record_retrieval(has_evidence=True)
        metrics.record_answer(generated=True, cited=True)
        metrics.record_answer(generated=True, cited=False)
        exposition = generate_latest(metrics.registry).decode("utf-8")
        for name, outcome in (
            ("rag_api_retrieval_outcomes_total", "nonempty"),
            ("rag_api_answer_outcomes_total", "cited"),
            ("rag_api_answer_outcomes_total", "abstained"),
        ):
            self.assertEqual(sample_value(exposition, name, {"outcome": outcome}), 1)

    def test_app_registries_do_not_share_traffic(self) -> None:
        """Keep app factories deterministic rather than using a global registry."""
        with TestClient(self.app()) as first, TestClient(self.app()) as second:
            first.get("/health")
            first_metrics = first.get("/metrics", headers=_AUTH_HEADERS)
            second_metrics = second.get("/metrics", headers=_AUTH_HEADERS)

        labels = {"method": "GET", "route": "/health", "status_code": "200"}
        self.assertEqual(
            sample_value(first_metrics.text, "rag_api_http_requests_total", labels),
            1,
        )
        self.assertIsNone(
            sample_value(second_metrics.text, "rag_api_http_requests_total", labels)
        )


if __name__ == "__main__":
    unittest.main()
