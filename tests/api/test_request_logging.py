"""Verify request correlation and redaction without model or network access."""

from __future__ import annotations

import io
import json
import logging
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from rag_pipeline.api import create_app

_TEST_TOKEN = "test-only-" + "x" * 32


class RequestLoggingTests(unittest.TestCase):
    """Keep HTTP diagnostics useful without retaining question or token data."""

    def setUp(self) -> None:
        """Capture isolated JSON events and supply an offline query runtime."""
        self.output = io.StringIO()
        self.logger = logging.Logger("test.api.requests", level=logging.INFO)
        self.logger.addHandler(logging.StreamHandler(self.output))
        self.runtime = Mock()
        self.runtime.retrieve.return_value = []
        token_patch = patch.dict("os.environ", {"RAG_API_BEARER_TOKEN": _TEST_TOKEN})
        token_patch.start()
        self.addCleanup(token_patch.stop)

    def events(self) -> list[dict[str, object]]:
        """Parse complete emitted lines so tests assert the public log schema."""
        return [json.loads(line) for line in self.output.getvalue().splitlines()]

    def test_success_and_handled_failures_have_correlation_ids(self) -> None:
        """Log public, authorized, unauthorized, and invalid requests once."""
        with TestClient(
            create_app(lambda: self.runtime, request_logger=self.logger)
        ) as client:
            responses = [
                client.get("/health"),
                client.post(
                    "/v1/retrieve",
                    json={"query": "receipt?"},
                    headers={"Authorization": f"Bearer {_TEST_TOKEN}"},
                ),
                client.post("/v1/retrieve", json={"query": "receipt?"}),
                client.post(
                    "/v1/answer",
                    json={"query": " "},
                    headers={"Authorization": f"Bearer {_TEST_TOKEN}"},
                ),
                client.get("/missing/private-path?token=hidden"),
            ]

        events = self.events()
        self.assertEqual(len(events), len(responses))
        self.assertEqual([r.status_code for r in responses], [200, 200, 401, 422, 404])
        self.assertEqual(
            [event["route"] for event in events],
            ["/health", "/v1/retrieve", "/v1/retrieve", "/v1/answer", "<unmatched>"],
        )
        self.assertEqual(
            [event["status_code"] for event in events], [200, 200, 401, 422, 404]
        )
        request_ids = [response.headers["x-request-id"] for response in responses]
        self.assertEqual(len(set(request_ids)), len(request_ids))
        for response, event in zip(responses, events, strict=True):
            self.assertEqual(event["request_id"], response.headers["x-request-id"])
            self.assertEqual(len(response.headers["x-request-id"]), 32)
            self.assertGreaterEqual(event["duration_ms"], 0)
            self.assertTrue(str(event["timestamp"]).endswith("+00:00"))
            self.assertEqual(
                set(event),
                {
                    "timestamp",
                    "request_id",
                    "method",
                    "route",
                    "status_code",
                    "duration_ms",
                },
            )
        self.runtime.retrieve.assert_called_once()

    def test_event_excludes_sensitive_request_data_and_client_id(self) -> None:
        """Never echo submitted questions, credentials, URLs, or caller IDs."""
        client_id = "untrusted-client-id"
        question = "private employee question"
        with TestClient(
            create_app(lambda: self.runtime, request_logger=self.logger)
        ) as client:
            response = client.post(
                "/v1/retrieve?access_token=private-query-token",
                json={"query": question},
                headers={
                    "Authorization": f"Bearer {_TEST_TOKEN}",
                    "X-Request-ID": client_id,
                },
            )

        event_text = self.output.getvalue()
        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(response.headers["x-request-id"], client_id)
        for secret in (_TEST_TOKEN, question, "private-query-token", client_id):
            self.assertNotIn(secret, event_text)
        self.assertEqual(self.events()[0]["route"], "/v1/retrieve")

    def test_unhandled_error_is_logged_without_exception_details(self) -> None:
        """Preserve exception behavior while recording a bounded 500 event."""
        self.runtime.retrieve.side_effect = ValueError("private provider detail")
        with TestClient(
            create_app(lambda: self.runtime, request_logger=self.logger)
        ) as client:
            with self.assertRaisesRegex(ValueError, "private provider detail"):
                client.post(
                    "/v1/retrieve",
                    json={"query": "receipt?"},
                    headers={"Authorization": f"Bearer {_TEST_TOKEN}"},
                )

        self.assertEqual(self.events()[0]["status_code"], 500)
        self.assertEqual(self.events()[0]["route"], "/v1/retrieve")
        self.assertNotIn("private provider detail", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
