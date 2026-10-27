"""Exercise fail-closed startup and bearer protection without model providers."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from rag_pipeline.api import create_app

_TEST_TOKEN = "test-only-" + "x" * 32


class ApiAuthenticationTests(unittest.TestCase):
    """Keep query authorization ahead of local RAG work and source disclosure."""

    def test_missing_or_malformed_server_token_fails_startup(self) -> None:
        """Refuse traffic rather than silently launching an open query API."""
        for value in (None, "", "too-short", "x" * 257, "x" * 31 + "!"):
            with self.subTest(value=value), patch.dict("os.environ", {}, clear=True):
                if value is not None:
                    with patch.dict("os.environ", {"RAG_API_BEARER_TOKEN": value}):
                        self._assert_startup_rejected()
                else:
                    self._assert_startup_rejected()

    def _assert_startup_rejected(self) -> None:
        factory = Mock()
        with self.assertRaisesRegex(ValueError, "RAG_API_BEARER_TOKEN"):
            with TestClient(create_app(factory)):
                pass
        factory.assert_not_called()

    def test_query_routes_require_valid_bearer_before_runtime(self) -> None:
        """Return one generic challenge for missing, wrong, or malformed auth."""
        runtime = Mock()
        runtime.retrieve.return_value = []
        with patch.dict("os.environ", {"RAG_API_BEARER_TOKEN": _TEST_TOKEN}):
            with TestClient(create_app(lambda: runtime)) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                self.assertEqual(client.get("/docs").status_code, 200)
                for route in ("/v1/retrieve", "/v1/answer"):
                    for headers in (
                        {},
                        {"Authorization": "Basic abc"},
                        {"Authorization": "Bearer wrong"},
                        {"Authorization": "Bearer " + "y" * 33},
                    ):
                        with self.subTest(route=route, headers=headers):
                            response = client.post(
                                route, json={"query": "policy?"}, headers=headers
                            )
                            self.assertEqual(response.status_code, 401)
                            self.assertEqual(
                                response.headers["www-authenticate"], "Bearer"
                            )
                            self.assertNotIn(_TEST_TOKEN, response.text)
                runtime.retrieve.assert_not_called()
                runtime.answer.assert_not_called()

                response = client.post(
                    "/v1/retrieve",
                    json={"query": "policy?"},
                    headers={"Authorization": f"Bearer {_TEST_TOKEN}"},
                )
                self.assertEqual(response.status_code, 200)
                runtime.retrieve.assert_called_once()

        runtime.close.assert_called_once()

    def test_openapi_declares_bearer_only_for_query_routes(self) -> None:
        """Advertise the protection so the interactive docs can authorize."""
        runtime = Mock()
        with patch.dict("os.environ", {"RAG_API_BEARER_TOKEN": _TEST_TOKEN}):
            with TestClient(create_app(lambda: runtime)) as client:
                schema = client.get("/openapi.json").json()
        schemes = schema["components"]["securitySchemes"]
        self.assertEqual(len(schemes), 1)
        self.assertEqual(next(iter(schemes.values()))["scheme"], "bearer")
        self.assertNotIn("security", schema["paths"]["/health"]["get"])
        for route in ("/v1/retrieve", "/v1/answer"):
            self.assertIn("security", schema["paths"][route]["post"])

    def test_environment_change_does_not_rotate_running_process(self) -> None:
        """Keep the startup credential stable until the ASGI lifespan restarts."""
        runtime = Mock()
        runtime.retrieve.return_value = []
        replacement = "replacement-" + "y" * 32
        with patch.dict("os.environ", {"RAG_API_BEARER_TOKEN": _TEST_TOKEN}):
            with TestClient(create_app(lambda: runtime)) as client:
                with patch.dict("os.environ", {"RAG_API_BEARER_TOKEN": replacement}):
                    accepted = client.post(
                        "/v1/retrieve",
                        json={"query": "policy?"},
                        headers={"Authorization": f"Bearer {_TEST_TOKEN}"},
                    )
                    rejected = client.post(
                        "/v1/retrieve",
                        json={"query": "policy?"},
                        headers={"Authorization": f"Bearer {replacement}"},
                    )
        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(rejected.status_code, 401)
        runtime.retrieve.assert_called_once()


if __name__ == "__main__":
    unittest.main()
