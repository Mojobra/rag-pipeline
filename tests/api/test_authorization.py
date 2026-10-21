"""Verify opt-in source authorization without network or model downloads."""

from __future__ import annotations

import json
import tempfile
import traceback
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from rag_pipeline.api import create_app
from rag_pipeline.api.authorization import load_access_policy

_ADMIN = "admin-" + "a" * 32
_ALICE = "alice-" + "b" * 32
_BOB = "bob-" + "c" * 32
_PRINCIPALS = {
    "principals": [
        {"id": "alice", "token": _ALICE, "groups": ["finance"]},
        {"id": "bob", "token": _BOB, "groups": []},
    ]
}
_POLICY = {
    "version": 1,
    "sources": [
        {"source": "expense.md", "groups": ["finance"]},
        {"source": "notes.md", "principals": ["alice"]},
    ],
}


class SourceAuthorizationTests(unittest.TestCase):
    """Keep a configured policy fail-closed and separate query/admin tokens."""

    def setUp(self) -> None:
        """Create a private temporary policy and isolated API environment."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.policy_path = Path(directory.name) / "policy.json"
        self.policy_path.write_text(json.dumps(_POLICY), encoding="utf-8")
        environment = patch.dict(
            "os.environ",
            {
                "RAG_API_BEARER_TOKEN": _ADMIN,
                "RAG_API_PRINCIPALS_JSON": json.dumps(_PRINCIPALS),
                "RAG_API_ACCESS_POLICY_PATH": str(self.policy_path),
            },
            clear=True,
        )
        environment.start()
        self.addCleanup(environment.stop)

    def test_policy_maps_exact_sources_and_denies_unlisted_callers(self) -> None:
        """A group or direct grant gives access, but missing grants give none."""
        policy = load_access_policy(sha256(_ADMIN.encode("ascii")).digest())
        self.assertIsNotNone(policy)
        assert policy is not None
        self.assertEqual(policy.sources_for_token(_ALICE), ("expense.md", "notes.md"))
        self.assertEqual(policy.sources_for_token(_BOB), ())
        self.assertIsNone(policy.sources_for_token(_ADMIN))
        self.assertIsNone(policy.sources_for_token("unknown-" + "x" * 32))

    def test_query_routes_pass_server_scope_and_reject_admin_token(self) -> None:
        """Neither request fields nor the metrics token can broaden query scope."""
        runtime = Mock()
        runtime.retrieve.return_value = []
        runtime.answer.return_value.answer = "No evidence."
        runtime.answer.return_value.generated_answer = None
        with TestClient(create_app(lambda: runtime)) as client:
            self.assertEqual(
                client.get(
                    "/metrics", headers={"Authorization": f"Bearer {_ADMIN}"}
                ).status_code,
                200,
            )
            for route in ("/v1/retrieve", "/v1/answer"):
                self.assertEqual(
                    client.post(
                        route,
                        headers={"Authorization": f"Bearer {_ADMIN}"},
                        json={"query": "expense?"},
                    ).status_code,
                    401,
                )
                self.assertEqual(
                    client.post(
                        route,
                        headers={"Authorization": f"Bearer {_BOB}"},
                        json={"query": "expense?", "source_allowlist": ["expense.md"]},
                    ).status_code,
                    422,
                )
            self.assertEqual(
                client.post(
                    "/v1/retrieve",
                    headers={"Authorization": f"Bearer {_ALICE}"},
                    json={"query": "expense?"},
                ).status_code,
                200,
            )
            runtime.retrieve.assert_called_once_with(
                "expense?",
                top_k=4,
                score_threshold=None,
                source_allowlist=("expense.md", "notes.md"),
            )
            self.assertEqual(
                client.post(
                    "/v1/answer",
                    headers={"Authorization": f"Bearer {_BOB}"},
                    json={"query": "expense?"},
                ).status_code,
                200,
            )
            runtime.answer.assert_called_once_with(
                "expense?", top_k=4, score_threshold=0.2, source_allowlist=()
            )
        runtime.close.assert_called_once()

    def test_partial_or_malformed_configuration_fails_startup(self) -> None:
        """Do not silently fall back to unrestricted queries on policy errors."""
        cases = (
            {"RAG_API_ACCESS_POLICY_PATH": ""},
            {"RAG_API_PRINCIPALS_JSON": ""},
            {"RAG_API_PRINCIPALS_JSON": "not-json"},
            {"RAG_API_PRINCIPALS_JSON": '{"principals":[],"principals":[]}'},
            {"RAG_API_ACCESS_POLICY_PATH": str(self.policy_path / "missing")},
            {
                "RAG_API_PRINCIPALS_JSON": json.dumps(
                    {"principals": [{"id": "alice", "token": _ADMIN}]}
                )
            },
        )
        for values in cases:
            with self.subTest(values=values), patch.dict("os.environ", values):
                factory = Mock()
                with self.assertRaises(ValueError):
                    with TestClient(create_app(factory)):
                        pass
                factory.assert_not_called()

    def test_parser_errors_do_not_expose_secret_input(self) -> None:
        """Startup diagnostics must not print a malformed credential value."""
        secret = "sensitive-token-" + "z" * 32
        with patch.dict(
            "os.environ",
            {"RAG_API_PRINCIPALS_JSON": '{"principals":' + secret},
        ):
            with self.assertRaises(ValueError) as caught:
                load_access_policy(b"admin-digest")
        rendered = "".join(traceback.format_exception(caught.exception))
        self.assertNotIn(secret, rendered)

    def test_invalid_grants_and_duplicate_sources_fail_startup(self) -> None:
        """Typos, duplicate grants, and unknown principals must not broaden access."""
        bad_sources = (
            [{"source": "x", "principals": ["missing"]}],
            [{"source": "x", "groups": ["finance"]}] * 2,
            [{"source": "x", "principals": [], "groups": []}],
            [{"source": " ", "groups": ["finance"]}],
        )
        for sources in bad_sources:
            with self.subTest(sources=sources):
                self.policy_path.write_text(
                    json.dumps({"version": 1, "sources": sources}), encoding="utf-8"
                )
                with self.assertRaises(ValueError):
                    load_access_policy(sha256(_ADMIN.encode("ascii")).digest())


if __name__ == "__main__":
    unittest.main()
