"""Exercise the Graph adapter with deterministic HTTP and token doubles."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import httpx

from rag_pipeline.exceptions import (
    SharePointConfigurationError,
    SharePointInputError,
    SharePointProviderError,
)
from rag_pipeline.infrastructure.sharepoint import (
    MsalTokenProvider,
    SharePointFile,
    SharePointGraphClient,
    SharePointImportLimits,
)


class SharePointGraphClientTests(unittest.TestCase):
    """Check bounded traversal and safe content transport without live Graph."""

    def test_recursive_pagination_filters_formats_and_preserves_item_identity(
        self,
    ) -> None:
        requests: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            self.assertEqual(request.headers["Authorization"], "Bearer test-token")
            if "/items/folder/children" in request.url.path:
                return httpx.Response(
                    200,
                    json={
                        "value": [
                            {
                                "id": "nested",
                                "name": "policy.pdf",
                                "size": 5,
                                "file": {},
                                "webUrl": "https://tenant.sharepoint.com/policy.pdf",
                            }
                        ]
                    },
                )
            if request.url.params.get("$skiptoken") == "next":
                return httpx.Response(
                    200,
                    json={
                        "value": [
                            {"id": "folder", "name": "Nested", "folder": {}},
                            {"id": "skip", "name": "image.png", "file": {}},
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={
                    "value": [
                        {"id": "first", "name": "expense.md", "size": 7, "file": {}}
                    ],
                    "@odata.nextLink": (
                        "https://graph.microsoft.com/v1.0/drives/drive/root/children"
                        "?$skiptoken=next"
                    ),
                },
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
            reader = SharePointGraphClient(
                "drive", token_provider=lambda: "test-token", http_client=http_client
            )
            files = reader.list_files(SharePointImportLimits())

        self.assertEqual(len(requests), 3)
        self.assertEqual({file.item_id for file in files}, {"first", "nested"})
        self.assertEqual(files[0].source, "sharepoint://drive/first")
        self.assertEqual(files[1].web_url, "https://tenant.sharepoint.com/policy.pdf")
        renamed = SharePointFile("drive", "first", "renamed.md", 7)
        self.assertEqual(files[0].source, renamed.source)

    def test_rejects_unsafe_or_repeated_pagination(self) -> None:
        for next_link in (
            "https://elsewhere.example/v1.0/drives/drive/root/children",
            "https://graph.microsoft.com/v1.0/drives/other/root/children",
            "https://graph.microsoft.com/v1.0/drives/drive/root/children?$top=200",
            "https://[bad",
        ):
            with self.subTest(next_link=next_link):

                def respond(
                    _request: httpx.Request, *, link: str = next_link
                ) -> httpx.Response:
                    return httpx.Response(
                        200, json={"value": [], "@odata.nextLink": link}
                    )

                with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                    reader = SharePointGraphClient(
                        "drive", token_provider=lambda: "token", http_client=client
                    )
                    with self.assertRaises(SharePointInputError):
                        reader.list_files(SharePointImportLimits())

    def test_declared_size_and_file_count_limits_fail_before_download(self) -> None:
        def respond(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "value": [
                        {"id": "a", "name": "a.txt", "size": 6, "file": {}},
                        {"id": "b", "name": "b.txt", "size": 6, "file": {}},
                    ]
                },
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            reader = SharePointGraphClient(
                "drive", token_provider=lambda: "token", http_client=client
            )
            for limits in (
                SharePointImportLimits(max_files=1),
                SharePointImportLimits(max_total_bytes=10),
                SharePointImportLimits(max_file_bytes=5),
            ):
                with (
                    self.subTest(limits=limits),
                    self.assertRaises(SharePointInputError),
                ):
                    reader.list_files(limits)

    def test_download_redirect_does_not_forward_graph_token(self) -> None:
        requests: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if request.url.host == "graph.microsoft.com":
                return httpx.Response(
                    302,
                    headers={"location": "https://tenant.sharepoint.com/download"},
                )
            return httpx.Response(200, content=b"expense policy")

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            reader = SharePointGraphClient(
                "drive", token_provider=lambda: "private-token", http_client=client
            )
            contents = reader.download_file(
                SharePointFile("drive", "item", "policy.txt", 4), max_bytes=100
            )

        self.assertEqual(contents, b"expense policy")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer private-token")
        self.assertNotIn("Authorization", requests[1].headers)

    def test_rejects_oversized_content_and_malformed_redirects(self) -> None:
        for location in (
            "http://tenant.sharepoint.com/file",
            "https://[bad",
            "https://unrelated.example/file",
        ):
            with self.subTest(location=location):

                def redirect(
                    _request: httpx.Request, *, target: str = location
                ) -> httpx.Response:
                    return httpx.Response(302, headers={"location": target})

                with httpx.Client(transport=httpx.MockTransport(redirect)) as client:
                    reader = SharePointGraphClient(
                        "drive", token_provider=lambda: "token", http_client=client
                    )
                    with self.assertRaises(SharePointInputError):
                        reader.download_file(
                            SharePointFile("drive", "item", "a.txt", 1), max_bytes=2
                        )

        with httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, content=b"too large")
            )
        ) as client:
            reader = SharePointGraphClient(
                "drive", token_provider=lambda: "token", http_client=client
            )
            with self.assertRaisesRegex(SharePointInputError, "max_file_bytes"):
                reader.download_file(
                    SharePointFile("drive", "item", "a.txt", 1), max_bytes=2
                )

    def test_rejects_oversized_listing_page(self) -> None:
        with httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200, content=b"x" * (5 * 1024 * 1024 + 1)
                )
            )
        ) as client:
            reader = SharePointGraphClient(
                "drive", token_provider=lambda: "token", http_client=client
            )
            with self.assertRaisesRegex(SharePointInputError, "page exceeds"):
                reader.list_files(SharePointImportLimits())

    def test_provider_error_omits_graph_response_body(self) -> None:
        secret = "sensitive-response"
        with httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(403, content=secret)
            )
        ) as client:
            reader = SharePointGraphClient(
                "drive", token_provider=lambda: "token", http_client=client
            )
            with self.assertRaises(SharePointProviderError) as raised:
                reader.list_files(SharePointImportLimits())
        self.assertNotIn(secret, str(raised.exception))
        self.assertIn("403", str(raised.exception))


class MsalTokenProviderTests(unittest.TestCase):
    """Ensure credential failures remain actionable without leaking secrets."""

    def test_missing_environment_variables(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(
                SharePointConfigurationError, "RAG_SHAREPOINT_TENANT_ID"
            ),
        ):
            MsalTokenProvider.from_environment()

    def test_rejects_invalid_identifiers_before_client_creation(self) -> None:
        with self.assertRaisesRegex(SharePointConfigurationError, "tenant_id"):
            MsalTokenProvider("not-a-uuid", "also-invalid", "secret")

    def test_token_failure_does_not_disclose_provider_error(self) -> None:
        class FailingApplication:
            def acquire_token_for_client(self, *, scopes: list[str]) -> object:
                if scopes != ["https://graph.microsoft.com/.default"]:
                    raise AssertionError("Unexpected Graph scope")
                raise ValueError("sensitive-response")

        with patch(
            "msal.ConfidentialClientApplication", return_value=FailingApplication()
        ):
            provider = MsalTokenProvider(
                "00000000-0000-0000-0000-000000000001",
                "00000000-0000-0000-0000-000000000002",
                "private-secret",
            )
        with self.assertRaises(SharePointProviderError) as raised:
            provider()
        self.assertNotIn("sensitive-response", str(raised.exception))
        self.assertNotIn("private-secret", repr(provider))


if __name__ == "__main__":
    unittest.main()
