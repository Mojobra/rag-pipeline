"""Exercise Confluence Cloud transport and validation with offline HTTP doubles."""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

import httpx

from rag_pipeline.exceptions import (
    ConfluenceConfigurationError,
    ConfluenceInputError,
    ConfluenceProviderError,
)
from rag_pipeline.infrastructure.confluence import (
    ConfluenceCloudClient,
    ConfluenceImportLimits,
    ConfluencePage,
)

_SITE = "https://team.atlassian.net"
_SPACE_PATH = "/wiki/api/v2/spaces/42/pages"


def _page(
    page_id: str,
    *,
    title: str = "Expense policy",
    body: str = "<p>Claims within 30 days.</p>",
) -> dict[str, object]:
    return {
        "id": page_id,
        "spaceId": "42",
        "status": "current",
        "title": title,
        "version": {"number": 3},
        "body": {"storage": {"value": body}},
        "_links": {"webui": f"/spaces/OPS/pages/{page_id}/Expense+policy"},
    }


class ConfluenceCloudClientTests(unittest.TestCase):
    """Keep page traversal bounded and credentials on the intended origin."""

    def _reader(self, client: httpx.Client) -> ConfluenceCloudClient:
        return ConfluenceCloudClient(
            _SITE,
            email="reader@example.com",
            api_token="private-token",
            http_client=client,
        )

    def test_cursor_pagination_and_stable_page_identity(self) -> None:
        requests: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            self.assertEqual(request.url.path, _SPACE_PATH)
            self.assertTrue(request.headers["Authorization"].startswith("Basic "))
            if request.url.params.get("cursor") == "next":
                return httpx.Response(200, json={"results": [_page("200")]})
            self.assertEqual(request.url.params["body-format"], "storage")
            self.assertEqual(request.url.params["status"], "current")
            return httpx.Response(
                200,
                json={
                    "results": [_page("100")],
                    "_links": {"next": f"{_SPACE_PATH}?cursor=next"},
                },
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            pages = self._reader(client).list_pages("42", ConfluenceImportLimits())

        self.assertEqual(len(requests), 2)
        self.assertEqual([page.page_id for page in pages], ["100", "200"])
        self.assertEqual(pages[0].source, "confluence://team.atlassian.net/100")
        self.assertEqual(
            pages[0].web_url,
            "https://team.atlassian.net/wiki/spaces/OPS/pages/100/Expense+policy",
        )
        renamed = ConfluencePage(
            "team.atlassian.net", "42", "100", "New title", "<p>New</p>", 4
        )
        self.assertEqual(pages[0].source, renamed.source)

    def test_unsafe_repeated_and_malformed_cursor_links(self) -> None:
        for next_link in (
            "https://evil.example/wiki/api/v2/spaces/42/pages?cursor=x",
            "/wiki/api/v2/spaces/99/pages?cursor=x",
            f"{_SPACE_PATH}?status=current&body-format=storage&limit=25",
            "https://[bad",
        ):
            with self.subTest(next_link=next_link):

                def respond(
                    _request: httpx.Request, *, link: str = next_link
                ) -> httpx.Response:
                    return httpx.Response(
                        200, json={"results": [], "_links": {"next": link}}
                    )

                with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                    with self.assertRaises(ConfluenceInputError):
                        self._reader(client).list_pages("42", ConfluenceImportLimits())

    def test_page_count_and_html_byte_caps(self) -> None:
        def respond(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"results": [_page("100", body="é" * 4), _page("200")]},
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            reader = self._reader(client)
            for limits in (
                ConfluenceImportLimits(max_pages=1),
                ConfluenceImportLimits(max_page_bytes=7),
                ConfluenceImportLimits(max_total_bytes=32),
            ):
                with (
                    self.subTest(limits=limits),
                    self.assertRaises(ConfluenceInputError),
                ):
                    reader.list_pages("42", limits)

    def test_rejects_missing_body_wrong_space_and_duplicate_ids(self) -> None:
        invalid_cases = [
            [{**_page("100"), "body": {}}],
            [{**_page("100"), "spaceId": "99"}],
            [_page("100"), _page("100")],
            [{**_page("100"), "status": "draft"}],
            [{**_page("100"), "title": None}],
            [{**_page("100"), "version": {"number": "3"}}],
            [{**_page("100"), "body": {"storage": {"value": None}}}],
            [{**_page("100"), "title": "x" * 1025}],
            [{**_page("100"), "title": "bad\ud800"}],
            [{**_page("100"), "body": {"storage": {"value": "bad\ud800"}}}],
            [{**_page("100"), "_links": {"webui": "//evil.example/page"}}],
        ]
        for results in invalid_cases:
            with self.subTest(results=results):

                def respond(
                    _request: httpx.Request, *, items: list[dict[str, object]] = results
                ) -> httpx.Response:
                    return httpx.Response(
                        200, content=json.dumps({"results": items}).encode("utf-8")
                    )

                with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                    with self.assertRaises(ConfluenceInputError):
                        self._reader(client).list_pages("42", ConfluenceImportLimits())

    def test_provider_error_omits_response_body_and_does_not_follow_redirect(
        self,
    ) -> None:
        for status in (302, 401, 429):
            with self.subTest(status=status):
                requests: list[httpx.Request] = []

                def respond(
                    request: httpx.Request,
                    *,
                    code: int = status,
                    captured: list[httpx.Request] = requests,
                ) -> httpx.Response:
                    captured.append(request)
                    return httpx.Response(
                        code,
                        content="secret-response",
                        headers={"location": "https://evil.example/token"},
                    )

                with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                    with self.assertRaises(ConfluenceProviderError) as raised:
                        self._reader(client).list_pages("42", ConfluenceImportLimits())
                self.assertEqual(len(requests), 1)
                self.assertNotIn("secret-response", str(raised.exception))
                self.assertIn(str(status), str(raised.exception))

    def test_response_size_cap_and_invalid_json(self) -> None:
        for body in (b"x" * 1025, b"not JSON"):
            with self.subTest(body_size=len(body)):

                def respond(
                    _request: httpx.Request, *, content: bytes = body
                ) -> httpx.Response:
                    return httpx.Response(200, content=content)

                with (
                    patch(
                        "rag_pipeline.infrastructure.confluence._MAX_LISTING_BYTES",
                        1024,
                    ),
                    httpx.Client(transport=httpx.MockTransport(respond)) as client,
                ):
                    with self.assertRaises(ConfluenceInputError):
                        self._reader(client).list_pages("42", ConfluenceImportLimits())


class ConfluenceConfigurationTests(unittest.TestCase):
    """Reject unsafe cloud roots and missing credentials before network I/O."""

    def test_requires_environment_credentials(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(
                ConfluenceConfigurationError, "RAG_CONFLUENCE_API_TOKEN"
            ),
        ):
            ConfluenceCloudClient.from_environment()

    def test_rejects_non_cloud_or_credential_bearing_site_url(self) -> None:
        for url in (
            "http://team.atlassian.net",
            "https://localhost",
            "https://user:pass@team.atlassian.net",
            "https://team.atlassian.net/wiki",
            "https://[bad",
            "https://bad..atlassian.net",
        ):
            with self.subTest(url=url), self.assertRaises(ConfluenceConfigurationError):
                ConfluenceCloudClient(
                    url, email="reader@example.com", api_token="secret"
                )

    def test_limit_validation(self) -> None:
        with self.assertRaises(ConfluenceConfigurationError):
            ConfluenceImportLimits(max_pages=0)

    def test_rejects_blank_token_and_invalid_email(self) -> None:
        for email, token in (("reader@example.com", "   "), ("invalid", "token")):
            with (
                self.subTest(email=email, token=token),
                self.assertRaises(ConfluenceConfigurationError),
            ):
                ConfluenceCloudClient(_SITE, email=email, api_token=token)


if __name__ == "__main__":
    unittest.main()
