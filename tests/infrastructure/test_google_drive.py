"""Exercise Google Drive listing, downloads, and ADC without network access."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import httpx

from rag_pipeline.exceptions import (
    GoogleDriveConfigurationError,
    GoogleDriveInputError,
    GoogleDriveProviderError,
)
from rag_pipeline.infrastructure.google_drive import (
    GoogleAuthTokenProvider,
    GoogleDriveClient,
    GoogleDriveFile,
    GoogleDriveImportLimits,
    create_google_drive_client,
)

_ROOT = "folder_123"


def _blob(file_id: str, *, parent: str = _ROOT, size: str = "5") -> dict[str, object]:
    return {
        "id": file_id,
        "name": "policy.txt",
        "mimeType": "text/plain",
        "size": size,
        "parents": [parent],
        "capabilities": {"canDownload": True},
    }


def _doc(file_id: str, *, parent: str = _ROOT) -> dict[str, object]:
    return {
        "id": file_id,
        "name": "Google policy",
        "mimeType": "application/vnd.google-apps.document",
        "parents": [parent],
        "capabilities": {"canDownload": True},
    }


class GoogleDriveClientTests(unittest.TestCase):
    """Keep traversal complete and bounded before downloading content."""

    def _client(self, client: httpx.Client) -> GoogleDriveClient:
        return GoogleDriveClient(
            _ROOT, token_provider=lambda: "private-token", http_client=client
        )

    def test_nested_pagination_and_download_routes(self) -> None:
        requests: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            self.assertEqual(request.url.host, "www.googleapis.com")
            self.assertEqual(request.headers["Authorization"], "Bearer private-token")
            if request.url.path == "/drive/v3/files":
                self.assertEqual(request.url.params["supportsAllDrives"], "true")
                self.assertEqual(
                    request.url.params["includeItemsFromAllDrives"], "true"
                )
                if request.url.params.get("pageToken") == "next":
                    return httpx.Response(200, json={"files": [_doc("doc_1")]})
                if "child_1" in request.url.params["q"]:
                    return httpx.Response(
                        200, json={"files": [_blob("blob_2", parent="child_1")]}
                    )
                return httpx.Response(
                    200,
                    json={
                        "files": [
                            _blob("blob_1"),
                            {
                                "id": "child_1",
                                "mimeType": "application/vnd.google-apps.folder",
                                "parents": [_ROOT],
                            },
                            {
                                "id": "sheet_1",
                                "name": "Budget.txt",
                                "mimeType": "application/vnd.google-apps.spreadsheet",
                                "parents": [_ROOT],
                            },
                        ],
                        "nextPageToken": "next",
                    },
                )
            if request.url.path.endswith("/export"):
                self.assertEqual(request.url.params["mimeType"], "text/plain")
                return httpx.Response(200, content=b"Google policy text")
            self.assertEqual(request.url.params["alt"], "media")
            return httpx.Response(200, content=b"hello")

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            reader = self._client(client)
            files = reader.list_files(GoogleDriveImportLimits())
            self.assertEqual(
                [file.file_id for file in files], ["blob_1", "blob_2", "doc_1"]
            )
            self.assertEqual(files[2].extraction_suffix, ".txt")
            self.assertEqual(reader.download_file(files[0], max_bytes=10), b"hello")
            self.assertEqual(
                reader.download_file(files[2], max_bytes=100), b"Google policy text"
            )
        self.assertEqual(len(requests), 5)
        self.assertEqual(files[0].source, "gdrive://blob_1")
        self.assertEqual(
            GoogleDriveFile(_ROOT, "blob_1", "renamed.txt", "text/plain", 5).source,
            files[0].source,
        )

    def test_rejects_incomplete_duplicate_and_foreign_items(self) -> None:
        cases = [
            {"files": [], "incompleteSearch": True},
            {"files": [_blob("a"), _blob("a")]},
            {"files": [_blob("a", parent="different")]},
            {"files": [{**_blob("a"), "capabilities": {"canDownload": False}}]},
            {"files": [{**_blob("a"), "size": "not-a-size"}]},
            {"files": [{**_blob("a"), "size": "9" * 5000}]},
            {"files": [{**_doc("a"), "name": "bad/filename"}]},
            {"files": [{**_doc("a"), "name": "bad\ud800"}]},
        ]
        for listing in cases:
            with self.subTest(listing=listing):
                with httpx.Client(
                    transport=httpx.MockTransport(
                        lambda _request, data=listing: httpx.Response(
                            200, content=json.dumps(data, ensure_ascii=True).encode()
                        )
                    )
                ) as client:
                    with self.assertRaises(GoogleDriveInputError):
                        self._client(client).list_files(GoogleDriveImportLimits())

    def test_repeated_page_token_and_limits(self) -> None:
        cases = [
            ({"files": [], "nextPageToken": "again"}, GoogleDriveImportLimits()),
            ({"files": [_blob("a"), _blob("b")]}, GoogleDriveImportLimits(max_files=1)),
            (
                {"files": [_blob("a", size="6")]},
                GoogleDriveImportLimits(max_file_bytes=5),
            ),
            (
                {"files": [_blob("a"), _blob("b")]},
                GoogleDriveImportLimits(max_total_bytes=9),
            ),
        ]
        for listing, limits in cases:
            with self.subTest(listing=listing, limits=limits):
                with httpx.Client(
                    transport=httpx.MockTransport(
                        lambda _request, data=listing: httpx.Response(200, json=data)
                    )
                ) as client:
                    with self.assertRaises(GoogleDriveInputError):
                        self._client(client).list_files(limits)

    def test_malformed_and_oversized_listing(self) -> None:
        for body in (b"not json", json.dumps({"files": {}}).encode(), b"x" * 1025):
            with self.subTest(body=body[:20]):
                with (
                    patch(
                        "rag_pipeline.infrastructure.google_drive._MAX_LISTING_BYTES",
                        1024,
                    ),
                    httpx.Client(
                        transport=httpx.MockTransport(
                            lambda _request, content=body: httpx.Response(
                                200, content=content
                            )
                        )
                    ) as client,
                ):
                    with self.assertRaises(GoogleDriveInputError):
                        self._client(client).list_files(GoogleDriveImportLimits())

    def test_redirect_and_provider_errors_never_expose_response_body(self) -> None:
        calls: list[httpx.Request] = []

        def respond(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(
                302,
                content=b"private-response",
                headers={"Location": "https://evil.example/steal"},
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            reader = self._client(client)
            with self.assertRaises(GoogleDriveProviderError) as raised:
                reader.list_files(GoogleDriveImportLimits())
            file = GoogleDriveFile(_ROOT, "blob_1", "policy.txt", "text/plain", 5)
            with self.assertRaises(GoogleDriveProviderError):
                reader.download_file(file, max_bytes=10)
        self.assertEqual(len(calls), 2)
        self.assertNotIn("private-response", str(raised.exception))
        self.assertNotIn("evil.example", str(raised.exception))

    def test_actual_download_limit_and_cross_import_file(self) -> None:
        with httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, content=b"too many bytes")
            )
        ) as client:
            reader = self._client(client)
            file = GoogleDriveFile(_ROOT, "blob_1", "policy.txt", "text/plain", 3)
            with self.assertRaises(GoogleDriveInputError):
                reader.download_file(file, max_bytes=5)
            foreign = GoogleDriveFile("other", "blob_1", "policy.txt", "text/plain", 3)
            with self.assertRaises(GoogleDriveInputError):
                reader.download_file(foreign, max_bytes=5)

    def test_google_doc_export_honors_provider_cap(self) -> None:
        document = GoogleDriveFile(
            _ROOT, "doc_1", "Policy", "application/vnd.google-apps.document", None
        )
        with (
            patch("rag_pipeline.infrastructure.google_drive._EXPORT_MAX_BYTES", 5),
            httpx.Client(
                transport=httpx.MockTransport(
                    lambda _request: httpx.Response(200, content=b"123456")
                )
            ) as client,
        ):
            with self.assertRaises(GoogleDriveInputError):
                self._client(client).download_file(document, max_bytes=20)

    def test_validates_ids_limits_and_credentials(self) -> None:
        for bad_id in ("", "' OR trashed=false", "../other", "non ascii ä"):
            with self.subTest(bad_id=bad_id), self.assertRaises(GoogleDriveInputError):
                GoogleDriveClient(bad_id, token_provider=lambda: "token")
        with self.assertRaises(GoogleDriveConfigurationError):
            GoogleDriveImportLimits(max_files=0)
        with patch("google.auth.default", side_effect=RuntimeError("private")):
            with self.assertRaises(GoogleDriveConfigurationError) as raised:
                GoogleAuthTokenProvider.from_application_default()
        self.assertNotIn("private", str(raised.exception))

        class Credentials:
            valid = False
            token = "secret-token"

            def refresh(self, _request: object) -> None:
                self.valid = True

        provider = GoogleAuthTokenProvider(Credentials())
        self.assertEqual(provider(), "secret-token")
        self.assertNotIn("secret-token", repr(provider))
        with patch("google.auth.default", return_value=(Credentials(), None)) as load:
            GoogleAuthTokenProvider.from_application_default()
        load.assert_called_once_with(
            scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )

    def test_invalid_folder_id_precedes_adc_lookup(self) -> None:
        with patch(
            "rag_pipeline.infrastructure.google_drive.GoogleAuthTokenProvider.from_application_default"
        ) as credentials:
            with self.assertRaises(GoogleDriveInputError):
                create_google_drive_client("bad/id")
        credentials.assert_not_called()
