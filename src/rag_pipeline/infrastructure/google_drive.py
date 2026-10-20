"""Read a bounded Google Drive folder with application default credentials.

The adapter handles read-only Drive v3 transport and validates metadata before
the application stages files. Credential and response details never enter errors.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from rag_pipeline.exceptions import (
    GoogleDriveConfigurationError,
    GoogleDriveInputError,
    GoogleDriveProviderError,
)
from rag_pipeline.ingestion.extraction import SUPPORTED_FILE_EXTENSIONS

_DRIVE_ROOT = "https://www.googleapis.com/drive/v3/files"
_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
_FOLDER_MIME = "application/vnd.google-apps.folder"
_DOCUMENT_MIME = "application/vnd.google-apps.document"
_MIB = 1024 * 1024
_EXPORT_MAX_BYTES = 10_000_000
_MAX_LISTING_BYTES = 5 * _MIB
_MAX_LISTING_REQUESTS = 1000
_MAX_FOLDERS = 1000
_PAGE_SIZE = 200
_LIST_FIELDS = (
    "nextPageToken,incompleteSearch,"
    "files(id,name,mimeType,size,parents,capabilities(canDownload))"
)


@dataclass(frozen=True, slots=True)
class GoogleDriveImportLimits:
    """Cap supported file count and downloaded bytes for one manual import."""

    max_files: int = 100
    max_file_bytes: int = 20 * _MIB
    max_total_bytes: int = 100 * _MIB

    def __post_init__(self) -> None:
        for name in ("max_files", "max_file_bytes", "max_total_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise GoogleDriveConfigurationError(
                    f"{name} must be a positive integer."
                )


@dataclass(frozen=True, slots=True)
class GoogleDriveFile:
    """Describe one downloadable file independently of its mutable name."""

    root_folder_id: str
    file_id: str
    name: str
    mime_type: str
    size: int | None

    def __post_init__(self) -> None:
        _require_id(self.root_folder_id, "root_folder_id")
        _require_id(self.file_id, "file_id")
        try:
            name_bytes = (
                self.name.encode("utf-8") if isinstance(self.name, str) else b""
            )
        except UnicodeError as exc:
            raise GoogleDriveInputError("Drive returned an invalid file name.") from exc
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or len(name_bytes) > 1024
            or any(char in "/\\" or ord(char) < 32 for char in self.name)
        ):
            raise GoogleDriveInputError("Drive returned an invalid file name.")
        if not isinstance(self.mime_type, str) or not self.mime_type:
            raise GoogleDriveInputError("Drive returned an invalid MIME type.")
        if self.mime_type != _DOCUMENT_MIME and (
            self.mime_type.startswith("application/vnd.google-apps.")
            or Path(self.name).suffix.lower() not in SUPPORTED_FILE_EXTENSIONS
        ):
            raise GoogleDriveInputError("Drive returned an unsupported file type.")
        if self.mime_type == _DOCUMENT_MIME:
            if self.size is not None:
                raise GoogleDriveInputError("Drive returned a size for a Google Doc.")
        elif (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size < 0
        ):
            raise GoogleDriveInputError("Drive returned an invalid file size.")

    @property
    def source(self) -> str:
        """Keep citation identity stable across renames and folder moves."""
        return f"gdrive://{self.file_id}"

    @property
    def extraction_suffix(self) -> str:
        """Use plain text for native Docs and the original parser for blobs."""
        return (
            ".txt"
            if self.mime_type == _DOCUMENT_MIME
            else Path(self.name).suffix.lower()
        )


@dataclass(slots=True)
class GoogleAuthTokenProvider:
    """Refresh a read-only ADC token without exposing credentials in repr."""

    credentials: Any = field(repr=False)

    @classmethod
    def from_application_default(cls) -> GoogleAuthTokenProvider:
        """Find local or workload credentials when a live import starts."""
        import google.auth

        try:
            credentials, _project = google.auth.default(scopes=[_DRIVE_SCOPE])
        except Exception as exc:
            raise GoogleDriveConfigurationError(
                "Google Drive credentials are unavailable; configure ADC."
            ) from exc
        return cls(credentials)

    def __call__(self) -> str:
        """Refresh expired credentials and return an in-memory access token."""
        from google.auth.transport.requests import Request

        try:
            if not self.credentials.valid:
                self.credentials.refresh(Request())
            token = self.credentials.token
        except Exception as exc:
            raise GoogleDriveProviderError(
                "Could not acquire a Google Drive access token."
            ) from exc
        if not isinstance(token, str) or not token:
            raise GoogleDriveProviderError(
                "Google Drive credentials returned no token."
            )
        return token


class GoogleDriveClient:
    """Traverse one folder tree and download supported files through Drive v3.

    A caller-provided HTTP client and token callable keep transport tests offline.
    Only a client created here is closed by the context manager.
    """

    def __init__(
        self,
        root_folder_id: str,
        *,
        token_provider: Callable[[], str],
        http_client: httpx.Client | None = None,
    ) -> None:
        self.root_folder_id = _require_id(root_folder_id, "root_folder_id")
        self._token_provider = token_provider
        self._http_client = http_client or httpx.Client(
            timeout=30.0, follow_redirects=False
        )
        self._owns_client = http_client is None

    def __enter__(self) -> GoogleDriveClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        """Release only the HTTP pool this connector created."""
        if self._owns_client:
            self._http_client.close()

    def list_files(self, limits: GoogleDriveImportLimits) -> list[GoogleDriveFile]:
        """Validate a complete bounded folder traversal before any download.

        Duplicate IDs, incomplete listings, denied downloads, or limit breaches
        fail before the application can initialize models or mutate Qdrant.
        Drive pagination does not promise a point-in-time snapshot.
        """
        if not isinstance(limits, GoogleDriveImportLimits):
            raise TypeError("limits must be GoogleDriveImportLimits.")
        folders = deque([self.root_folder_id])
        seen_folders = {self.root_folder_id}
        seen_items: set[str] = set()
        files: list[GoogleDriveFile] = []
        declared_bytes = 0
        request_count = 0

        while folders:
            folder_id = folders.popleft()
            page_token: str | None = None
            seen_tokens: set[str] = set()
            while True:
                request_count += 1
                if request_count > _MAX_LISTING_REQUESTS:
                    raise GoogleDriveInputError(
                        "Drive listing exceeded its page limit."
                    )
                listing = self._list_page(folder_id, page_token)
                if (
                    listing.get("incompleteSearch") is not None
                    and listing.get("incompleteSearch") is not False
                ):
                    raise GoogleDriveInputError("Drive returned an incomplete listing.")
                for raw in listing["files"]:
                    if not isinstance(raw, dict):
                        raise GoogleDriveInputError("Drive returned an invalid file.")
                    item_id = _require_id(raw.get("id"), "file_id")
                    if item_id in seen_items or item_id == self.root_folder_id:
                        raise GoogleDriveInputError("Drive returned a duplicate item.")
                    seen_items.add(item_id)
                    parents = raw.get("parents")
                    if not isinstance(parents, list) or folder_id not in parents:
                        raise GoogleDriveInputError(
                            "Drive returned an item outside the requested folder."
                        )
                    mime_type = raw.get("mimeType")
                    if not isinstance(mime_type, str) or not mime_type:
                        raise GoogleDriveInputError(
                            "Drive returned an invalid MIME type."
                        )
                    if mime_type == _FOLDER_MIME:
                        seen_folders.add(item_id)
                        if len(seen_folders) > _MAX_FOLDERS:
                            raise GoogleDriveInputError(
                                "Drive folder traversal exceeded its limit."
                            )
                        folders.append(item_id)
                        continue
                    name = raw.get("name")
                    if not isinstance(name, str):
                        raise GoogleDriveInputError("Drive returned an invalid name.")
                    if (
                        mime_type.startswith("application/vnd.google-apps.")
                        and mime_type != _DOCUMENT_MIME
                    ):
                        continue
                    if (
                        mime_type != _DOCUMENT_MIME
                        and Path(name).suffix.lower() not in SUPPORTED_FILE_EXTENSIONS
                    ):
                        continue
                    capability = raw.get("capabilities")
                    if (
                        not isinstance(capability, dict)
                        or capability.get("canDownload") is not True
                    ):
                        raise GoogleDriveInputError(
                            "Drive denies downloading a supported file."
                        )
                    size = (
                        None
                        if mime_type == _DOCUMENT_MIME
                        else _parse_size(raw.get("size"))
                    )
                    file = GoogleDriveFile(
                        self.root_folder_id, item_id, name, mime_type, size
                    )
                    if size is not None:
                        if size > limits.max_file_bytes:
                            raise GoogleDriveInputError(
                                "A Drive file exceeds max_file_bytes."
                            )
                        declared_bytes += size
                    files.append(file)
                    if (
                        len(files) > limits.max_files
                        or declared_bytes > limits.max_total_bytes
                    ):
                        raise GoogleDriveInputError(
                            "Drive folder exceeds the configured import limits."
                        )
                next_token = listing.get("nextPageToken")
                if next_token is None:
                    break
                if (
                    not isinstance(next_token, str)
                    or not next_token
                    or len(next_token) > 4096
                    or any(ord(char) < 32 for char in next_token)
                    or next_token in seen_tokens
                ):
                    raise GoogleDriveInputError("Drive returned an invalid page token.")
                seen_tokens.add(next_token)
                page_token = next_token
        return sorted(files, key=lambda file: file.source)

    def download_file(self, file: GoogleDriveFile, *, max_bytes: int) -> bytes:
        """Stream a listed file or export a Doc under an actual byte cap.

        Requests go only to a fixed Google API origin and never follow redirects,
        so the bearer token cannot be forwarded to a provider-supplied URL.
        """
        if (
            not isinstance(file, GoogleDriveFile)
            or file.root_folder_id != self.root_folder_id
        ):
            raise GoogleDriveInputError("The file belongs to another Drive import.")
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise GoogleDriveConfigurationError("max_bytes must be positive.")
        if file.size is not None and file.size > max_bytes:
            raise GoogleDriveInputError("A Drive file exceeds max_file_bytes.")
        document = file.mime_type == _DOCUMENT_MIME
        url = f"{_DRIVE_ROOT}/{file.file_id}"
        if document:
            url += "/export"
        params = (
            {"mimeType": "text/plain"}
            if document
            else {"alt": "media", "supportsAllDrives": "true"}
        )
        try:
            with self._http_client.stream(
                "GET",
                url,
                params=params,
                headers={"Authorization": f"Bearer {self._access_token()}"},
                follow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    raise GoogleDriveProviderError(
                        f"Drive content request failed (HTTP {response.status_code})."
                    )
                return _read_bounded(
                    response,
                    min(max_bytes, _EXPORT_MAX_BYTES) if document else max_bytes,
                )
        except httpx.HTTPError as exc:
            raise GoogleDriveProviderError("Drive content request failed.") from exc

    def _list_page(self, folder_id: str, page_token: str | None) -> dict[str, Any]:
        query = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": _LIST_FIELDS,
            "pageSize": str(_PAGE_SIZE),
            "spaces": "drive",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }
        if page_token is not None:
            query["pageToken"] = page_token
        try:
            with self._http_client.stream(
                "GET",
                _DRIVE_ROOT,
                params=query,
                headers={
                    "Authorization": f"Bearer {self._access_token()}",
                    "Accept": "application/json",
                },
                follow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    raise GoogleDriveProviderError(
                        f"Drive listing failed (HTTP {response.status_code})."
                    )
                content = _read_bounded(response, _MAX_LISTING_BYTES)
        except httpx.HTTPError as exc:
            raise GoogleDriveProviderError("Drive listing request failed.") from exc
        try:
            listing = json.loads(content)
        except (ValueError, RecursionError) as exc:
            raise GoogleDriveInputError("Drive returned invalid listing JSON.") from exc
        if not isinstance(listing, dict) or not isinstance(listing.get("files"), list):
            raise GoogleDriveInputError("Drive returned an invalid listing page.")
        return listing

    def _access_token(self) -> str:
        try:
            token = self._token_provider()
        except GoogleDriveProviderError:
            raise
        except Exception as exc:
            raise GoogleDriveProviderError(
                "Could not acquire a Google Drive access token."
            ) from exc
        if not isinstance(token, str) or not token:
            raise GoogleDriveProviderError(
                "Google Drive token provider returned no token."
            )
        return token


def create_google_drive_client(folder_id: str) -> GoogleDriveClient:
    """Bind a live folder reader to scoped ADC for the local CLI."""
    validated_folder_id = _require_id(folder_id, "root_folder_id")
    return GoogleDriveClient(
        validated_folder_id,
        token_provider=GoogleAuthTokenProvider.from_application_default(),
    )


def _read_bounded(response: httpx.Response, max_bytes: int) -> bytes:
    content = bytearray()
    for block in response.iter_bytes():
        if len(content) + len(block) > max_bytes:
            raise GoogleDriveInputError("Drive response exceeds its byte limit.")
        content.extend(block)
    return bytes(content)


def _parse_size(value: object) -> int:
    if (
        isinstance(value, str)
        and 1 <= len(value) <= 19
        and value.isascii()
        and value.isdecimal()
        and int(value) <= 2**63 - 1
    ):
        return int(value)
    raise GoogleDriveInputError("Drive returned an invalid file size.")


def _require_id(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 256
        or not value.isascii()
        or any(not (char.isalnum() or char in "_-") for char in value)
    ):
        raise GoogleDriveInputError(f"Drive {name} must be a file or folder ID.")
    return value
