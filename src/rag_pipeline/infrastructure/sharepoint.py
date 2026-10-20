"""Read bounded SharePoint drive snapshots through Microsoft Graph.

The adapter owns app-only token acquisition and HTTP transport. It never writes
to SharePoint or exposes bearer tokens and preauthenticated download URLs in
errors; application code decides how retrieved files are indexed.
"""

from __future__ import annotations

import ipaddress
import json
import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlsplit
from uuid import UUID

import httpx

from rag_pipeline.exceptions import (
    SharePointConfigurationError,
    SharePointInputError,
    SharePointProviderError,
)
from rag_pipeline.ingestion.extraction import SUPPORTED_FILE_EXTENSIONS

_GRAPH_ROOT = "https://graph.microsoft.com/v1.0"
_GRAPH_SCOPE = "https://graph.microsoft.com/.default"
_PAGE_LIMIT = 1000
_PAGE_BYTES = 5 * 1024 * 1024
_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class SharePointImportLimits:
    """Bound one manual import's file count and memory-facing download sizes."""

    max_files: int = 100
    max_file_bytes: int = 20 * _MIB
    max_total_bytes: int = 100 * _MIB

    def __post_init__(self) -> None:
        for name in ("max_files", "max_file_bytes", "max_total_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise SharePointConfigurationError(
                    f"{name} must be a positive integer."
                )


@dataclass(frozen=True, slots=True)
class SharePointFile:
    """Keep Graph item identity separate from mutable display name and URL."""

    drive_id: str
    item_id: str
    name: str
    size: int
    web_url: str | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.drive_id, "drive_id")
        _validate_identifier(self.item_id, "item_id")
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or "/" in self.name
            or "\\" in self.name
            or "\x00" in self.name
        ):
            raise SharePointInputError("Graph returned an invalid file name.")
        if (
            isinstance(self.size, bool)
            or not isinstance(self.size, int)
            or self.size < 0
        ):
            raise SharePointInputError("Graph returned an invalid file size.")
        if self.web_url is not None:
            _validate_https_url(self.web_url, "file webUrl")

    @property
    def source(self) -> str:
        """Return a rename-stable source key for Qdrant and citations."""
        return f"sharepoint://{quote(self.drive_id, safe='')}/{quote(self.item_id, safe='')}"


@dataclass(slots=True)
class MsalTokenProvider:
    """Acquire cacheable app-only Graph tokens from environment-held credentials.

    The tenant and client IDs identify one Entra application; the secret stays
    in process memory and is never included in errors or object reprs.
    """

    tenant_id: str
    client_id: str
    client_secret: str = field(repr=False)
    _application: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("tenant_id", self.tenant_id),
            ("client_id", self.client_id),
        ):
            try:
                UUID(value)
            except (TypeError, ValueError) as exc:
                raise SharePointConfigurationError(f"{name} must be a UUID.") from exc
        if not isinstance(self.client_secret, str) or not self.client_secret.strip():
            raise SharePointConfigurationError(
                "RAG_SHAREPOINT_CLIENT_SECRET is required."
            )

        import msal

        try:
            self._application = msal.ConfidentialClientApplication(
                self.client_id,
                authority=f"https://login.microsoftonline.com/{self.tenant_id}",
                client_credential=self.client_secret,
            )
        except Exception as exc:
            raise SharePointProviderError(
                "Could not initialize Graph app authentication."
            ) from exc

    @classmethod
    def from_environment(cls) -> MsalTokenProvider:
        """Read required credentials without accepting secrets on the CLI."""
        names = (
            "RAG_SHAREPOINT_TENANT_ID",
            "RAG_SHAREPOINT_CLIENT_ID",
            "RAG_SHAREPOINT_CLIENT_SECRET",
        )
        values = [os.getenv(name) for name in names]
        missing = [name for name, value in zip(names, values, strict=True) if not value]
        if missing:
            raise SharePointConfigurationError(
                f"Missing SharePoint environment variable(s): {', '.join(missing)}."
            )
        return cls(values[0] or "", values[1] or "", values[2] or "")

    def __call__(self) -> str:
        """Get a cached token or request a new one; never expose MSAL details."""
        try:
            result = self._application.acquire_token_for_client(scopes=[_GRAPH_SCOPE])
        except Exception as exc:
            raise SharePointProviderError(
                "Could not acquire a Graph access token."
            ) from exc
        token = result.get("access_token") if isinstance(result, dict) else None
        if not isinstance(token, str) or not token:
            raise SharePointProviderError(
                "Graph app authentication failed; check the Entra app and consent."
            )
        return token


class SharePointGraphClient:
    """List and download supported files from one Graph drive, read-only.

    A supplied HTTP client and token callable make every network path testable
    offline. The caller owns this context manager; it closes only HTTP clients
    created internally, leaving injected test or shared clients to their owner.
    """

    def __init__(
        self,
        drive_id: str,
        *,
        token_provider: Callable[[], str],
        http_client: httpx.Client | None = None,
    ) -> None:
        _validate_identifier(drive_id, "drive_id")
        self.drive_id = drive_id
        self._token_provider = token_provider
        self._http_client = http_client or httpx.Client(
            timeout=30.0, follow_redirects=False
        )
        self._owns_client = http_client is None

    def __enter__(self) -> SharePointGraphClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        """Release the connector-owned HTTP connection pool after a run."""
        if self._owns_client:
            self._http_client.close()

    def list_files(self, limits: SharePointImportLimits) -> list[SharePointFile]:
        """Walk all drive folders and validate one bounded, paginated file list.

        This reads metadata only. Duplicate IDs, malformed pages, oversized
        files, excess totals, or unsafe continuation URLs fail before any
        file download or database mutation can begin.
        """
        if not isinstance(limits, SharePointImportLimits):
            raise TypeError("limits must be SharePointImportLimits.")
        folders: deque[str | None] = deque([None])
        seen_folders: set[str] = set()
        seen_items: set[str] = set()
        seen_pages: set[str] = set()
        files: list[SharePointFile] = []
        declared_bytes = 0

        while folders:
            folder_id = folders.popleft()
            initial_url = self._children_url(folder_id)
            url: str | None = initial_url
            expected_path = urlsplit(initial_url).path
            while url:
                if url in seen_pages or len(seen_pages) >= _PAGE_LIMIT:
                    raise SharePointInputError(
                        "Graph pagination repeated or exceeded its limit."
                    )
                seen_pages.add(url)
                page = self._get_page(url, expected_path=expected_path)
                for raw_item in page["value"]:
                    if not isinstance(raw_item, dict):
                        raise SharePointInputError(
                            "Graph returned an invalid drive item."
                        )
                    item_id = _validate_identifier(raw_item.get("id"), "item_id")
                    if item_id in seen_items:
                        raise SharePointInputError(
                            "Graph returned a duplicate drive item."
                        )
                    seen_items.add(item_id)
                    if isinstance(raw_item.get("folder"), dict):
                        if item_id in seen_folders:
                            raise SharePointInputError(
                                "Graph returned a repeated folder."
                            )
                        seen_folders.add(item_id)
                        folders.append(item_id)
                        continue
                    if not isinstance(raw_item.get("file"), dict):
                        continue
                    name = raw_item.get("name")
                    if not isinstance(name, str):
                        raise SharePointInputError(
                            "Graph returned an invalid file name."
                        )
                    if Path(name).suffix.lower() not in SUPPORTED_FILE_EXTENSIONS:
                        continue
                    size = raw_item.get("size")
                    if isinstance(size, bool) or not isinstance(size, int):
                        raise SharePointInputError(
                            "Graph returned an invalid file size."
                        )
                    file = SharePointFile(
                        drive_id=self.drive_id,
                        item_id=item_id,
                        name=name,
                        size=size,
                        web_url=raw_item.get("webUrl"),
                    )
                    if file.size > limits.max_file_bytes:
                        raise SharePointInputError(
                            "A SharePoint file exceeds max_file_bytes."
                        )
                    files.append(file)
                    declared_bytes += file.size
                    if (
                        len(files) > limits.max_files
                        or declared_bytes > limits.max_total_bytes
                    ):
                        raise SharePointInputError(
                            "SharePoint drive exceeds the configured import limits."
                        )
                url = page.get("@odata.nextLink")
                if url is not None and not isinstance(url, str):
                    raise SharePointInputError("Graph returned an invalid nextLink.")
        return sorted(files, key=lambda file: file.source)

    def download_file(self, file: SharePointFile, *, max_bytes: int) -> bytes:
        """Follow Graph's preauthenticated redirect without forwarding Bearer.

        A streaming byte cap protects memory even if Graph reports an incorrect
        item size. Redirect URLs never appear in errors or logs.
        """
        if file.drive_id != self.drive_id:
            raise SharePointInputError("The file belongs to another drive.")
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise SharePointConfigurationError("max_bytes must be positive.")
        path = f"/v1.0/drives/{quote(self.drive_id, safe='')}/items/{quote(file.item_id, safe='')}/content"
        url = f"https://graph.microsoft.com{path}"
        token = self._access_token()
        try:
            with self._http_client.stream(
                "GET",
                url,
                headers={"Authorization": f"Bearer {token}"},
                follow_redirects=False,
            ) as response:
                if response.status_code == 200:
                    return _bounded_content(response, max_bytes)
                if response.status_code != 302:
                    raise SharePointProviderError(
                        f"Graph download request failed (HTTP {response.status_code})."
                    )
                location = response.headers.get("location")
            download_host = _validate_https_url(location, "download redirect")
            if not download_host.endswith(".sharepoint.com"):
                raise SharePointInputError(
                    "Graph returned an unsafe download redirect."
                )
            with self._http_client.stream(
                "GET", location, follow_redirects=False
            ) as response:
                if response.status_code != 200:
                    raise SharePointProviderError(
                        f"SharePoint content download failed (HTTP {response.status_code})."
                    )
                return _bounded_content(response, max_bytes)
        except httpx.HTTPError as exc:
            raise SharePointProviderError("SharePoint content request failed.") from exc

    def _children_url(self, folder_id: str | None) -> str:
        base = f"{_GRAPH_ROOT}/drives/{quote(self.drive_id, safe='')}"
        path = (
            "/root/children"
            if folder_id is None
            else f"/items/{quote(folder_id, safe='')}/children"
        )
        query = urlencode({"$select": "id,name,size,file,folder,webUrl", "$top": 200})
        return f"{base}{path}?{query}"

    def _get_page(self, url: str, *, expected_path: str) -> dict[str, Any]:
        try:
            parts = urlsplit(url)
        except ValueError as exc:
            raise SharePointInputError(
                "Graph returned an unsafe continuation URL."
            ) from exc
        if (
            parts.scheme != "https"
            or parts.netloc != "graph.microsoft.com"
            or parts.path != expected_path
            or parts.fragment
        ):
            raise SharePointInputError("Graph returned an unsafe continuation URL.")
        try:
            with self._http_client.stream(
                "GET",
                url,
                headers={
                    "Authorization": f"Bearer {self._access_token()}",
                    "Accept": "application/json",
                },
                follow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    raise SharePointProviderError(
                        f"Graph drive listing failed (HTTP {response.status_code})."
                    )
                content = _bounded_content(
                    response,
                    _PAGE_BYTES,
                    error_message="Graph listing page exceeds the size limit.",
                )
        except httpx.HTTPError as exc:
            raise SharePointProviderError(
                "Graph drive listing request failed."
            ) from exc
        try:
            data = json.loads(content)
        except ValueError as exc:
            raise SharePointInputError("Graph returned invalid listing JSON.") from exc
        if not isinstance(data, dict) or not isinstance(data.get("value"), list):
            raise SharePointInputError("Graph returned an invalid listing page.")
        return data

    def _access_token(self) -> str:
        try:
            token = self._token_provider()
        except SharePointProviderError:
            raise
        except Exception as exc:
            raise SharePointProviderError(
                "Could not acquire a Graph access token."
            ) from exc
        if not isinstance(token, str) or not token:
            raise SharePointProviderError("Graph token provider returned no token.")
        return token


def create_sharepoint_client(drive_id: str) -> SharePointGraphClient:
    """Create a live Graph reader from environment credentials for the CLI."""
    return SharePointGraphClient(
        drive_id, token_provider=MsalTokenProvider.from_environment()
    )


def _bounded_content(
    response: httpx.Response,
    max_bytes: int,
    *,
    error_message: str = "SharePoint file exceeds max_file_bytes.",
) -> bytes:
    content = bytearray()
    for block in response.iter_bytes():
        if len(content) + len(block) > max_bytes:
            raise SharePointInputError(error_message)
        content.extend(block)
    return bytes(content)


def _validate_identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or any(char.isspace() or ord(char) < 32 for char in value)
        or "/" in value
        or "\\" in value
    ):
        raise SharePointInputError(f"Graph {name} must be a non-empty identifier.")
    return value


def _validate_https_url(value: object, description: str) -> str:
    if not isinstance(value, str):
        raise SharePointInputError(f"Graph returned an invalid {description}.")
    try:
        parts = urlsplit(value)
        host = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise SharePointInputError(f"Graph returned an unsafe {description}.") from exc
    if (
        parts.scheme != "https"
        or not host
        or "." not in host
        or port not in (None, 443)
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
    ):
        raise SharePointInputError(f"Graph returned an unsafe {description}.")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return host
    raise SharePointInputError(f"Graph returned an unsafe {description}.")
