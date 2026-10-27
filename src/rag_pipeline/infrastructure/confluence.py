"""Read a bounded Confluence Cloud space through the REST API v2.

The adapter owns credential validation and read-only HTTP transport. It never
logs API tokens, page bodies, or continuation URLs; application code decides
how validated pages are converted into indexed documents.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urljoin, urlsplit

import httpx

from rag_pipeline.exceptions import (
    ConfluenceConfigurationError,
    ConfluenceInputError,
    ConfluenceProviderError,
)

_MIB = 1024 * 1024
_MAX_LISTING_BYTES = 64 * _MIB
_MAX_LISTING_REQUESTS = 1000
_PAGE_SIZE = 25


@dataclass(frozen=True, slots=True)
class ConfluenceImportLimits:
    """Cap one manual space import's page count and source HTML bytes."""

    max_pages: int = 100
    max_page_bytes: int = 2 * _MIB
    max_total_bytes: int = 20 * _MIB

    def __post_init__(self) -> None:
        for name in ("max_pages", "max_page_bytes", "max_total_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfluenceConfigurationError(
                    f"{name} must be a positive integer."
                )


@dataclass(frozen=True, slots=True)
class ConfluencePage:
    """Represent one current page with a rename-stable source identity."""

    site_host: str
    space_id: str
    page_id: str
    title: str
    body_html: str
    version: int
    web_url: str | None = None

    def __post_init__(self) -> None:
        if not _is_cloud_host(self.site_host):
            raise ConfluenceInputError("Confluence returned an invalid site host.")
        _require_id(self.space_id, "space_id")
        _require_id(self.page_id, "page_id")
        if not isinstance(self.title, str) or not self.title.strip():
            raise ConfluenceInputError("Confluence returned an invalid page title.")
        if not isinstance(self.body_html, str):
            raise ConfluenceInputError("Confluence returned an invalid page body.")
        try:
            title_bytes = len(self.title.encode("utf-8"))
            self.body_html.encode("utf-8")
        except UnicodeError as exc:
            raise ConfluenceInputError(
                "Confluence returned invalid page text encoding."
            ) from exc
        if title_bytes > 1024:
            raise ConfluenceInputError("Confluence returned an invalid page title.")
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version <= 0
        ):
            raise ConfluenceInputError("Confluence returned an invalid page version.")
        if self.web_url is not None and (
            len(self.web_url) > 4096
            or not self.web_url.startswith(f"https://{self.site_host}/wiki/")
            or any(ord(char) < 32 for char in self.web_url)
        ):
            raise ConfluenceInputError("Confluence returned an invalid page URL.")

    @property
    def source(self) -> str:
        """Keep Qdrant and citation IDs stable when a page title changes."""
        return f"confluence://{self.site_host}/{self.page_id}"


class ConfluenceCloudClient:
    """List current pages in one cloud space using operator-scoped credentials.

    Basic authentication with an API token is deliberately limited to a manual
    local CLI workflow. The caller owns the context manager; externally supplied
    HTTP clients remain under their creator's ownership for deterministic tests.
    """

    def __init__(
        self,
        base_url: str,
        *,
        email: str,
        api_token: str,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url, self.site_host = _validate_site_url(base_url)
        if (
            not isinstance(email, str)
            or email.count("@") != 1
            or not all(email.split("@"))
            or any(char.isspace() for char in email)
        ):
            raise ConfluenceConfigurationError(
                "RAG_CONFLUENCE_EMAIL must be an account email."
            )
        if (
            not isinstance(api_token, str)
            or not api_token.strip()
            or "\r" in api_token
            or "\n" in api_token
        ):
            raise ConfluenceConfigurationError("RAG_CONFLUENCE_API_TOKEN is required.")
        self._auth = httpx.BasicAuth(email, api_token)
        self._client = (
            http_client
            if http_client is not None
            else httpx.Client(timeout=30.0, follow_redirects=False)
        )
        self._owns_client = http_client is None

    @classmethod
    def from_environment(cls) -> ConfluenceCloudClient:
        """Read the site and credentials without accepting secrets on the CLI."""
        names = (
            "RAG_CONFLUENCE_BASE_URL",
            "RAG_CONFLUENCE_EMAIL",
            "RAG_CONFLUENCE_API_TOKEN",
        )
        values = [os.getenv(name) for name in names]
        missing = [name for name, value in zip(names, values, strict=True) if not value]
        if missing:
            raise ConfluenceConfigurationError(
                f"Missing Confluence environment variable(s): {', '.join(missing)}."
            )
        return cls(values[0] or "", email=values[1] or "", api_token=values[2] or "")

    def __enter__(self) -> ConfluenceCloudClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        """Release only the HTTP pool created by this adapter."""
        if self._owns_client:
            self._client.close()

    def list_pages(
        self, space_id: str, limits: ConfluenceImportLimits
    ) -> list[ConfluencePage]:
        """Finish one bounded traversal of current pages in a space.

        The v2 cursor link is followed only on the original site and endpoint.
        Duplicate IDs, missing storage bodies, non-current pages, and limit
        violations fail before application code can mutate Qdrant. Confluence
        does not guarantee a point-in-time view across cursor requests.
        """
        _require_id(space_id, "space_id")
        if not isinstance(limits, ConfluenceImportLimits):
            raise TypeError("limits must be ConfluenceImportLimits.")
        path = f"/wiki/api/v2/spaces/{space_id}/pages"
        query = urlencode(
            {
                "status": "current",
                "body-format": "storage",
                "limit": min(_PAGE_SIZE, limits.max_pages),
            }
        )
        url: str | None = f"{self.base_url}{path}?{query}"
        seen_urls: set[str] = set()
        seen_ids: set[str] = set()
        pages: list[ConfluencePage] = []
        total_bytes = 0

        while url is not None:
            if url in seen_urls or len(seen_urls) >= _MAX_LISTING_REQUESTS:
                raise ConfluenceInputError(
                    "Confluence pagination repeated or exceeded its limit."
                )
            seen_urls.add(url)
            listing = self._get_listing(url, expected_path=path)
            for raw_page in listing["results"]:
                page = self._parse_page(raw_page, space_id=space_id)
                if page.page_id in seen_ids:
                    raise ConfluenceInputError("Confluence returned a duplicate page.")
                seen_ids.add(page.page_id)
                page_bytes = len(page.body_html.encode("utf-8"))
                if page_bytes > limits.max_page_bytes:
                    raise ConfluenceInputError(
                        "A Confluence page exceeds max_page_bytes."
                    )
                total_bytes += page_bytes
                pages.append(page)
                if (
                    len(pages) > limits.max_pages
                    or total_bytes > limits.max_total_bytes
                ):
                    raise ConfluenceInputError(
                        "Confluence space exceeds the configured import limits."
                    )
            links = listing.get("_links", {})
            if not isinstance(links, dict):
                raise ConfluenceInputError("Confluence returned invalid page links.")
            next_link = links.get("next")
            if next_link is None:
                url = None
            elif (
                not isinstance(next_link, str)
                or not next_link
                or any(ord(char) < 32 for char in next_link)
            ):
                raise ConfluenceInputError(
                    "Confluence returned an invalid continuation link."
                )
            else:
                try:
                    url = urljoin(self.base_url, next_link)
                except ValueError as exc:
                    raise ConfluenceInputError(
                        "Confluence returned an unsafe continuation link."
                    ) from exc
                self._validate_listing_url(url, expected_path=path)
        return sorted(pages, key=lambda page: page.source)

    def _get_listing(self, url: str, *, expected_path: str) -> dict[str, Any]:
        self._validate_listing_url(url, expected_path=expected_path)
        try:
            with self._client.stream(
                "GET",
                url,
                auth=self._auth,
                headers={"Accept": "application/json"},
                follow_redirects=False,
            ) as response:
                if response.status_code != 200:
                    raise ConfluenceProviderError(
                        f"Confluence page listing failed (HTTP {response.status_code})."
                    )
                content = _read_bounded(response, _MAX_LISTING_BYTES)
        except httpx.HTTPError as exc:
            raise ConfluenceProviderError(
                "Confluence page listing request failed."
            ) from exc
        try:
            data = json.loads(content)
        except (ValueError, RecursionError) as exc:
            raise ConfluenceInputError(
                "Confluence returned invalid listing JSON."
            ) from exc
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise ConfluenceInputError("Confluence returned an invalid page listing.")
        return data

    def _validate_listing_url(self, url: str, *, expected_path: str) -> None:
        try:
            parts = urlsplit(url)
        except ValueError as exc:
            raise ConfluenceInputError(
                "Confluence returned an unsafe continuation link."
            ) from exc
        if (
            parts.scheme != "https"
            or parts.netloc != self.site_host
            or parts.path != expected_path
            or parts.fragment
            or len(url) > 8192
        ):
            raise ConfluenceInputError(
                "Confluence returned an unsafe continuation link."
            )

    def _parse_page(self, raw: object, *, space_id: str) -> ConfluencePage:
        if not isinstance(raw, dict):
            raise ConfluenceInputError("Confluence returned an invalid page.")
        page_id = _require_id(raw.get("id"), "page_id")
        if raw.get("spaceId") != space_id or raw.get("status") != "current":
            raise ConfluenceInputError(
                "Confluence returned a page outside the requested current space."
            )
        version = raw.get("version")
        body = raw.get("body")
        storage = body.get("storage") if isinstance(body, dict) else None
        if not isinstance(version, dict) or not isinstance(storage, dict):
            raise ConfluenceInputError(
                "Confluence returned a page without storage body or version."
            )
        title = raw.get("title")
        body_html = storage.get("value")
        version_number = version.get("number")
        if (
            not isinstance(title, str)
            or not isinstance(body_html, str)
            or isinstance(version_number, bool)
            or not isinstance(version_number, int)
        ):
            raise ConfluenceInputError("Confluence returned invalid page content.")
        links = raw.get("_links", {})
        if not isinstance(links, dict):
            raise ConfluenceInputError("Confluence returned invalid page links.")
        web_url = self._page_web_url(links.get("webui"))
        return ConfluencePage(
            site_host=self.site_host,
            space_id=space_id,
            page_id=page_id,
            title=title,
            body_html=body_html,
            version=version_number,
            web_url=web_url,
        )

    def _page_web_url(self, webui: object) -> str | None:
        if webui is None:
            return None
        if (
            not isinstance(webui, str)
            or not webui.startswith("/")
            or webui.startswith("//")
            or len(webui) > 4096
            or "\\" in webui
            or "#" in webui
            or any(ord(char) < 32 for char in webui)
            or any(segment == ".." for segment in webui.split("/"))
        ):
            raise ConfluenceInputError("Confluence returned an unsafe page URL.")
        path = webui if webui.startswith("/wiki/") else f"/wiki{webui}"
        return f"{self.base_url}{path}"


def _read_bounded(response: httpx.Response, max_bytes: int) -> bytes:
    content = bytearray()
    for block in response.iter_bytes():
        if len(content) + len(block) > max_bytes:
            raise ConfluenceInputError("Confluence listing exceeds the response limit.")
        content.extend(block)
    return bytes(content)


def _validate_site_url(value: object) -> tuple[str, str]:
    if not isinstance(value, str):
        raise ConfluenceConfigurationError(
            "RAG_CONFLUENCE_BASE_URL must be an HTTPS cloud site root."
        )
    try:
        parts = urlsplit(value)
        host = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise ConfluenceConfigurationError(
            "RAG_CONFLUENCE_BASE_URL must be an HTTPS cloud site root."
        ) from exc
    if (
        parts.scheme != "https"
        or not host
        or not _is_cloud_host(host)
        or port not in (None, 443)
        or parts.username is not None
        or parts.password is not None
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
    ):
        raise ConfluenceConfigurationError(
            "RAG_CONFLUENCE_BASE_URL must be an HTTPS *.atlassian.net site root."
        )
    return f"https://{host}", host


def _is_cloud_host(value: object) -> bool:
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.endswith(".atlassian.net")
        or len(value) > 253
    ):
        return False
    labels = value.split(".")
    return all(
        label
        and len(label) <= 63
        and label[0].isalnum()
        and label[-1].isalnum()
        and all(char.isalnum() or char == "-" for char in label)
        for label in labels
    )


def _require_id(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.isascii()
        or not value.isdecimal()
        or len(value) > 32
    ):
        raise ConfluenceInputError(f"{name} must be a numeric Confluence ID.")
    return value
