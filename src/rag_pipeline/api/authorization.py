"""Load a private source-access policy for opt-in HTTP query authorization.

This adapter binds high-entropy query credentials to local principals and
grants access to exact indexed source identifiers; it does not sync remote ACLs.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictStr, ValidationError

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{32,256}\Z")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_MAX_POLICY_BYTES = 1_048_576
_SOURCE = Annotated[StrictStr, Field(min_length=1, max_length=4096)]
_IDENTIFIER = Annotated[StrictStr, Field(min_length=1, max_length=128)]


class _PrincipalSpec(BaseModel):
    """Validate one locally configured caller and its group memberships."""

    model_config = ConfigDict(extra="forbid")
    id: _IDENTIFIER
    token: StrictStr
    groups: list[_IDENTIFIER] = Field(default_factory=list, max_length=100)


class _PrincipalsSpec(BaseModel):
    """Bound the number of query callers loaded from the environment."""

    model_config = ConfigDict(extra="forbid")
    principals: list[_PrincipalSpec] = Field(min_length=1, max_length=100)


class _GrantSpec(BaseModel):
    """Represent an explicit allow grant for one indexed source identifier."""

    model_config = ConfigDict(extra="forbid")
    source: _SOURCE
    principals: list[_IDENTIFIER] = Field(default_factory=list, max_length=100)
    groups: list[_IDENTIFIER] = Field(default_factory=list, max_length=100)


class _PolicySpec(BaseModel):
    """Validate the bounded on-disk source grant format."""

    model_config = ConfigDict(extra="forbid")
    version: int = Field(strict=True, ge=1, le=1)
    sources: list[_GrantSpec] = Field(max_length=1000)


@dataclass(frozen=True, slots=True)
class _Principal:
    id: str
    groups: frozenset[str]
    token_digest: bytes


@dataclass(frozen=True, slots=True)
class _Grant:
    source: str
    principals: frozenset[str]
    groups: frozenset[str]


@dataclass(frozen=True, slots=True)
class AccessPolicy:
    """Immutable startup snapshot of caller credentials and source grants.

    Raw tokens are discarded after loading. A token maps to one principal;
    unlisted sources and principals have no query access.
    """

    principals: tuple[_Principal, ...]
    grants: tuple[_Grant, ...]

    def sources_for_token(self, token: str) -> tuple[str, ...] | None:
        """Return exact visible sources, or None for an unknown credential.

        Digest comparison checks every configured principal; an authenticated
        principal may still have an empty result when no sources are granted.
        """
        candidate = sha256(token.encode("ascii")).digest()
        principal = None
        for item in self.principals:
            if compare_digest(candidate, item.token_digest):
                principal = item
        if principal is None:
            return None
        return tuple(
            grant.source
            for grant in self.grants
            if principal.id in grant.principals
            or not principal.groups.isdisjoint(grant.groups)
        )


def load_access_policy(admin_token_digest: bytes) -> AccessPolicy | None:
    """Load opt-in query identities and source grants at API startup.

    Both environment settings must be present together. Invalid, oversized,
    or inconsistent policy data fails startup; the private file is never
    written by the API. Rotation and grant changes require a process restart.
    """
    raw_principals = os.getenv("RAG_API_PRINCIPALS_JSON")
    policy_path = os.getenv("RAG_API_ACCESS_POLICY_PATH")
    if raw_principals is None and policy_path is None:
        return None
    if not raw_principals or not policy_path:
        raise ValueError(
            "RAG_API_PRINCIPALS_JSON and RAG_API_ACCESS_POLICY_PATH "
            "must be configured together."
        )
    if len(raw_principals.encode("utf-8")) > _MAX_POLICY_BYTES:
        raise ValueError("RAG_API_PRINCIPALS_JSON exceeds the size limit.")
    try:
        principal_spec = _PrincipalsSpec.model_validate(
            _parse_strict_json(raw_principals)
        )
        with Path(policy_path).open("rb") as policy_file:
            raw_policy = policy_file.read(_MAX_POLICY_BYTES + 1)
        if len(raw_policy) > _MAX_POLICY_BYTES:
            raise ValueError("Access policy exceeds the size limit.")
        policy_spec = _PolicySpec.model_validate(_parse_strict_json(raw_policy))
    except (OSError, UnicodeError, ValueError, ValidationError):
        raise ValueError("Invalid HTTP query access configuration.") from None

    principals: list[_Principal] = []
    ids: set[str] = set()
    digests: set[bytes] = {admin_token_digest}
    for item in principal_spec.principals:
        if (
            _IDENTIFIER_PATTERN.fullmatch(item.id) is None
            or len(item.groups) != len(set(item.groups))
            or any(
                _IDENTIFIER_PATTERN.fullmatch(group) is None for group in item.groups
            )
            or TOKEN_PATTERN.fullmatch(item.token) is None
            or item.id in ids
        ):
            raise ValueError("Invalid HTTP query principal configuration.")
        digest = sha256(item.token.encode("ascii")).digest()
        if digest in digests:
            raise ValueError("HTTP query and admin tokens must be distinct.")
        ids.add(item.id)
        digests.add(digest)
        principals.append(_Principal(item.id, frozenset(item.groups), digest))

    grants: list[_Grant] = []
    seen_sources: set[str] = set()
    for grant in policy_spec.sources:
        if (
            grant.source in seen_sources
            or not grant.source.strip()
            or any(ord(char) < 32 for char in grant.source)
            or not (grant.principals or grant.groups)
            or len(grant.principals) != len(set(grant.principals))
            or len(grant.groups) != len(set(grant.groups))
            or any(principal not in ids for principal in grant.principals)
            or any(
                _IDENTIFIER_PATTERN.fullmatch(group) is None for group in grant.groups
            )
        ):
            raise ValueError("Invalid HTTP source access policy.")
        seen_sources.add(grant.source)
        grants.append(
            _Grant(grant.source, frozenset(grant.principals), frozenset(grant.groups))
        )
    return AccessPolicy(tuple(principals), tuple(grants))


def _parse_strict_json(raw: str | bytes) -> object:
    """Reject duplicate object keys so access settings cannot be shadowed."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key.")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=unique_object)
