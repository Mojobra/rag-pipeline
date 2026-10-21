"""Load a private source-access policy for opt-in HTTP query authorization.

This adapter binds high-entropy query credentials to local principals, tenants,
and exact indexed sources; it does not sync remote ACLs or verify human identity.
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
    tenant: _IDENTIFIER | None = None


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
    tenant: _IDENTIFIER | None = None


class _PolicySpec(BaseModel):
    """Validate the bounded on-disk source grant format."""

    model_config = ConfigDict(extra="forbid")
    version: int = Field(strict=True, ge=1, le=1)
    sources: list[_GrantSpec] = Field(max_length=1000)


class _TenantSpec(BaseModel):
    """Bind one tenant ID to an operator-owned local Qdrant location."""

    model_config = ConfigDict(extra="forbid")
    id: _IDENTIFIER
    store_path: _SOURCE
    collection_name: _IDENTIFIER


class _TenantPolicySpec(BaseModel):
    """Validate the v2 policy envelope before cross-tenant checks."""

    model_config = ConfigDict(extra="forbid")
    version: int = Field(strict=True, ge=2, le=2)
    tenants: list[_TenantSpec] = Field(min_length=1, max_length=16)
    sources: list[_GrantSpec] = Field(max_length=1000)


@dataclass(frozen=True, slots=True)
class _Principal:
    id: str
    groups: frozenset[str]
    token_digest: bytes
    tenant_id: str | None


@dataclass(frozen=True, slots=True)
class _Grant:
    source: str
    principals: frozenset[str]
    groups: frozenset[str]
    tenant_id: str | None


@dataclass(frozen=True, slots=True)
class TenantStorage:
    """Validated, distinct on-disk Qdrant location for one local tenant."""

    id: str
    store_path: Path
    collection_name: str


@dataclass(frozen=True, slots=True)
class QueryAccess:
    """Trusted query scope derived from a token, never from request fields."""

    tenant_id: str | None
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AccessPolicy:
    """Immutable startup snapshot of caller credentials and source grants.

    Raw tokens are discarded after loading. A token maps to one principal;
    unlisted sources and principals have no query access.
    """

    principals: tuple[_Principal, ...]
    grants: tuple[_Grant, ...]
    tenants: tuple[TenantStorage, ...] = ()

    def sources_for_token(self, token: str) -> tuple[str, ...] | None:
        """Return visible sources while preserving the v1 policy interface."""
        access = self.access_for_token(token)
        return None if access is None else access.sources

    def access_for_token(self, token: str) -> QueryAccess | None:
        """Resolve an authenticated tenant and exact source allowlist.

        Digest comparison checks every configured principal; an authenticated
        principal may still have an empty source list. V2 grants are evaluated
        only within the token's tenant, even when source or group names repeat.
        """
        candidate = sha256(token.encode("ascii")).digest()
        principal = None
        for item in self.principals:
            if compare_digest(candidate, item.token_digest):
                principal = item
        if principal is None:
            return None
        return QueryAccess(
            tenant_id=principal.tenant_id,
            sources=tuple(
                grant.source
                for grant in self.grants
                if grant.tenant_id == principal.tenant_id
                and (
                    principal.id in grant.principals
                    or not principal.groups.isdisjoint(grant.groups)
                )
            ),
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
        policy_data = _parse_strict_json(raw_policy)
        policy_spec: _PolicySpec | _TenantPolicySpec
        if isinstance(policy_data, dict) and policy_data.get("version") == 2:
            policy_spec = _TenantPolicySpec.model_validate(policy_data)
        else:
            policy_spec = _PolicySpec.model_validate(policy_data)
    except (OSError, UnicodeError, ValueError, ValidationError):
        raise ValueError("Invalid HTTP query access configuration.") from None

    tenants = (
        _validate_tenants(policy_spec.tenants)
        if isinstance(policy_spec, _TenantPolicySpec)
        else ()
    )
    tenant_ids = {tenant.id for tenant in tenants}
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
            or (item.tenant not in tenant_ids if tenants else item.tenant is not None)
        ):
            raise ValueError("Invalid HTTP query principal configuration.")
        digest = sha256(item.token.encode("ascii")).digest()
        if digest in digests:
            raise ValueError("HTTP query and admin tokens must be distinct.")
        ids.add(item.id)
        digests.add(digest)
        principals.append(
            _Principal(item.id, frozenset(item.groups), digest, item.tenant)
        )

    grants: list[_Grant] = []
    seen_sources: set[tuple[str | None, str]] = set()
    for grant in policy_spec.sources:
        grant_principals = set(grant.principals)
        if (
            (grant.tenant, grant.source) in seen_sources
            or not grant.source.strip()
            or any(ord(char) < 32 for char in grant.source)
            or not (grant.principals or grant.groups)
            or len(grant.principals) != len(set(grant.principals))
            or len(grant.groups) != len(set(grant.groups))
            or any(principal not in ids for principal in grant.principals)
            or (grant.tenant not in tenant_ids if tenants else grant.tenant is not None)
            or any(
                principal.id in grant_principals and principal.tenant_id != grant.tenant
                for principal in principals
            )
            or any(
                _IDENTIFIER_PATTERN.fullmatch(group) is None for group in grant.groups
            )
        ):
            raise ValueError("Invalid HTTP source access policy.")
        seen_sources.add((grant.tenant, grant.source))
        grants.append(
            _Grant(
                grant.source,
                frozenset(grant.principals),
                frozenset(grant.groups),
                grant.tenant,
            )
        )
    return AccessPolicy(tuple(principals), tuple(grants), tenants)


def _validate_tenants(specs: list[_TenantSpec]) -> tuple[TenantStorage, ...]:
    """Reject duplicate IDs and path aliases/nesting before opening any store.

    Paths resolve relative to the API process working directory. Each tenant
    must own a separate directory, not merely a collection in shared files.
    """
    tenants: list[TenantStorage] = []
    try:
        for item in specs:
            if (
                _IDENTIFIER_PATTERN.fullmatch(item.id) is None
                or _IDENTIFIER_PATTERN.fullmatch(item.collection_name) is None
                or not item.store_path.strip()
                or any(tenant.id == item.id for tenant in tenants)
            ):
                raise ValueError("Invalid tenant identifier or collection.")
            path = Path(item.store_path).expanduser().resolve()
            if path == Path(path.anchor) or any(
                path == tenant.store_path
                or path in tenant.store_path.parents
                or tenant.store_path in path.parents
                for tenant in tenants
            ):
                raise ValueError("Tenant store paths must be disjoint directories.")
            if path.exists() and not path.is_dir():
                raise ValueError("Tenant store path is not a directory.")
            tenants.append(TenantStorage(item.id, path, item.collection_name))
    except (OSError, RuntimeError, ValueError):
        raise ValueError("Invalid HTTP tenant storage configuration.") from None
    return tuple(tenants)


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
