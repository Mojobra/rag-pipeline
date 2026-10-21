"""Authenticate local HTTP queries and derive optional source-scoped access."""

from __future__ import annotations

import os
from hashlib import sha256
from hmac import compare_digest
from typing import Annotated, cast

from fastapi import HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rag_pipeline.api.authorization import TOKEN_PATTERN, AccessPolicy, QueryAccess

_BEARER_SCHEME = HTTPBearer(auto_error=False)
_UNAUTHORIZED = "A valid bearer token is required."


def load_bearer_token_digest() -> bytes:
    """Validate the configured high-entropy token before the app accepts traffic.

    The raw environment value is never put in API responses or app state. The
    fixed-size digest supports timing-resistant comparisons during requests.
    A missing or malformed token fails startup rather than disabling auth.
    """
    token = os.getenv("RAG_API_BEARER_TOKEN")
    if token is None or TOKEN_PATTERN.fullmatch(token) is None:
        raise ValueError(
            "RAG_API_BEARER_TOKEN must contain 32-256 URL-safe characters. "
            "Generate a random token before starting the API."
        )
    return sha256(token.encode("ascii")).digest()


def require_bearer_token(
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Security(_BEARER_SCHEME)
    ],
) -> None:
    """Reject missing or incorrect admin credentials before protected work.

    All failures share one response and a Bearer challenge. Only the token's
    digest is stored for the lifetime of the process; rotation needs restart.
    """
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or TOKEN_PATTERN.fullmatch(credentials.credentials) is None
    ):
        raise _authentication_error()

    expected_digest = cast(bytes, request.app.state.bearer_token_digest)
    candidate_digest = sha256(credentials.credentials.encode("ascii")).digest()
    if not compare_digest(candidate_digest, expected_digest):
        raise _authentication_error()


def require_query_access(
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Security(_BEARER_SCHEME)
    ],
) -> QueryAccess | None:
    """Authenticate a query and derive its source allowlist server-side.

    Legacy mode retains the shared token and unrestricted local behavior.
    Policy mode accepts only principal tokens, never the metrics/admin token;
    a QueryAccess with empty sources means the caller has no readable evidence.
    """
    policy = cast(AccessPolicy | None, request.app.state.access_policy)
    if policy is None:
        require_bearer_token(request, credentials)
        return None
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or TOKEN_PATTERN.fullmatch(credentials.credentials) is None
    ):
        raise _authentication_error()
    access = policy.access_for_token(credentials.credentials)
    if access is None:
        raise _authentication_error()
    return access


def _authentication_error() -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=_UNAUTHORIZED,
        headers={"WWW-Authenticate": "Bearer"},
    )
