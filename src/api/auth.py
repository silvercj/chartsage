"""Supabase JWT verification via cached JWKS.

Authenticated browser requests carry `Authorization: Bearer <supabase access token>`.
We verify the token's signature against Supabase's public JWKS (asymmetric ES256/RS256),
plus audience + expiry, entirely locally — no per-request call to Supabase.
"""
import os
from uuid import UUID

import jwt
from jwt import PyJWKClient

_SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
_AUDIENCE = "authenticated"
_ALGORITHMS = ["ES256", "RS256"]

_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        if not _SUPABASE_URL:
            raise RuntimeError("SUPABASE_URL must be set for JWT verification")
        url = f"{_SUPABASE_URL}/auth/v1/.well-known/jwks.json"
        _jwks_client = PyJWKClient(url, cache_keys=True)
    return _jwks_client


def verify_token(token: str, *, _public_key=None) -> UUID | None:
    """Return the user UUID (the `sub` claim) if the token is valid, else None.

    `_public_key` is a test injection point: pass a PEM public key to verify
    against directly instead of fetching the live JWKS.
    """
    try:
        if _public_key is not None:
            key = _public_key
            algorithms = ["RS256", "ES256"]
        else:
            key = _get_jwks_client().get_signing_key_from_jwt(token).key
            algorithms = _ALGORITHMS
        claims = jwt.decode(
            token,
            key,
            algorithms=algorithms,
            audience=_AUDIENCE,
            options={"verify_exp": True},
        )
    except Exception:
        return None
    sub = claims.get("sub")
    if not sub:
        return None
    try:
        return UUID(str(sub))
    except (ValueError, AttributeError):
        return None
