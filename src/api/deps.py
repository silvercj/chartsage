"""FastAPI dependencies shared across endpoints."""
import hmac
import os
from dataclasses import dataclass
from uuid import UUID
from fastapi import Header, HTTPException
from auth import verify_token


def get_anon_id(x_anon_id: str | None = Header(None)) -> UUID:
    """Parse the X-Anon-Id header into a UUID; reject missing or malformed."""
    if not x_anon_id:
        raise HTTPException(
            status_code=400,
            detail={"code": "MISSING_ANON_ID",
                    "message": "X-Anon-Id header is required."},
        )
    try:
        return UUID(x_anon_id)
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_ANON_ID",
                    "message": "X-Anon-Id is not a valid UUID."},
        )


@dataclass
class Identity:
    """Who is calling. Authenticated when a valid Supabase JWT was presented."""
    user_id: UUID | None = None
    anon_id: UUID | None = None

    @property
    def is_authenticated(self) -> bool:
        return self.user_id is not None

    @property
    def distinct_id(self) -> str:
        """Stable analytics id: the user id when authenticated, else the anon id."""
        return str(self.user_id) if self.user_id else str(self.anon_id)


def get_identity(
    authorization: str | None = Header(None),
    x_anon_id: str | None = Header(None),
) -> Identity:
    """Resolve the caller's identity.

    A valid Bearer token wins (authenticated). Otherwise fall back to the
    anonymous X-Anon-Id header. A Bearer token that is present but invalid is a
    hard 401 (the client should refresh + retry) — we never silently downgrade
    to anonymous. Error codes match the legacy get_anon_id codes so existing
    tests stay green.
    """
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        user_id = verify_token(token)
        if user_id is not None:
            return Identity(user_id=user_id)
        raise HTTPException(
            status_code=401,
            detail={"code": "INVALID_TOKEN",
                    "message": "Your session is invalid or expired. Please sign in again."},
        )
    if x_anon_id:
        try:
            return Identity(anon_id=UUID(x_anon_id))
        except (ValueError, AttributeError):
            raise HTTPException(
                status_code=400,
                detail={"code": "INVALID_ANON_ID",
                        "message": "X-Anon-Id is not a valid UUID."},
            )
    raise HTTPException(
        status_code=400,
        detail={"code": "MISSING_ANON_ID",
                "message": "Authentication or X-Anon-Id header is required."},
    )


def get_identity_optional(
    authorization: str | None = Header(None),
    x_anon_id: str | None = Header(None),
) -> Identity:
    """Like get_identity but NEVER raises — for public endpoints that may be
    called with no auth (e.g. Next's server-side generateMetadata). A valid
    Bearer yields the user; anything else yields an anonymous Identity."""
    if authorization and authorization.lower().startswith("bearer "):
        user_id = verify_token(authorization[7:].strip())
        if user_id is not None:
            return Identity(user_id=user_id)
    if x_anon_id:
        try:
            return Identity(anon_id=UUID(x_anon_id))
        except (ValueError, AttributeError):
            pass
    return Identity()


def require_admin(x_admin_token: str | None = Header(None)) -> None:
    """Gate admin endpoints on a shared secret.

    Reads ADMIN_API_TOKEN from the environment at call time (so it is test- and
    rotation-friendly) and compares it to the X-Admin-Token header with a
    constant-time compare. Fail-closed: if the env var is unset/empty or the
    header is missing or mismatched, 403. The token is never logged.
    """
    expected = os.environ.get("ADMIN_API_TOKEN", "")
    if not expected or not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(
            status_code=403,
            detail={"code": "FORBIDDEN", "message": "Admin access required."},
        )
    return None
