"""FastAPI dependencies shared across endpoints."""
from uuid import UUID
from fastapi import Header, HTTPException


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
