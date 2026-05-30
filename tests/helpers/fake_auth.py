"""Identity builders for integration tests."""
from uuid import UUID

from deps import Identity


def auth_identity(user_id: str) -> Identity:
    return Identity(user_id=UUID(user_id))


def anon_identity(anon_id: str) -> Identity:
    return Identity(anon_id=UUID(anon_id))
