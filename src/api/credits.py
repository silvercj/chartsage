"""Credit economy config + the InsufficientCredits signal.

The atomic credit data operations live on the DB object (SupabaseDB / FakeDB),
backed by Postgres functions. This module holds the tunable costs (read from the
environment, with psychologically-tuned defaults) and the exception those
operations raise when a balance can't cover a spend.
"""
import os


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


REPORT_COST = _int_env("REPORT_COST", 100)
GENERATE_MORE_COST = _int_env("GENERATE_MORE_COST", 40)
SIGNUP_GRANT = _int_env("SIGNUP_GRANT", 300)


class InsufficientCredits(Exception):
    """Raised by the DB layer when a spend exceeds the available balance."""
