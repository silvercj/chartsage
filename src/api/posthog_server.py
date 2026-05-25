"""Server-side PostHog wrapper.

Wraps the official posthog Python SDK with a single rule: analytics
must never break product flow. Errors are caught and logged at WARN.
"""
import logging
import os
from typing import Any, Optional


_POSTHOG_KEY = os.environ.get("POSTHOG_API_KEY")
_POSTHOG_HOST = os.environ.get("POSTHOG_HOST", "https://us.i.posthog.com")


def _build_default_client() -> Optional[Any]:
    if not _POSTHOG_KEY:
        return None
    from posthog import Posthog
    return Posthog(project_api_key=_POSTHOG_KEY, host=_POSTHOG_HOST)


class PostHogServer:
    def __init__(self, _client: Any = None):
        self.client = _client if _client is not None else _build_default_client()

    def capture(
        self,
        distinct_id: str,
        event: str,
        properties: Optional[dict] = None,
    ) -> None:
        if self.client is None:
            return   # not configured; silent no-op
        try:
            self.client.capture(
                distinct_id=str(distinct_id),
                event=event,
                properties=properties or {},
            )
        except Exception as e:
            logging.warning("[POSTHOG] capture failed for %s: %s", event, e)
