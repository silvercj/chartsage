"""In-memory PostHog client that records events without sending them."""


class FakePostHog:
    def __init__(self):
        self.events: list[dict] = []

    def capture(self, distinct_id: str, event: str, properties: dict | None = None) -> None:
        self.events.append({
            "distinct_id": str(distinct_id),
            "event": event,
            "properties": properties or {},
        })

    def find(self, event_name: str) -> list[dict]:
        """Helper for tests: return all events with a given name."""
        return [e for e in self.events if e["event"] == event_name]
