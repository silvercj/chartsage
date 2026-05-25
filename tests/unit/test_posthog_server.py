import logging
from unittest.mock import MagicMock
import pytest
from tests.helpers.fake_posthog import FakePostHog


def test_capture_records_event():
    p = FakePostHog()
    p.capture("anon-uuid-1", "report_viewed", {"reportId": "r1"})
    assert p.events == [{
        "distinct_id": "anon-uuid-1",
        "event": "report_viewed",
        "properties": {"reportId": "r1"},
    }]


def test_capture_without_properties():
    p = FakePostHog()
    p.capture("u", "landing_viewed")
    assert p.events[0]["properties"] == {}


def test_find_filters_by_event_name():
    p = FakePostHog()
    p.capture("u", "a")
    p.capture("u", "b")
    p.capture("u", "a")
    assert len(p.find("a")) == 2
    assert len(p.find("c")) == 0


def test_server_client_swallows_errors(caplog):
    """The real PostHogServer must never let analytics failures break product flow."""
    from posthog_server import PostHogServer

    boom = MagicMock()
    boom.capture.side_effect = RuntimeError("posthog is down")
    server = PostHogServer(_client=boom)

    with caplog.at_level(logging.WARNING):
        server.capture("u", "some_event", {"k": "v"})   # must not raise

    # Logged a warning
    assert any("posthog" in r.message.lower() for r in caplog.records)


def test_server_client_passes_through_to_underlying():
    from posthog_server import PostHogServer

    mock_client = MagicMock()
    server = PostHogServer(_client=mock_client)
    server.capture("u", "evt", {"reportId": "r1"})

    mock_client.capture.assert_called_once_with(
        distinct_id="u", event="evt", properties={"reportId": "r1"},
    )
