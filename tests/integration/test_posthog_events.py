import io
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from uuid import uuid4

from tests.helpers.fake_claude import FakeClaude, tool_use
from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_storage import FakeStorage
from tests.helpers.fake_posthog import FakePostHog


def _csv(df):
    b = io.StringIO()
    df.to_csv(b, index=False)
    return b.getvalue().encode("utf-8")


@pytest.fixture
def fakes_and_client(sales):
    calls = [tool_use("frequency_bar_chart",
                      {"column": "region", "title": f"T{i}", "intent": f"i{i}"},
                      id_=f"tu_{i}")
             for i in range(10)]
    fake_claude = FakeClaude([
        {"tool_calls": calls},
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "S.", "captions": [f"c{i}" for i in range(10)], "data_quality": []})]},
    ])
    fake_db = FakeDB()
    fake_storage = FakeStorage()
    fake_posthog = FakePostHog()

    from main import app, get_claude_client, get_db, get_storage, get_posthog
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=fake_claude)
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[get_storage] = lambda: fake_storage
    app.dependency_overrides[get_posthog] = lambda: fake_posthog
    yield TestClient(app), fake_posthog
    app.dependency_overrides.clear()


def test_generate_report_fires_started_and_succeeded(fakes_and_client, sales):
    tc, ph = fakes_and_client
    anon = str(uuid4())
    resp = tc.post("/generate-report",
                   files={"file": ("sales.csv", _csv(sales), "text/csv")},
                   headers={"X-Anon-Id": anon})
    assert resp.status_code == 200

    started = ph.find("report_generation_started")
    succeeded = ph.find("report_generation_succeeded")
    assert len(started) == 1
    assert len(succeeded) == 1
    # Keys are camelCase
    s = succeeded[0]["properties"]
    for required in ("reportId", "rowCount", "columnCount", "chartCount",
                     "modelSelection", "estCostUsd", "elapsedMs"):
        assert required in s, f"missing key {required} in {list(s.keys())}"
    # No snake_case leakage
    for k in s:
        assert "_" not in k or k.startswith("$"), f"property {k} contains underscore"
    # Smarter-analysis: deep + custom-prompt adoption must be measurable on the funnel.
    assert s["deep"] is False
    assert s["customPrompt"] is False


def test_custom_prompt_flagged_in_generation_events(fakes_and_client, sales):
    """The free 'focus' field is otherwise invisible (no credit spend), so its
    adoption must ride the generation events as a boolean."""
    tc, ph = fakes_and_client
    anon = str(uuid4())
    resp = tc.post("/generate-report",
                   files={"file": ("sales.csv", _csv(sales), "text/csv")},
                   data={"custom_prompt": "focus on region performance"},
                   headers={"X-Anon-Id": anon})
    assert resp.status_code == 200
    started = ph.find("report_generation_started")[0]["properties"]
    succeeded = ph.find("report_generation_succeeded")[0]["properties"]
    assert started["customPrompt"] is True
    assert succeeded["customPrompt"] is True
    # Anon path is never deep (deep requires auth, gated before the started event).
    assert started["deep"] is False


def test_anon_limit_blocked_event_camelcase_only(fakes_and_client, sales):
    tc, ph = fakes_and_client
    anon = str(uuid4())
    tc.post("/generate-report",
            files={"file": ("sales.csv", _csv(sales), "text/csv")},
            headers={"X-Anon-Id": anon})
    tc.post("/generate-report",
            files={"file": ("sales.csv", _csv(sales), "text/csv")},
            headers={"X-Anon-Id": anon})
    blocked = ph.find("anon_limit_blocked")
    assert len(blocked) == 1
    for k in blocked[0]["properties"]:
        assert "_" not in k or k.startswith("$")
