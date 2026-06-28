"""The public showcase report served at /sample-report.

It must generate ONCE from the bundled samples/showcase.csv, cache at the fixed
SAMPLE_REPORT_ID, be marked public (any visitor can view it), and never debit credits
or count against the anon free-report cap. A second call serves the cached row without
re-invoking the model.
"""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from main import app, SAMPLE_REPORT_ID, get_claude_client, get_db, get_storage, get_posthog
from tests.helpers.fake_claude import FakeClaude, tool_use, ten_distinct_chart_calls
from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_storage import FakeStorage
from tests.helpers.fake_posthog import FakePostHog


@pytest.fixture
def ctx():
    fake_db, fake_storage, fake_posthog = FakeDB(), FakeStorage(), FakePostHog()
    # One shared FakeClaude so we can assert how many model calls happened across requests.
    fake_claude = FakeClaude([
        {"tool_calls": ten_distinct_chart_calls()},   # 10 distinct charts (>= target, no reach-for-more)
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "A strong quarter.", "captions": [f"c{i}" for i in range(10)],
                                  "data_quality": []})]},
    ])
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=fake_claude)
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[get_storage] = lambda: fake_storage
    app.dependency_overrides[get_posthog] = lambda: fake_posthog
    yield TestClient(app), fake_db, fake_claude, fake_posthog
    app.dependency_overrides.clear()


def test_first_call_generates_public_sample(ctx):
    tc, db, fake_claude, _ = ctx
    resp = tc.get("/sample-report")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == SAMPLE_REPORT_ID

    row = db.get_report(SAMPLE_REPORT_ID)
    assert row is not None
    assert row["is_public"] is True            # any visitor can view it
    assert row["user_id"] is None and row["anon_id"] is None
    assert len(row["report_json"]["charts"]) >= 3


def test_second_call_serves_cache_without_regenerating(ctx):
    tc, db, fake_claude, _ = ctx
    tc.get("/sample-report")
    calls_after_first = len(fake_claude.calls)
    resp = tc.get("/sample-report")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == SAMPLE_REPORT_ID
    assert len(fake_claude.calls) == calls_after_first   # no extra model calls on the cache hit


def test_sample_never_debits_or_counts_against_anon_cap(ctx):
    tc, db, _, _ = ctx
    tc.get("/sample-report")
    # No anon free-report log entries and no credit ledger spend from the sample path.
    assert db._anon_log == []                # anon free-report log untouched
    assert all(t.get("reason") != "report" for t in db._txns)
    # The sample row carries no anon_id, so it never counts against any visitor's cap.
    assert db.count_anon_reports("any-anon-id") == 0


def test_sample_report_is_viewable_anonymously(ctx):
    tc, _, _, _ = ctx
    tc.get("/sample-report")                 # generate it
    # A logged-out visitor can GET the report itself (public visibility).
    resp = tc.get(f"/report/{SAMPLE_REPORT_ID}")
    assert resp.status_code == 200
    assert resp.json()["charts"]
