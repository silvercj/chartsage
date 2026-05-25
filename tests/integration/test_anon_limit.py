import io
import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from uuid import uuid4

from tests.helpers.fake_claude import FakeClaude, tool_use
from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_storage import FakeStorage
from tests.helpers.fake_posthog import FakePostHog


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _ten_chart_fake():
    calls = [tool_use("frequency_bar_chart",
                      {"column": "region", "title": f"T{i}", "intent": f"i{i}"},
                      id_=f"tu_{i}")
             for i in range(10)]
    return FakeClaude([
        {"tool_calls": calls},
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "S.", "captions": [f"c{i}" for i in range(10)], "data_quality": []})]},
    ])


@pytest.fixture
def client_and_fakes(sales):
    fake_db = FakeDB()
    fake_storage = FakeStorage()
    fake_posthog = FakePostHog()

    def fake_claude_factory():
        return MagicMock(messages_create=_ten_chart_fake())

    from main import app, get_claude_client, get_db, get_storage, get_posthog
    app.dependency_overrides[get_claude_client] = fake_claude_factory
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[get_storage] = lambda: fake_storage
    app.dependency_overrides[get_posthog] = lambda: fake_posthog

    yield TestClient(app), fake_db, fake_storage, fake_posthog
    app.dependency_overrides.clear()


def _post_report(tc, sales, anon_id):
    return tc.post("/generate-report",
                   files={"file": ("sales.csv", _csv_bytes(sales), "text/csv")},
                   headers={"X-Anon-Id": anon_id})


def test_first_report_succeeds(client_and_fakes, sales):
    tc, db, _, _ = client_and_fakes
    anon = str(uuid4())
    resp = _post_report(tc, sales, anon)
    assert resp.status_code == 200
    assert db.count_anon_reports(anon) == 1


def test_second_report_blocked_with_anon_limit_code(client_and_fakes, sales):
    tc, db, _, ph = client_and_fakes
    anon = str(uuid4())
    _post_report(tc, sales, anon)
    resp = _post_report(tc, sales, anon)
    assert resp.status_code == 403
    body = resp.json()
    assert body["detail"]["code"] == "ANON_LIMIT_REACHED"
    # PostHog: server fired anon_limit_blocked
    assert len(ph.find("anon_limit_blocked")) == 1


def test_different_anon_not_blocked(client_and_fakes, sales):
    tc, *_ = client_and_fakes
    anon_a = str(uuid4())
    anon_b = str(uuid4())
    _post_report(tc, sales, anon_a)
    resp = _post_report(tc, sales, anon_b)
    assert resp.status_code == 200
