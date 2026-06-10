import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from uuid import uuid4

from tests.helpers.fake_claude import FakeClaude, tool_use, ten_distinct_chart_calls
from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_storage import FakeStorage
from tests.helpers.fake_posthog import FakePostHog
from tests.helpers.fake_auth import auth_identity, anon_identity


def _csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _ten_chart_fake():
    return FakeClaude([
        {"tool_calls": ten_distinct_chart_calls()},
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "S.", "captions": [f"c{i}" for i in range(10)],
                                  "data_quality": []})]},
    ])


class _Holder:
    def __init__(self):
        self.current = None

    def __call__(self):
        return self.current


@pytest.fixture
def ctx(sales):
    fake_db, fake_storage, fake_posthog = FakeDB(), FakeStorage(), FakePostHog()
    holder = _Holder()
    from main import app, get_claude_client, get_db, get_storage, get_posthog, get_identity
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=_ten_chart_fake())
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[get_storage] = lambda: fake_storage
    app.dependency_overrides[get_posthog] = lambda: fake_posthog
    app.dependency_overrides[get_identity] = holder
    yield TestClient(app), fake_db, fake_posthog, holder
    app.dependency_overrides.clear()


def test_authenticated_reports_consume_credits(ctx, sales):
    tc, db, _, holder = ctx
    user = str(uuid4())
    holder.current = auth_identity(user)
    # 300 starter / 100 = exactly 3 reports succeed...
    for _ in range(3):
        resp = tc.post("/generate-report",
                       files={"file": ("s.csv", _csv_bytes(sales), "text/csv")})
        assert resp.status_code == 200
    assert db.get_balance(user) == 0
    # ...the 4th is out of credits.
    resp = tc.post("/generate-report",
                   files={"file": ("s.csv", _csv_bytes(sales), "text/csv")})
    assert resp.status_code == 402
    assert resp.json()["detail"]["code"] == "OUT_OF_CREDITS"
    assert len(db._rows) == 3


def test_anonymous_generate_more_returns_402(ctx, sales):
    tc, db, _, holder = ctx
    holder.current = anon_identity(str(uuid4()))
    resp = tc.post("/generate-report",
                   files={"file": ("s.csv", _csv_bytes(sales), "text/csv")})
    assert resp.status_code == 200
    sid = resp.json()["session_id"]
    resp2 = tc.post(f"/report/{sid}/generate-more")
    assert resp2.status_code == 402
    assert resp2.json()["detail"]["code"] == "UPGRADE_REQUIRED"


def test_authenticated_generate_more_allowed(ctx, sales):
    tc, db, _, holder = ctx
    holder.current = auth_identity(str(uuid4()))
    resp = tc.post("/generate-report",
                   files={"file": ("s.csv", _csv_bytes(sales), "text/csv")})
    sid = resp.json()["session_id"]
    new = FakeClaude([
        {"tool_calls": [tool_use("scatter_chart", {"x_col": "order_id", "y_col": "revenue", "title": "More", "intent": "n"}, id_="m0")]},
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "U.", "captions": ["c"], "data_quality": []})]},
    ])
    from main import app, get_claude_client
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=new)
    resp2 = tc.post(f"/report/{sid}/generate-more")
    assert resp2.status_code == 200
    assert len(resp2.json()["charts"]) == 11
    user_id = holder.current.user_id
    assert db.get_balance(str(user_id)) == 160
