import io

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from uuid import uuid4

from tests.helpers.fake_claude import FakeClaude, tool_use, ten_distinct_chart_calls
from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_storage import FakeStorage
from tests.helpers.fake_posthog import FakePostHog
from tests.helpers.fake_auth import auth_identity


def _csv_bytes(df):
    buf = io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode("utf-8")


def _ten_chart_fake():
    return FakeClaude([
        {"tool_calls": ten_distinct_chart_calls()},
        {"tool_calls": [tool_use("submit_narrative", {"summary": "S.", "captions": [f"c{i}" for i in range(10)], "data_quality": []})]},
    ])


class _Holder:
    def __init__(self): self.current = None
    def __call__(self): return self.current


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
    yield TestClient(app), fake_db, holder
    app.dependency_overrides.clear()


def _post(tc, sales):
    return tc.post("/generate-report", files={"file": ("s.csv", _csv_bytes(sales), "text/csv")})


def test_report_spends_100(ctx, sales):
    tc, db, holder = ctx
    user = str(uuid4()); holder.current = auth_identity(user)
    resp = _post(tc, sales)
    assert resp.status_code == 200
    assert db.get_balance(user) == 200                      # 300 - 100
    txns = db.list_transactions(user)
    assert txns[0]["delta"] == -100 and txns[0]["reason"] == "report"


def test_report_out_of_credits(ctx, sales):
    tc, db, holder = ctx
    user = str(uuid4()); holder.current = auth_identity(user)
    db.ensure_profile(user, 300)
    db.spend_credits(user, 250, "adjustment")               # leave 50 (< 100)
    resp = _post(tc, sales)
    assert resp.status_code == 402
    body = resp.json()["detail"]
    assert body["code"] == "OUT_OF_CREDITS" and body["cost"] == 100 and body["balance"] == 50
    assert len(db._rows) == 0                                # no report saved
    assert db.get_balance(user) == 50                        # no spend


def test_generate_more_spends_40(ctx, sales):
    tc, db, holder = ctx
    user = str(uuid4()); holder.current = auth_identity(user)
    sid = _post(tc, sales).json()["session_id"]             # balance now 200
    new = FakeClaude([
        {"tool_calls": [tool_use("scatter_chart", {"x_col": "order_id", "y_col": "revenue", "title": "M", "intent": "n"}, id_="m0")]},
        {"tool_calls": [tool_use("submit_narrative", {"summary": "U.", "captions": ["c"], "data_quality": []})]},
    ])
    from main import app, get_claude_client
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=new)
    resp = tc.post(f"/report/{sid}/generate-more")
    assert resp.status_code == 200
    assert db.get_balance(user) == 160                       # 200 - 40


def test_generate_more_uses_stored_csv_key_not_url_id(ctx, sales):
    """Regression: reports.id is a Postgres uuid (returned in dashed form) but the
    CSV is stored under the original uuid4().hex report_id. generate-more must
    download via the stored csv_storage_key, not rebuild {url_id}.csv — otherwise
    opening a report from My Reports (dashed id) 404s with SOURCE_DATA_UNAVAILABLE."""
    tc, db, holder = ctx
    user = str(uuid4()); holder.current = auth_identity(user)
    sid = _post(tc, sales).json()["session_id"]              # hex id; CSV under {sid}.csv
    # Simulate Postgres uuid normalization: the row id becomes the dashed form while
    # the CSV stays under its original (hex) key recorded in csv_storage_key.
    dashed = "06cfac63-d0d5-4378-b9f9-02e76ffa178d"
    row = db._rows.pop(sid); row["id"] = dashed; db._rows[dashed] = row
    new = FakeClaude([
        {"tool_calls": [tool_use("scatter_chart", {"x_col": "order_id", "y_col": "revenue", "title": "M", "intent": "n"}, id_="m0")]},
        {"tool_calls": [tool_use("submit_narrative", {"summary": "U.", "captions": ["c"], "data_quality": []})]},
    ])
    from main import app, get_claude_client
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=new)
    resp = tc.post(f"/report/{dashed}/generate-more")
    assert resp.status_code == 200                            # old code: 404 SOURCE_DATA_UNAVAILABLE


def test_generate_more_out_of_credits(ctx, sales):
    tc, db, holder = ctx
    user = str(uuid4()); holder.current = auth_identity(user)
    sid = _post(tc, sales).json()["session_id"]             # 200 left
    db.spend_credits(user, 170, "adjustment")               # leave 30 (< 40)
    resp = tc.post(f"/report/{sid}/generate-more")
    assert resp.status_code == 402
    assert resp.json()["detail"]["code"] == "OUT_OF_CREDITS"


def test_failed_generation_does_not_debit(ctx, sales):
    tc, db, holder = ctx
    user = str(uuid4()); holder.current = auth_identity(user)
    # Claude blows up during generation -> 5xx, and the user must NOT be charged.
    from main import app, get_claude_client
    boom = MagicMock()
    boom.messages_create.side_effect = RuntimeError("claude exploded")
    app.dependency_overrides[get_claude_client] = lambda: boom
    resp = tc.post("/generate-report", files={"file": ("s.csv", _csv_bytes(sales), "text/csv")})
    assert resp.status_code >= 500
    assert db.get_balance(user) == 300        # pre-check granted 300; no spend on failure
    assert len(db._rows) == 0
