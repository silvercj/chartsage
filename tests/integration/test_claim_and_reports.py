import pytest
from fastapi.testclient import TestClient
from uuid import uuid4

from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_posthog import FakePostHog
from tests.helpers.fake_auth import auth_identity, anon_identity


class _Holder:
    def __init__(self):
        self.current = None

    def __call__(self):
        return self.current


def _report_json(n_charts):
    charts = [{"chart_id": f"c{i}", "spec": {"kind": "bar"}, "caption": f"cap{i}"}
              for i in range(n_charts)]
    return {"generated_at": "2026-01-01T00:00:00Z", "summary": "S",
            "data_quality": [], "charts": charts, "layout": [], "metadata": {}}


@pytest.fixture
def ctx():
    fake_db, fake_posthog = FakeDB(), FakePostHog()
    holder = _Holder()
    from main import app, get_db, get_posthog, get_identity
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[get_posthog] = lambda: fake_posthog
    app.dependency_overrides[get_identity] = holder
    yield TestClient(app), fake_db, holder
    app.dependency_overrides.clear()


def test_claim_moves_anon_reports_to_user(ctx):
    tc, db, holder = ctx
    anon, user = str(uuid4()), str(uuid4())
    db.save_report("r1", anon, None, _report_json(3), "k", "t")
    holder.current = auth_identity(user)
    resp = tc.post("/claim-anon-reports", headers={"X-Anon-Id": anon})
    assert resp.status_code == 200
    assert resp.json()["claimed"] == 1
    row = db.get_report("r1")
    assert row["user_id"] == user and row["anon_id"] is None


def test_claim_is_idempotent(ctx):
    tc, db, holder = ctx
    anon, user = str(uuid4()), str(uuid4())
    db.save_report("r1", anon, None, _report_json(1), "k", "t")
    holder.current = auth_identity(user)
    tc.post("/claim-anon-reports", headers={"X-Anon-Id": anon})
    resp2 = tc.post("/claim-anon-reports", headers={"X-Anon-Id": anon})
    assert resp2.json()["claimed"] == 0


def test_claim_without_anon_header_is_noop(ctx):
    tc, _, holder = ctx
    holder.current = auth_identity(str(uuid4()))
    resp = tc.post("/claim-anon-reports")
    assert resp.status_code == 200
    assert resp.json()["claimed"] == 0


def test_claim_requires_auth(ctx):
    tc, _, holder = ctx
    holder.current = anon_identity(str(uuid4()))
    resp = tc.post("/claim-anon-reports", headers={"X-Anon-Id": str(uuid4())})
    assert resp.status_code == 401


def test_my_reports_isolates_users(ctx):
    tc, db, holder = ctx
    u1, u2 = str(uuid4()), str(uuid4())
    db.save_report("r1", None, u1, _report_json(3), "k", "One")
    db.save_report("r2", None, u2, _report_json(2), "k", "Two")
    holder.current = auth_identity(u1)
    resp = tc.get("/my-reports")
    assert resp.status_code == 200
    body = resp.json()
    assert [r["id"] for r in body] == ["r1"]
    assert body[0]["chartCount"] == 3
    assert body[0]["title"] == "One"
    assert body[0]["kinds"] == ["bar"]


def test_my_reports_requires_auth(ctx):
    tc, _, holder = ctx
    holder.current = anon_identity(str(uuid4()))
    resp = tc.get("/my-reports")
    assert resp.status_code == 401
