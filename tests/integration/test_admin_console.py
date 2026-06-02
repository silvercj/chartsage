import os
import pytest
from fastapi.testclient import TestClient

from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_posthog import FakePostHog

ADMIN = "test-admin-token"
UID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def client_and_fakes():
    os.environ["ADMIN_API_TOKEN"] = ADMIN
    db = FakeDB()
    db.add_user(UID, "alice@example.com")
    db.ensure_profile(UID, 300)
    ph = FakePostHog()
    from main import app, get_db, get_posthog
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_posthog] = lambda: ph
    yield TestClient(app), db, ph
    app.dependency_overrides.clear()
    os.environ.pop("ADMIN_API_TOKEN", None)


def _h(tok=ADMIN):
    return {"X-Admin-Token": tok}


def test_search_requires_token(client_and_fakes):
    tc, _, _ = client_and_fakes
    assert tc.get("/admin/accounts").status_code == 403
    assert tc.get("/admin/accounts", headers=_h("wrong")).status_code == 403


def test_search_returns_filtered(client_and_fakes):
    tc, _, _ = client_and_fakes
    r = tc.get("/admin/accounts", params={"q": "alice"}, headers=_h())
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1 and body[0]["email"] == "alice@example.com"
    assert body[0]["credits_balance"] == 300


def test_detail_and_unknown(client_and_fakes):
    tc, _, _ = client_and_fakes
    assert tc.get(f"/admin/accounts/{UID}", headers=_h()).json()["credits_balance"] == 300
    assert tc.get("/admin/accounts/00000000-0000-0000-0000-000000000000", headers=_h()).status_code == 404


def test_grant_happy_path(client_and_fakes):
    tc, db, ph = client_and_fakes
    r = tc.post(f"/admin/accounts/{UID}/grant", json={"amount": 1000}, headers=_h())
    assert r.status_code == 200
    assert r.json()["credits_balance"] == 1300
    assert db.get_balance(UID) == 1300
    granted = ph.find("admin_credit_grant")
    assert len(granted) == 1
    p = granted[0]["properties"]
    assert p["amount"] == 1000 and p["newBalance"] == 1300
    assert p["targetEmail"] == "alice@example.com" and p["source"] == "admin_console"
    for k in p:
        assert "_" not in k or k.startswith("$")
    assert any(t["delta"] == 1000 and t["reason"] == "admin_grant" for t in db.list_transactions(UID))


def test_grant_rejects_token_and_amount_and_unknown(client_and_fakes):
    tc, _, _ = client_and_fakes
    assert tc.post(f"/admin/accounts/{UID}/grant", json={"amount": 1000}).status_code == 403
    assert tc.post(f"/admin/accounts/{UID}/grant", json={"amount": 0}, headers=_h()).status_code == 422
    assert tc.post(f"/admin/accounts/{UID}/grant", json={"amount": -5}, headers=_h()).status_code == 422
    assert tc.post(f"/admin/accounts/{UID}/grant", json={"amount": 999999}, headers=_h()).status_code == 422
    assert tc.post("/admin/accounts/00000000-0000-0000-0000-000000000000/grant",
                   json={"amount": 10}, headers=_h()).status_code == 404
