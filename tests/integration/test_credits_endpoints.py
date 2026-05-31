import pytest
from fastapi.testclient import TestClient
from uuid import uuid4

from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_posthog import FakePostHog
from tests.helpers.fake_auth import auth_identity, anon_identity


class _Holder:
    def __init__(self): self.current = None
    def __call__(self): return self.current


@pytest.fixture
def ctx():
    fake_db, fake_posthog = FakeDB(), FakePostHog()
    holder = _Holder()
    from main import app, get_db, get_posthog, get_identity
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[get_posthog] = lambda: fake_posthog
    app.dependency_overrides[get_identity] = holder
    yield TestClient(app), fake_db, fake_posthog, holder
    app.dependency_overrides.clear()


def test_me_grants_starter_once(ctx):
    tc, db, ph, holder = ctx
    user = str(uuid4()); holder.current = auth_identity(user)
    assert tc.get("/me").json()["credits_balance"] == 300
    assert tc.get("/me").json()["credits_balance"] == 300     # no double grant
    grants = [t for t in db._txns if t["reason"] == "signup_grant"]
    assert len(grants) == 1
    assert len(ph.find("credits_granted")) == 1               # fired once, on first /me


def test_me_requires_auth(ctx):
    tc, _, _, holder = ctx
    holder.current = anon_identity(str(uuid4()))
    assert tc.get("/me").status_code == 401


def test_credits_history(ctx):
    tc, db, _, holder = ctx
    user = str(uuid4()); holder.current = auth_identity(user)
    db.ensure_profile(user, 300); db.spend_credits(user, 100, "report", "r1")
    rows = tc.get("/credits/history").json()
    assert rows[0]["delta"] == -100 and rows[0]["reason"] == "report"
    assert any(r["reason"] == "signup_grant" for r in rows)


def test_upgrade_intent(ctx):
    tc, db, ph, holder = ctx
    user = str(uuid4()); holder.current = auth_identity(user)
    resp = tc.post("/upgrade-intent", json={"email": "x@y.com"})
    assert resp.status_code == 200
    assert db._intent[user] == "x@y.com"
    assert len(ph.find("upgrade_intent_captured")) == 1


def test_upgrade_intent_requires_auth(ctx):
    tc, _, _, holder = ctx
    holder.current = anon_identity(str(uuid4()))
    assert tc.post("/upgrade-intent", json={"email": "x@y.com"}).status_code == 401
