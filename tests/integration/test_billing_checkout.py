# tests/integration/test_billing_checkout.py
import types
import pytest
from uuid import uuid4
from fastapi.testclient import TestClient

from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_posthog import FakePostHog
from tests.helpers.fake_auth import auth_identity, anon_identity


class _Holder:
    def __init__(self): self.current = None
    def __call__(self): return self.current


@pytest.fixture
def ctx(monkeypatch):
    monkeypatch.setenv("STRIPE_PRICE_STANDARD", "price_test_std")
    db, ph = FakeDB(), FakePostHog()
    holder = _Holder()
    from main import app, get_db, get_posthog, get_identity
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_posthog] = lambda: ph
    app.dependency_overrides[get_identity] = holder
    yield TestClient(app), db, ph, holder
    app.dependency_overrides.clear()


def test_checkout_requires_auth(ctx):
    tc, _, _, holder = ctx
    holder.current = anon_identity(str(uuid4()))
    r = tc.post("/billing/checkout", json={"package_id": "standard"})
    assert r.status_code == 401 and r.json()["detail"]["code"] == "AUTH_REQUIRED"


def test_checkout_unknown_package(ctx):
    tc, _, _, holder = ctx
    holder.current = auth_identity(str(uuid4()))
    r = tc.post("/billing/checkout", json={"package_id": "enterprise"})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "UNKNOWN_PACKAGE"


def test_checkout_unconfigured_price_503(ctx, monkeypatch):
    tc, _, _, holder = ctx
    holder.current = auth_identity(str(uuid4()))
    monkeypatch.delenv("STRIPE_PRICE_STANDARD", raising=False)   # no price configured
    r = tc.post("/billing/checkout", json={"package_id": "standard"})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "BILLING_UNAVAILABLE"


def test_checkout_creates_session(ctx, monkeypatch):
    tc, db, ph, holder = ctx
    user = str(uuid4())
    holder.current = auth_identity(user)
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(id="cs_test_1", url="https://checkout.stripe.test/cs_test_1")

    import stripe
    monkeypatch.setattr(stripe.checkout.Session, "create", staticmethod(fake_create))

    r = tc.post("/billing/checkout", json={"package_id": "standard"})
    assert r.status_code == 200
    assert r.json()["url"] == "https://checkout.stripe.test/cs_test_1"
    assert captured["line_items"][0]["price"] == "price_test_std"   # from env, server-side
    assert captured["mode"] == "payment"
    assert captured["client_reference_id"] == user
    assert captured["metadata"]["credits"] == "2000"               # server-resolved, string
    assert captured["metadata"]["user_id"] == user
    assert captured["metadata"]["package_id"] == "standard"
    assert "session_id={CHECKOUT_SESSION_ID}" in captured["success_url"]
    ev = ph.find("checkout_started")
    assert len(ev) == 1 and ev[0]["properties"]["credits"] == 2000
