# tests/integration/test_billing_webhook.py
import pytest
from uuid import uuid4
from fastapi.testclient import TestClient

from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_posthog import FakePostHog


@pytest.fixture
def ctx():
    db, ph = FakeDB(), FakePostHog()
    from main import app, get_db, get_posthog
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_posthog] = lambda: ph
    yield TestClient(app), db, ph
    app.dependency_overrides.clear()


def _completed_event(event_id, user_id, credits=2000, session_id="cs_1", paid=True):
    return {
        "id": event_id,
        "type": "checkout.session.completed",
        "data": {"object": {
            "id": session_id,
            "payment_status": "paid" if paid else "unpaid",
            "amount_total": 1500, "currency": "gbp",
            "metadata": {"user_id": user_id, "credits": str(credits), "package_id": "standard"},
        }},
    }


def test_webhook_bad_signature_400(ctx, monkeypatch):
    tc, db, ph = ctx
    from main import SignatureVerificationError

    def boom(payload, sig, secret):
        raise SignatureVerificationError("bad signature", "sig")

    monkeypatch.setattr("stripe.Webhook.construct_event", boom)
    r = tc.post("/billing/webhook", content=b"{}", headers={"stripe-signature": "x"})
    assert r.status_code == 400 and r.json()["detail"]["code"] == "INVALID_SIGNATURE"


def test_webhook_paid_grants_and_tracks(ctx, monkeypatch):
    tc, db, ph = ctx
    user = str(uuid4())
    ev = _completed_event("evt_1", user)
    monkeypatch.setattr("stripe.Webhook.construct_event", lambda payload, sig, secret: ev)
    r = tc.post("/billing/webhook", content=b"{}", headers={"stripe-signature": "x"})
    assert r.status_code == 200 and r.json()["received"] is True
    assert db.get_balance(user) == 2000
    purchased = ph.find("credits_purchased")
    assert len(purchased) == 1 and purchased[0]["properties"]["credits"] == 2000


def test_webhook_replay_no_double_grant(ctx, monkeypatch):
    tc, db, ph = ctx
    user = str(uuid4())
    ev = _completed_event("evt_1", user)
    monkeypatch.setattr("stripe.Webhook.construct_event", lambda payload, sig, secret: ev)
    tc.post("/billing/webhook", content=b"{}", headers={"stripe-signature": "x"})
    tc.post("/billing/webhook", content=b"{}", headers={"stripe-signature": "x"})   # replay
    assert db.get_balance(user) == 2000                 # NOT 4000
    assert len(ph.find("credits_purchased")) == 1       # fired exactly once


def test_webhook_ignores_other_event_types(ctx, monkeypatch):
    tc, db, ph = ctx
    ev = {"id": "evt_2", "type": "payment_intent.created", "data": {"object": {}}}
    monkeypatch.setattr("stripe.Webhook.construct_event", lambda payload, sig, secret: ev)
    r = tc.post("/billing/webhook", content=b"{}", headers={"stripe-signature": "x"})
    assert r.status_code == 200 and r.json()["received"] is True
    assert len(ph.find("credits_purchased")) == 0


def test_webhook_unpaid_no_grant(ctx, monkeypatch):
    tc, db, ph = ctx
    user = str(uuid4())
    ev = _completed_event("evt_unpaid", user, paid=False)
    monkeypatch.setattr("stripe.Webhook.construct_event", lambda payload, sig, secret: ev)
    r = tc.post("/billing/webhook", content=b"{}", headers={"stripe-signature": "x"})
    assert r.status_code == 200 and r.json()["received"] is True
    assert db.get_balance(user) == 0                     # unpaid -> never credited
    assert len(ph.find("credits_purchased")) == 0


def test_webhook_malformed_credits_no_grant_no_500(ctx, monkeypatch):
    tc, db, ph = ctx
    user = str(uuid4())
    ev = _completed_event("evt_badcredits", user)
    ev["data"]["object"]["metadata"]["credits"] = "not-a-number"
    monkeypatch.setattr("stripe.Webhook.construct_event", lambda payload, sig, secret: ev)
    r = tc.post("/billing/webhook", content=b"{}", headers={"stripe-signature": "x"})
    assert r.status_code == 200 and r.json()["received"] is True   # not a 500
    assert db.get_balance(user) == 0
    assert len(ph.find("credits_purchased")) == 0


def test_webhook_malformed_user_id_no_grant_no_500(ctx, monkeypatch):
    tc, db, ph = ctx
    ev = _completed_event("evt_baduser", "not-a-uuid")
    monkeypatch.setattr("stripe.Webhook.construct_event", lambda payload, sig, secret: ev)
    r = tc.post("/billing/webhook", content=b"{}", headers={"stripe-signature": "x"})
    assert r.status_code == 200 and r.json()["received"] is True   # not a 500
    assert len(ph.find("credits_purchased")) == 0
