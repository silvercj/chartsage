import pytest
from fastapi.testclient import TestClient
from uuid import uuid4

from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_posthog import FakePostHog


@pytest.fixture
def client_and_fakes():
    db = FakeDB(); ph = FakePostHog()
    from main import app, get_db, get_posthog
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_posthog] = lambda: ph
    yield TestClient(app), db, ph
    app.dependency_overrides.clear()


def _h():
    return {"X-Anon-Id": str(uuid4())}


def test_valid_message_stored_and_tracked(client_and_fakes):
    tc, db, ph = client_and_fakes
    r = tc.post("/contact", json={"email": "a@b.com", "message": "Help please"}, headers=_h())
    assert r.status_code == 200 and r.json()["ok"] is True
    assert len(db._support_messages) == 1
    assert db._support_messages[0]["message"] == "Help please"
    assert db._support_messages[0]["anon_id"] is not None
    ev = ph.find("support_request")
    assert len(ev) == 1 and ev[0]["properties"]["hasEmail"] is True and ev[0]["properties"]["length"] == 11


def test_honeypot_silently_dropped(client_and_fakes):
    tc, db, ph = client_and_fakes
    r = tc.post("/contact", json={"message": "spam", "company": "Acme Bots"}, headers=_h())
    assert r.status_code == 200 and r.json()["ok"] is True
    assert len(db._support_messages) == 0
    assert len(ph.find("support_request")) == 0


def test_empty_message_422(client_and_fakes):
    tc, db, ph = client_and_fakes
    r = tc.post("/contact", json={"message": "   "}, headers=_h())
    assert r.status_code == 422 and r.json()["detail"]["code"] == "INVALID_MESSAGE"
    assert len(db._support_messages) == 0


def test_oversize_message_422(client_and_fakes):
    tc, db, ph = client_and_fakes
    r = tc.post("/contact", json={"message": "x" * 4001}, headers=_h())
    assert r.status_code == 422 and r.json()["detail"]["code"] == "INVALID_MESSAGE"
