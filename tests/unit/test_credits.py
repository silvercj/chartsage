import importlib


def test_default_costs():
    import credits
    importlib.reload(credits)
    assert credits.REPORT_COST == 100
    assert credits.GENERATE_MORE_COST == 40
    assert credits.SIGNUP_GRANT == 300


def test_costs_env_override(monkeypatch):
    monkeypatch.setenv("REPORT_COST", "250")
    monkeypatch.setenv("GENERATE_MORE_COST", "80")
    monkeypatch.setenv("SIGNUP_GRANT", "1000")
    import credits
    importlib.reload(credits)
    assert (credits.REPORT_COST, credits.GENERATE_MORE_COST, credits.SIGNUP_GRANT) == (250, 80, 1000)
    monkeypatch.undo()
    importlib.reload(credits)


def test_insufficient_credits_is_exception():
    import credits
    assert issubclass(credits.InsufficientCredits, Exception)


import pytest
from uuid import uuid4
from credits import InsufficientCredits
from tests.helpers.fake_db import FakeDB


def test_ensure_profile_grants_once():
    db = FakeDB(); u = str(uuid4())
    assert db.ensure_profile(u, 300) == 300
    assert db.ensure_profile(u, 300) == 300          # idempotent — no double grant
    assert db.get_balance(u) == 300


def test_spend_decrements_and_logs():
    db = FakeDB(); u = str(uuid4())
    db.ensure_profile(u, 300)
    assert db.spend_credits(u, 100, "report", "r1") == 200
    txns = db.list_transactions(u)
    assert txns[0]["delta"] == -100 and txns[0]["reason"] == "report" and txns[0]["ref"] == "r1"


def test_spend_insufficient_raises():
    db = FakeDB(); u = str(uuid4())
    db.ensure_profile(u, 300)
    with pytest.raises(InsufficientCredits):
        db.spend_credits(u, 9999, "report", "r1")
    assert db.get_balance(u) == 300                  # unchanged


def test_grant_adds():
    db = FakeDB(); u = str(uuid4())
    db.ensure_profile(u, 300)
    assert db.grant_credits(u, 50, "adjustment") == 350


def test_upgrade_intent_recorded():
    db = FakeDB(); u = str(uuid4())
    db.record_upgrade_intent(u, "x@y.com")
    assert db._intent[u] == "x@y.com"
