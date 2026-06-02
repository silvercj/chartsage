# tests/unit/test_fake_db_stripe.py
from uuid import uuid4
from tests.helpers.fake_db import FakeDB


def test_process_purchase_grants_and_records():
    db = FakeDB()
    uid = str(uuid4())
    r = db.process_stripe_purchase("evt_1", uid, 2000, "cs_1")
    assert r == {"granted": True, "balance": 2000}
    assert db.get_balance(uid) == 2000
    assert any(t["reason"] == "stripe_purchase" and t["ref"] == "cs_1" for t in db._txns)


def test_process_purchase_is_idempotent():
    db = FakeDB()
    uid = str(uuid4())
    db.process_stripe_purchase("evt_1", uid, 2000, "cs_1")
    r2 = db.process_stripe_purchase("evt_1", uid, 2000, "cs_1")   # replay of same event
    assert r2 == {"granted": False, "balance": 2000}
    assert db.get_balance(uid) == 2000   # NOT 4000
