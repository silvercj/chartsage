from tests.helpers.fake_db import FakeDB


def _seed():
    db = FakeDB()
    db.add_user("11111111-1111-1111-1111-111111111111", "alice@example.com")
    db.add_user("22222222-2222-2222-2222-222222222222", "bob@test.io")
    db.ensure_profile("11111111-1111-1111-1111-111111111111", 300)
    return db


def test_search_filters_by_email_substring():
    db = _seed()
    res = db.search_accounts("alice", 50)
    assert len(res) == 1
    assert res[0]["email"] == "alice@example.com"
    assert res[0]["credits_balance"] == 300
    assert res[0]["user_id"] == "11111111-1111-1111-1111-111111111111"


def test_search_empty_query_returns_all_capped():
    db = _seed()
    assert len(db.search_accounts("", 50)) == 2
    assert len(db.search_accounts("", 1)) == 1


def test_account_detail_has_balance_and_txns():
    db = _seed()
    d = db.get_account_detail("11111111-1111-1111-1111-111111111111")
    assert d["credits_balance"] == 300
    assert d["email"] == "alice@example.com"
    assert any(t["reason"] == "signup_grant" for t in d["transactions"])


def test_account_detail_unknown_user_is_none():
    db = _seed()
    assert db.get_account_detail("99999999-9999-9999-9999-999999999999") is None
