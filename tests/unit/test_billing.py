# tests/unit/test_billing.py
from billing import get_package, public_catalogue, price_id_for


def test_get_package_known():
    p = get_package("standard")
    assert p is not None
    assert p["credits"] == 2000 and p["label"] == "Standard"


def test_get_package_unknown():
    assert get_package("enterprise") is None


def test_public_catalogue_shape():
    cat = public_catalogue()
    assert len(cat) == 3
    assert {c["id"] for c in cat} == {"starter", "standard", "pro"}
    for c in cat:
        assert set(c.keys()) == {"id", "label", "credits", "gbp"}
    starter = next(c for c in cat if c["id"] == "starter")
    assert starter["credits"] == 600 and starter["gbp"] == 5


def test_price_id_for_reads_env(monkeypatch):
    monkeypatch.setenv("STRIPE_PRICE_PRO", "price_live_pro_123")
    assert price_id_for(get_package("pro")) == "price_live_pro_123"


def test_price_id_for_unset_is_none(monkeypatch):
    monkeypatch.delenv("STRIPE_PRICE_PRO", raising=False)
    assert price_id_for(get_package("pro")) is None
