# tests/integration/test_billing_packages.py
from fastapi.testclient import TestClient


def test_packages_returns_three():
    from main import app
    tc = TestClient(app)
    r = tc.get("/billing/packages")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 3
    assert {d["id"] for d in data} == {"starter", "standard", "pro"}
    std = next(d for d in data if d["id"] == "standard")
    assert std["credits"] == 2000 and std["price_display"] == "$15" and std["label"] == "Standard"
