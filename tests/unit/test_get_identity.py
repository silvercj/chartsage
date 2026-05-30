from uuid import uuid4

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from deps import Identity, get_identity


def _client():
    app = FastAPI()

    @app.get("/whoami")
    def whoami(identity: Identity = Depends(get_identity)):
        return {
            "user_id": str(identity.user_id) if identity.user_id else None,
            "anon_id": str(identity.anon_id) if identity.anon_id else None,
            "auth": identity.is_authenticated,
        }

    return TestClient(app)


def test_anon_only_is_unauthenticated():
    anon = str(uuid4())
    res = _client().get("/whoami", headers={"X-Anon-Id": anon})
    assert res.status_code == 200
    assert res.json() == {"user_id": None, "anon_id": anon, "auth": False}


def test_no_headers_returns_400_missing_anon():
    res = _client().get("/whoami")
    assert res.status_code == 400
    assert "MISSING_ANON_ID" in res.text


def test_malformed_anon_returns_400_invalid_anon():
    res = _client().get("/whoami", headers={"X-Anon-Id": "not-a-uuid"})
    assert res.status_code == 400
    assert "INVALID_ANON_ID" in res.text


def test_valid_bearer_is_authenticated(monkeypatch):
    uid = uuid4()
    monkeypatch.setattr("deps.verify_token", lambda t: uid)
    res = _client().get("/whoami", headers={"Authorization": "Bearer good"})
    assert res.status_code == 200
    body = res.json()
    assert body["user_id"] == str(uid)
    assert body["auth"] is True


def test_invalid_bearer_returns_401_no_downgrade(monkeypatch):
    # A present-but-invalid token is a hard 401 even if X-Anon-Id is also sent.
    monkeypatch.setattr("deps.verify_token", lambda t: None)
    res = _client().get(
        "/whoami",
        headers={"Authorization": "Bearer bad", "X-Anon-Id": str(uuid4())},
    )
    assert res.status_code == 401
    assert "INVALID_TOKEN" in res.text
