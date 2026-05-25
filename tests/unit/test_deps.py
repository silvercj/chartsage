import pytest
from uuid import UUID
from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends


def _make_app(get_anon_id):
    app = FastAPI()

    @app.get("/echo")
    def echo(anon_id: UUID = Depends(get_anon_id)):
        return {"anon_id": str(anon_id)}

    return TestClient(app)


def test_valid_uuid_header_passes():
    from deps import get_anon_id
    client = _make_app(get_anon_id)
    valid = "11111111-1111-1111-1111-111111111111"
    res = client.get("/echo", headers={"X-Anon-Id": valid})
    assert res.status_code == 200
    assert res.json() == {"anon_id": valid}


def test_missing_header_returns_400():
    from deps import get_anon_id
    client = _make_app(get_anon_id)
    res = client.get("/echo")
    assert res.status_code == 400
    assert "MISSING_ANON_ID" in res.text


def test_malformed_uuid_returns_400():
    from deps import get_anon_id
    client = _make_app(get_anon_id)
    res = client.get("/echo", headers={"X-Anon-Id": "not-a-uuid"})
    assert res.status_code == 400
    assert "INVALID_ANON_ID" in res.text


def test_empty_header_returns_400():
    from deps import get_anon_id
    client = _make_app(get_anon_id)
    res = client.get("/echo", headers={"X-Anon-Id": ""})
    assert res.status_code == 400
