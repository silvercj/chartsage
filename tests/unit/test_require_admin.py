import pytest
from fastapi import HTTPException

from deps import require_admin


def test_rejects_when_no_token(monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret-abc")
    with pytest.raises(HTTPException) as e:
        require_admin(None)
    assert e.value.status_code == 403
    assert e.value.detail["code"] == "FORBIDDEN"


def test_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret-abc")
    with pytest.raises(HTTPException) as e:
        require_admin("nope")
    assert e.value.status_code == 403


def test_accepts_correct_token(monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret-abc")
    assert require_admin("secret-abc") is None


def test_fail_closed_when_env_unset(monkeypatch):
    monkeypatch.delenv("ADMIN_API_TOKEN", raising=False)
    with pytest.raises(HTTPException) as e:
        require_admin("anything")
    assert e.value.status_code == 403
