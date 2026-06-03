import pytest
from uuid import uuid4
from fastapi.testclient import TestClient
from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_posthog import FakePostHog
from tests.helpers.fake_auth import auth_identity, anon_identity


class _Holder:
    def __init__(self): self.current = None
    def __call__(self): return self.current


@pytest.fixture
def ctx(monkeypatch):
    monkeypatch.setenv("FRONTEND_BASE_URL", "https://chartsage.app")
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    db, ph = FakeDB(), FakePostHog()
    holder = _Holder()
    from main import app, get_db, get_posthog, get_identity
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_posthog] = lambda: ph
    app.dependency_overrides[get_identity] = holder
    yield TestClient(app), db, ph, holder
    app.dependency_overrides.clear()


def _save(db, user_id):
    rid = str(uuid4())
    db.save_report(rid, anon_id=None, user_id=user_id, report_json={"title": "Sales", "charts": []},
                   csv_storage_key=None, title="Sales")
    return rid


def test_publish_owner_ok(ctx):
    tc, db, ph, holder = ctx
    user = str(uuid4()); rid = _save(db, user); holder.current = auth_identity(user)
    r = tc.post(f"/report/{rid}/publish")
    assert r.status_code == 200
    body = r.json()
    assert body["public_url"].endswith(f"/report/{rid}")
    assert body["embed_url"].endswith(f"/report/{rid}/embed")
    assert db.get_report(rid)["is_public"] is True
    assert len(ph.find("report_published")) == 1


def test_publish_non_owner_403(ctx):
    tc, db, ph, holder = ctx
    rid = _save(db, str(uuid4())); holder.current = auth_identity(str(uuid4()))
    r = tc.post(f"/report/{rid}/publish")
    assert r.status_code == 403 and r.json()["detail"]["code"] == "NOT_OWNER"
    assert db.get_report(rid)["is_public"] is False


def test_publish_anon_401(ctx):
    tc, db, ph, holder = ctx
    rid = _save(db, str(uuid4())); holder.current = anon_identity(str(uuid4()))
    assert tc.post(f"/report/{rid}/publish").status_code == 401


def test_publish_missing_404(ctx):
    tc, db, ph, holder = ctx
    holder.current = auth_identity(str(uuid4()))
    assert tc.post(f"/report/{uuid4()}/publish").status_code == 404


def test_unpublish_owner_flips(ctx):
    tc, db, ph, holder = ctx
    user = str(uuid4()); rid = _save(db, user); holder.current = auth_identity(user)
    db.set_report_visibility(rid, True)
    r = tc.post(f"/report/{rid}/unpublish")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert db.get_report(rid)["is_public"] is False
    assert len(ph.find("report_unpublished")) == 1
