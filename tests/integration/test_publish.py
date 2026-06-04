import pytest
from uuid import uuid4
from fastapi.testclient import TestClient
from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_posthog import FakePostHog
from tests.helpers.fake_storage import FakeStorage
from tests.helpers.fake_auth import auth_identity, anon_identity


class _Holder:
    def __init__(self): self.current = None
    def __call__(self): return self.current


@pytest.fixture
def ctx(monkeypatch):
    monkeypatch.setenv("FRONTEND_BASE_URL", "https://chartsage.app")
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    # Default OG renderer to a fast no-op so publish tests never launch real Chromium.
    # OG-specific tests re-patch render_og_image after this fixture runs.
    import pdf_export
    async def _fake_render(session_id):
        return b"\x89PNG"
    monkeypatch.setattr(pdf_export, "render_og_image", _fake_render)
    db, ph = FakeDB(), FakePostHog()
    holder = _Holder()
    from main import app, get_db, get_posthog, get_storage, get_identity, get_identity_optional
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_posthog] = lambda: ph
    app.dependency_overrides[get_storage] = lambda: FakeStorage()
    app.dependency_overrides[get_identity] = holder
    app.dependency_overrides[get_identity_optional] = holder
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


def test_publish_non_owner_404(ctx):
    # True-private: a private report is hidden from non-owners, so managing it 404s
    # (was 403 NOT_OWNER before true-private reports).
    tc, db, ph, holder = ctx
    rid = _save(db, str(uuid4())); holder.current = auth_identity(str(uuid4()))
    r = tc.post(f"/report/{rid}/publish")
    assert r.status_code == 404 and r.json()["detail"]["code"] == "NOT_FOUND"
    assert db.get_report(rid)["is_public"] is False


def test_publish_anon_non_owner_404(ctx):
    # A non-owner anon can't publish someone else's private report (404 hides it).
    # Anon OWNERS publishing their OWN report is allowed now — covered in test_report_privacy.py.
    tc, db, ph, holder = ctx
    rid = _save(db, str(uuid4())); holder.current = anon_identity(str(uuid4()))
    assert tc.post(f"/report/{rid}/publish").status_code == 404


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


def test_meta_owner_vs_anon(ctx):
    tc, db, ph, holder = ctx
    user = str(uuid4()); rid = _save(db, user)
    holder.current = auth_identity(user)
    m = tc.get(f"/report/{rid}/meta").json()
    assert m["is_public"] is False and m["owned"] is True and m["title"] == "Sales"
    holder.current = anon_identity(str(uuid4()))
    assert tc.get(f"/report/{rid}/meta").json()["owned"] is False


def test_meta_works_without_auth_headers(ctx):
    tc, db, ph, holder = ctx
    rid = _save(db, str(uuid4()))
    from main import app, get_identity, get_identity_optional
    app.dependency_overrides.pop(get_identity, None)
    app.dependency_overrides.pop(get_identity_optional, None)
    r = tc.get(f"/report/{rid}/meta")
    assert r.status_code == 200 and r.json()["owned"] is False


def test_reports_public_lists_only_public(ctx):
    tc, db, ph, holder = ctx
    pub = _save(db, str(uuid4())); priv = _save(db, str(uuid4()))
    db.set_report_visibility(pub, True)
    ids = {r["id"] for r in tc.get("/reports/public").json()}
    assert pub in ids and priv not in ids


def test_publish_generates_og(ctx, monkeypatch):
    tc, db, ph, holder = ctx
    user = str(uuid4()); rid = _save(db, user); holder.current = auth_identity(user)
    calls = {}
    async def fake_render(session_id):
        calls["rendered"] = session_id
        return b"\x89PNG-bytes"
    import pdf_export
    monkeypatch.setattr(pdf_export, "render_og_image", fake_render)
    from main import app, get_storage
    class FakeStorage:
        def upload_public_image(self, key, png): calls["key"] = key; return key
    app.dependency_overrides[get_storage] = lambda: FakeStorage()
    r = tc.post(f"/report/{rid}/publish")
    assert r.status_code == 200
    assert calls["rendered"] == rid and calls["key"] == f"{rid}.png"
    assert db.get_report(rid)["og_image_key"] == f"{rid}.png"


def test_publish_survives_og_failure(ctx, monkeypatch):
    tc, db, ph, holder = ctx
    user = str(uuid4()); rid = _save(db, user); holder.current = auth_identity(user)
    import pdf_export
    async def boom(session_id): raise RuntimeError("render failed")
    monkeypatch.setattr(pdf_export, "render_og_image", boom)
    r = tc.post(f"/report/{rid}/publish")
    assert r.status_code == 200 and db.get_report(rid)["is_public"] is True
