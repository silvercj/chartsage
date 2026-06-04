"""True-private reports: visibility + ownership enforcement."""
import uuid
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

import main
from deps import Identity
from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_auth import auth_identity, anon_identity
from tests.helpers.fake_storage import FakeStorage
from tests.helpers.fake_posthog import FakePostHog

OWNER = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"
ANON = "33333333-3333-3333-3333-333333333333"
ANON2 = "44444444-4444-4444-4444-444444444444"


def _client(db, identity=None):
    ident = identity if identity is not None else Identity()
    main.app.dependency_overrides[main.get_db] = lambda: db
    main.app.dependency_overrides[main.get_identity] = lambda: ident
    main.app.dependency_overrides[main.get_identity_optional] = lambda: ident
    main.app.dependency_overrides[main.get_storage] = lambda: FakeStorage()
    main.app.dependency_overrides[main.get_posthog] = lambda: FakePostHog()
    main.app.dependency_overrides[main.get_claude_client] = lambda: MagicMock()
    return TestClient(main.app)


def _seed(db, *, user_id=None, anon_id=None, is_public=False):
    rid = uuid.uuid4().hex
    db.save_report(
        rid,
        uuid.UUID(anon_id) if anon_id else None,
        uuid.UUID(user_id) if user_id else None,
        {"summary": "s", "charts": [], "layout": [], "data_quality": [],
         "key_metrics": [], "metadata": {}, "generated_at": "x"},
        None,
        "Secret title",
    )
    if is_public:
        db.set_report_visibility(rid, True)
    return rid


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    main.app.dependency_overrides.clear()


# ---- GET /report visibility -------------------------------------------------

def test_owner_views_private_report():
    db = FakeDB(); rid = _seed(db, user_id=OWNER)
    r = _client(db, auth_identity(OWNER)).get(f"/report/{rid}")
    assert r.status_code == 200 and r.json()["summary"] == "s"


def test_non_owner_private_report_404():
    db = FakeDB(); rid = _seed(db, user_id=OWNER)
    r = _client(db, auth_identity(OTHER)).get(f"/report/{rid}")
    assert r.status_code == 404


def test_anon_owner_views_own_private_report():
    db = FakeDB(); rid = _seed(db, anon_id=ANON)
    r = _client(db, anon_identity(ANON)).get(f"/report/{rid}")
    assert r.status_code == 200


def test_different_anon_cannot_view_private_report():
    db = FakeDB(); rid = _seed(db, anon_id=ANON)
    r = _client(db, anon_identity(ANON2)).get(f"/report/{rid}")
    assert r.status_code == 404


def test_public_report_viewable_by_anyone():
    db = FakeDB(); rid = _seed(db, user_id=OWNER, is_public=True)
    r = _client(db, None).get(f"/report/{rid}")   # no identity at all
    assert r.status_code == 200


def test_missing_report_404():
    db = FakeDB()
    r = _client(db, auth_identity(OWNER)).get(f"/report/{uuid.uuid4().hex}")
    assert r.status_code == 404


# ---- Task 2: anon-aware publish + /meta privacy -----------------------------

def test_anon_owner_can_publish(monkeypatch):
    async def _noop_og(_sid):
        return b""
    monkeypatch.setattr("pdf_export.render_og_image", _noop_og)
    db = FakeDB(); rid = _seed(db, anon_id=ANON)
    r = _client(db, anon_identity(ANON)).post(f"/report/{rid}/publish")
    assert r.status_code == 200
    assert db.get_report(rid)["is_public"] is True


def test_non_owner_cannot_publish_private_404():
    db = FakeDB(); rid = _seed(db, user_id=OWNER)
    r = _client(db, auth_identity(OTHER)).post(f"/report/{rid}/publish")
    assert r.status_code == 404


def test_meta_hides_title_of_private_report_from_non_owner():
    db = FakeDB(); rid = _seed(db, user_id=OWNER)
    r = _client(db, None).get(f"/report/{rid}/meta")
    assert r.status_code == 200
    body = r.json()
    assert body["is_public"] is False
    assert body["owned"] is False
    assert body["title"] == "Private report"   # real title not leaked to a non-owner


def test_meta_owner_sees_full_for_private():
    db = FakeDB(); rid = _seed(db, user_id=OWNER)
    body = _client(db, auth_identity(OWNER)).get(f"/report/{rid}/meta").json()
    assert body["owned"] is True
    assert body["title"] != "Private report"   # owner gets the real meta


# ---- Task 3: mutations require ownership ------------------------------------

@pytest.mark.parametrize("path,method,body", [
    ("/layout", "patch", []),
    ("/generate-more", "post", None),
    ("/add-chart", "post", {"mode": "describe", "prompt": "x"}),
    ("/deepen", "post", None),
])
def test_mutations_blocked_for_non_owner(path, method, body):
    db = FakeDB(); rid = _seed(db, user_id=OWNER)   # private, owned by OWNER
    client = _client(db, auth_identity(OTHER))
    kw = {} if body is None else {"json": body}
    r = getattr(client, method)(f"/report/{rid}{path}", **kw)
    assert r.status_code == 404


# ---- Task 4: exports enforce visibility ------------------------------------

@pytest.mark.parametrize("ext", ["pdf", "pptx", "xlsx", "zip", "md", "html"])
def test_export_private_blocked_for_non_owner(ext):
    db = FakeDB(); rid = _seed(db, user_id=OWNER)   # private, owned by OWNER
    r = _client(db, auth_identity(OTHER)).get(f"/report/{rid}/export.{ext}")
    assert r.status_code == 404


def test_export_md_owner_ok(monkeypatch):
    async def _noop_imgs(_sid):
        return []
    monkeypatch.setattr("pdf_export.render_chart_images", _noop_imgs)
    db = FakeDB(); rid = _seed(db, user_id=OWNER)
    r = _client(db, auth_identity(OWNER)).get(f"/report/{rid}/export.md")
    assert r.status_code == 200
