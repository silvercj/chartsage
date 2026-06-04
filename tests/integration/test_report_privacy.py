"""True-private reports: visibility + ownership enforcement."""
import uuid
import pytest
from fastapi.testclient import TestClient

import main
from deps import Identity
from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_auth import auth_identity, anon_identity

OWNER = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"
ANON = "33333333-3333-3333-3333-333333333333"
ANON2 = "44444444-4444-4444-4444-444444444444"


def _client(db, identity=None):
    ident = identity if identity is not None else Identity()
    main.app.dependency_overrides[main.get_db] = lambda: db
    main.app.dependency_overrides[main.get_identity] = lambda: ident
    main.app.dependency_overrides[main.get_identity_optional] = lambda: ident
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
