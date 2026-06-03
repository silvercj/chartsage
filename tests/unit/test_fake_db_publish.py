from uuid import uuid4
from tests.helpers.fake_db import FakeDB


def _save(db, user_id=None):
    rid = str(uuid4())
    db.save_report(rid, anon_id=None, user_id=user_id, report_json={"title": "T", "charts": []},
                   csv_storage_key=None, title="T")
    return rid


def test_set_visibility_and_list_public():
    db = FakeDB()
    r1 = _save(db); r2 = _save(db)
    assert db.get_report(r1)["is_public"] is False          # default
    assert db.set_report_visibility(r1, True, og_image_key="r1.png") is True
    assert db.get_report(r1)["is_public"] is True
    assert db.get_report(r1)["og_image_key"] == "r1.png"
    ids = {r["id"] for r in db.list_public_reports()}
    assert r1 in ids and r2 not in ids
    assert db.set_report_visibility(r1, False) is True
    assert db.get_report(r1)["is_public"] is False
    assert db.set_report_visibility("missing", True) is False
