"""Tests the db.py interface contract using FakeDB.

The real SupabaseDB has the same interface; its specific Postgres behavior
is exercised by integration tests and the manual smoke runbook.
"""
import pytest
from uuid import uuid4
from tests.helpers.fake_db import FakeDB


def _sample_report():
    return {
        "generated_at": "2026-05-24T00:00:00",
        "summary": "Sample.",
        "data_quality": [],
        "charts": [],
        "layout": [],
        "metadata": {},
    }


def test_save_and_get_round_trip():
    db = FakeDB()
    anon = uuid4()
    db.save_report("r1", anon, None, _sample_report(), "r1.csv", "Sample")
    row = db.get_report("r1")
    assert row is not None
    assert row["id"] == "r1"
    assert row["anon_id"] == str(anon)
    assert row["user_id"] is None
    assert row["csv_storage_key"] == "r1.csv"
    assert row["title"] == "Sample"
    assert row["report_json"]["summary"] == "Sample."


def test_get_missing_returns_none():
    db = FakeDB()
    assert db.get_report("nope") is None


def test_count_anon_reports():
    db = FakeDB()
    a, b = uuid4(), uuid4()
    db.save_report("r1", a, None, _sample_report(), None, "")
    db.save_report("r2", a, None, _sample_report(), None, "")
    db.save_report("r3", b, None, _sample_report(), None, "")
    assert db.count_anon_reports(a) == 2
    assert db.count_anon_reports(b) == 1
    assert db.count_anon_reports(uuid4()) == 0


def test_count_excludes_reports_with_user_id():
    db = FakeDB()
    a = uuid4()
    user = uuid4()
    db.save_report("r1", a, None, _sample_report(), None, "")
    db.save_report("r2", a, user, _sample_report(), None, "")   # migrated
    assert db.count_anon_reports(a) == 1


def test_update_layout():
    db = FakeDB()
    anon = uuid4()
    db.save_report("r1", anon, None, _sample_report(), None, "")
    new_layout = [{"chart_id": "c1", "position": "main", "order": 0}]
    assert db.update_layout("r1", new_layout) is True
    assert db.get_report("r1")["report_json"]["layout"] == new_layout


def test_update_layout_missing_returns_false():
    db = FakeDB()
    assert db.update_layout("nope", []) is False


def test_update_report_json_overwrites():
    db = FakeDB()
    anon = uuid4()
    db.save_report("r1", anon, None, _sample_report(), None, "")
    new_report = _sample_report()
    new_report["summary"] = "Updated."
    assert db.update_report_json("r1", new_report) is True
    assert db.get_report("r1")["report_json"]["summary"] == "Updated."


def test_round_trip_does_not_share_references():
    db = FakeDB()
    anon = uuid4()
    original = _sample_report()
    db.save_report("r1", anon, None, original, None, "")
    original["summary"] = "MUTATED"   # mutate the input after save
    assert db.get_report("r1")["report_json"]["summary"] == "Sample."   # unchanged
