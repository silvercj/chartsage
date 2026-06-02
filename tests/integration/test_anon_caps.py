from datetime import datetime, timezone, timedelta
from tests.helpers.fake_db import FakeDB


def test_log_and_count_today():
    db = FakeDB()
    db.log_anon_report("11111111-1111-1111-1111-111111111111", "1.2.3.4", "abc123")
    db.log_anon_report("22222222-2222-2222-2222-222222222222", "1.2.3.4", "def456")
    db.log_anon_report("33333333-3333-3333-3333-333333333333", "9.9.9.9", "ghi789")
    assert db.count_anon_reports_today() == 3
    assert db.count_anon_reports_today_by_ip("1.2.3.4") == 2
    assert db.count_anon_reports_today_by_ip("9.9.9.9") == 1


def test_count_today_excludes_old_rows():
    db = FakeDB()
    db.log_anon_report("11111111-1111-1111-1111-111111111111", "1.2.3.4", "abc")
    db._anon_log.append({"anon_id": None, "ip": "1.2.3.4", "fingerprint": "old",
                         "created_at": datetime.now(timezone.utc) - timedelta(days=2)})
    assert db.count_anon_reports_today() == 1
    assert db.count_anon_reports_today_by_ip("1.2.3.4") == 1
