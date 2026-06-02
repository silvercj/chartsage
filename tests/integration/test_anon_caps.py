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


import io
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from uuid import uuid4

import main as main_module
from tests.helpers.fake_claude import FakeClaude, tool_use
from tests.helpers.fake_storage import FakeStorage
from tests.helpers.fake_posthog import FakePostHog


def _csv(df):
    b = io.StringIO(); df.to_csv(b, index=False); return b.getvalue().encode("utf-8")


@pytest.fixture
def caps_client(sales):
    calls = [tool_use("frequency_bar_chart", {"column": "region", "title": "T", "intent": "i"}, id_="t0")]
    fake_claude = FakeClaude([
        {"tool_calls": calls},
        {"tool_calls": [tool_use("submit_narrative", {"summary": "S.", "captions": ["c"], "data_quality": []})]},
    ])
    db = FakeDB(); storage = FakeStorage(); ph = FakePostHog()
    from main import app, get_claude_client, get_db, get_storage, get_posthog
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=fake_claude)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_posthog] = lambda: ph
    yield TestClient(app), db, ph, fake_claude
    app.dependency_overrides.clear()


def _post(tc, anon, ip="5.5.5.5"):
    return tc.post("/generate-report",
                   files={"file": ("s.csv", _csv(pd.DataFrame({"region": ["N", "S"], "rev": [1, 2]})), "text/csv")},
                   headers={"X-Anon-Id": anon, "X-Forwarded-For": ip})


def test_per_ip_cap_blocks_with_429(caps_client, monkeypatch, sales):
    tc, db, ph, claude = caps_client
    monkeypatch.setattr(main_module, "ANON_IP_DAILY_CAP", 1)
    db.log_anon_report(str(uuid4()), "5.5.5.5", "x")
    r = _post(tc, str(uuid4()), ip="5.5.5.5")
    assert r.status_code == 429 and r.json()["detail"]["code"] == "RATE_LIMITED"
    assert len(ph.find("anon_cap_hit")) == 1 and ph.find("anon_cap_hit")[0]["properties"]["scope"] == "ip"


def test_global_cap_blocks_with_503(caps_client, monkeypatch, sales):
    tc, db, ph, claude = caps_client
    monkeypatch.setattr(main_module, "ANON_IP_DAILY_CAP", 100)
    monkeypatch.setattr(main_module, "ANON_GLOBAL_DAILY_CAP", 1)
    db.log_anon_report(str(uuid4()), "9.9.9.9", "x")
    r = _post(tc, str(uuid4()), ip="5.5.5.5")
    assert r.status_code == 503 and r.json()["detail"]["code"] == "FREE_TIER_AT_CAPACITY"
    assert ph.find("anon_cap_hit")[0]["properties"]["scope"] == "global"


def test_anon_success_logs_row(caps_client, sales):
    tc, db, ph, claude = caps_client
    r = _post(tc, str(uuid4()), ip="5.5.5.5")
    assert r.status_code == 200
    assert len(db._anon_log) == 1 and db._anon_log[0]["ip"] == "5.5.5.5" and db._anon_log[0]["fingerprint"]


def test_authed_unaffected_by_new_caps(caps_client, monkeypatch, sales):
    tc, db, ph, claude = caps_client
    monkeypatch.setattr(main_module, "ANON_GLOBAL_DAILY_CAP", 0)
    monkeypatch.setattr(main_module, "ANON_IP_DAILY_CAP", 0)
    from deps import get_identity, Identity
    uid = uuid4()
    main_module.app.dependency_overrides[get_identity] = lambda: Identity(user_id=uid)
    db.ensure_profile(uid, 300)
    r = tc.post("/generate-report",
                files={"file": ("s.csv", _csv(pd.DataFrame({"region": ["N", "S"], "rev": [1, 2]})), "text/csv")})
    assert r.status_code == 200
    main_module.app.dependency_overrides.pop(get_identity, None)
