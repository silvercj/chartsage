import io
import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from uuid import uuid4

from tests.helpers.fake_claude import FakeClaude, tool_use
from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_storage import FakeStorage
from tests.helpers.fake_posthog import FakePostHog


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _scripted_fake():
    chart_calls = [
        tool_use("frequency_bar_chart",
                 {"column": "region", "title": f"T{i}", "intent": f"i{i}"},
                 id_=f"tu_{i}")
        for i in range(10)
    ]
    return FakeClaude([
        {"tool_calls": chart_calls},
        {"tool_calls": [tool_use(
            "submit_narrative",
            {"summary": "S.", "captions": [f"c{i}" for i in range(10)], "data_quality": []},
        )]},
    ])


@pytest.fixture
def anon_id():
    return str(uuid4())


@pytest.fixture
def client(sales, anon_id):
    """Boot the app with fakes injected via dependency_overrides."""
    fake_claude = _scripted_fake()
    fake_db = FakeDB()
    fake_storage = FakeStorage()
    fake_posthog = FakePostHog()

    from main import app, get_claude_client, get_db, get_storage, get_posthog
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=fake_claude)
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[get_storage] = lambda: fake_storage
    app.dependency_overrides[get_posthog] = lambda: fake_posthog

    yield TestClient(app), fake_db, fake_storage, fake_posthog
    app.dependency_overrides.clear()


def _headers(anon_id):
    return {"X-Anon-Id": anon_id}


def test_happy_path_post_then_get(client, sales, anon_id):
    tc, _db, _storage, _ph = client
    resp = tc.post("/generate-report",
                   files={"file": ("sales.csv", _csv_bytes(sales), "text/csv")},
                   headers=_headers(anon_id))
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    resp2 = tc.get(f"/report/{session_id}", headers=_headers(anon_id))
    assert resp2.status_code == 200
    body = resp2.json()
    assert body["summary"] == "S."
    assert len(body["charts"]) == 10


def test_rejects_non_csv_xlsx(client, anon_id):
    tc, *_ = client
    resp = tc.post("/generate-report",
                   files={"file": ("data.txt", b"hello", "text/plain")},
                   headers=_headers(anon_id))
    assert resp.status_code == 422


def test_rejects_oversize_file(client, anon_id):
    tc, *_ = client
    big = b"a,b\n" + b"1,2\n" * 5_000_000
    resp = tc.post("/generate-report",
                   files={"file": ("big.csv", big, "text/csv")},
                   headers=_headers(anon_id))
    assert resp.status_code == 422


def test_rejects_corrupt_csv(client, anon_id):
    tc, *_ = client
    resp = tc.post("/generate-report",
                   files={"file": ("bad.csv", b"\x00\x01\x02broken", "text/csv")},
                   headers=_headers(anon_id))
    assert resp.status_code == 422


def test_get_nonexistent_session(client, anon_id):
    tc, *_ = client
    resp = tc.get("/report/does-not-exist", headers=_headers(anon_id))
    assert resp.status_code == 404


def test_missing_anon_id_returns_400(client, sales):
    tc, *_ = client
    resp = tc.post("/generate-report",
                   files={"file": ("sales.csv", _csv_bytes(sales), "text/csv")})
    assert resp.status_code == 400
    assert "MISSING_ANON_ID" in resp.text
