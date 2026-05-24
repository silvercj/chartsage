import io
import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


@pytest.fixture
def client(sales, monkeypatch):
    """Boot the app with a mock claude client and an in-memory redis stand-in."""
    from tests.helpers.fake_claude import FakeClaude, tool_use

    fake = FakeClaude([
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "region", "title": "Regions", "intent": "show mix"}),
            tool_use("aggregation_bar_chart", {
                "value_col": "revenue", "group_col": "region",
                "agg": "sum", "title": "Revenue by region", "intent": "show winners"}),
            tool_use("line_chart", {
                "date_col": "order_date", "value_col": "revenue",
                "agg": "sum", "granularity": "week",
                "title": "Weekly revenue", "intent": "trend"}),
        ]},
        {"tool_calls": [
            tool_use("submit_narrative", {
                "summary": "Sales report.",
                "captions": ["c1", "c2", "c3"],
                "data_quality": [],
            }),
        ]},
    ])

    fake_redis = {}

    class FakeRedis:
        def set(self, key, val, ex=None):
            fake_redis[key] = val
        def get(self, key):
            return fake_redis.get(key)

    from main import app, get_claude_client, get_redis
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=fake)
    app.dependency_overrides[get_redis] = lambda: FakeRedis()

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_happy_path_post_then_get(client, sales):
    resp = client.post(
        "/generate-report",
        files={"file": ("sales.csv", _csv_bytes(sales), "text/csv")},
    )
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    resp2 = client.get(f"/report/{session_id}")
    assert resp2.status_code == 200
    body = resp2.json()
    assert body["summary"] == "Sales report."
    assert len(body["charts"]) == 3


def test_rejects_non_csv_xlsx(client):
    resp = client.post(
        "/generate-report",
        files={"file": ("data.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 422


def test_rejects_oversize_file(client):
    big = b"a,b\n" + b"1,2\n" * 5_000_000   # ~25 MB
    resp = client.post(
        "/generate-report",
        files={"file": ("big.csv", big, "text/csv")},
    )
    assert resp.status_code == 422


def test_rejects_corrupt_csv(client):
    resp = client.post(
        "/generate-report",
        files={"file": ("bad.csv", b"\x00\x01\x02broken", "text/csv")},
    )
    assert resp.status_code == 422


def test_get_nonexistent_session(client):
    resp = client.get("/report/does-not-exist")
    assert resp.status_code == 404
