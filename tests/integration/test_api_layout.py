import io
import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from tests.helpers.fake_claude import FakeClaude, tool_use


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


@pytest.fixture
def client_with_report(sales):
    """Boot the app, create a real report in fake-redis, return (client, session_id, report)."""
    chart_calls = []
    for i in range(10):
        chart_calls.append(tool_use(
            "frequency_bar_chart",
            {"column": "region", "title": f"Chart {i}", "intent": f"intent {i}"},
            id_=f"tu_{i}",
        ))
    fake = FakeClaude([
        {"tool_calls": chart_calls},
        {"tool_calls": [tool_use(
            "submit_narrative",
            {"summary": "Sales.", "captions": [f"cap{i}" for i in range(10)], "data_quality": []},
        )]},
    ])
    fake_redis_data: dict[str, str] = {}

    class FakeRedis:
        def set(self, key, val, ex=None): fake_redis_data[key] = val
        def get(self, key): return fake_redis_data.get(key)

    from main import app, get_claude_client, get_redis
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=fake)
    app.dependency_overrides[get_redis] = lambda: FakeRedis()

    client = TestClient(app)

    resp = client.post(
        "/generate-report",
        files={"file": ("sales.csv", _csv_bytes(sales), "text/csv")},
    )
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    report = json.loads(fake_redis_data[f"report:{session_id}"])

    yield client, session_id, report
    app.dependency_overrides.clear()


def test_patch_layout_valid_returns_204(client_with_report):
    client, session_id, report = client_with_report
    # Swap positions of chart 0 and chart 1
    layout = report["layout"]
    layout[0], layout[1] = layout[1], layout[0]
    layout[0]["order"], layout[1]["order"] = 0, 1
    resp = client.patch(f"/report/{session_id}/layout", json=layout)
    assert resp.status_code == 204


def test_patch_layout_unknown_chart_id_returns_400(client_with_report):
    client, session_id, report = client_with_report
    bad_layout = [{"chart_id": "does-not-exist", "position": "main", "order": 0}]
    resp = client.patch(f"/report/{session_id}/layout", json=bad_layout)
    assert resp.status_code == 400
    assert "does-not-exist" in resp.text or "chart_id" in resp.text.lower()


def test_patch_layout_unknown_session_returns_404(client_with_report):
    client, _, _ = client_with_report
    resp = client.patch("/report/no-such-session/layout", json=[])
    assert resp.status_code == 404


def test_patch_layout_persists_to_redis(client_with_report):
    client, session_id, report = client_with_report
    # Move chart 5 (originally sidebar) to main
    layout = report["layout"]
    chart_5 = layout[5]
    chart_5["position"] = "main"
    chart_5["order"] = 5
    resp = client.patch(f"/report/{session_id}/layout", json=layout)
    assert resp.status_code == 204
    # Re-fetch and verify
    resp2 = client.get(f"/report/{session_id}")
    assert resp2.status_code == 200
    updated = resp2.json()
    moved = next(e for e in updated["layout"] if e["chart_id"] == chart_5["chart_id"])
    assert moved["position"] == "main"


def test_generate_more_appends_charts_to_sidebar(client_with_report, sales, monkeypatch):
    client, session_id, report = client_with_report

    # Build a second FakeClaude that returns 5 NEW charts + a narrative
    new_chart_calls = []
    for i in range(5):
        new_chart_calls.append(tool_use(
            "histogram_chart",
            {"column": "revenue", "title": f"More chart {i}", "intent": f"new intent {i}"},
            id_=f"tu_more_{i}",
        ))
    new_fake = FakeClaude([
        {"tool_calls": new_chart_calls},
        {"tool_calls": [tool_use(
            "submit_narrative",
            {"summary": "Updated.", "captions": [f"new cap {i}" for i in range(5)], "data_quality": []},
        )]},
    ])

    # Swap the claude client to use the new fake
    from main import app, get_claude_client
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=new_fake)

    initial_chart_count = len(report["charts"])
    initial_sidebar_count = sum(1 for e in report["layout"] if e["position"] == "sidebar")

    resp = client.post(f"/report/{session_id}/generate-more")
    assert resp.status_code == 200

    updated = resp.json()
    assert len(updated["charts"]) == initial_chart_count + 5
    assert len(updated["layout"]) == len(updated["charts"])

    new_sidebar_count = sum(1 for e in updated["layout"] if e["position"] == "sidebar")
    assert new_sidebar_count == initial_sidebar_count + 5

    # New chart_ids are unique
    all_ids = [c["chart_id"] for c in updated["charts"]]
    assert len(all_ids) == len(set(all_ids))


def test_generate_more_unknown_session(client_with_report):
    client, _, _ = client_with_report
    resp = client.post("/report/no-such-session/generate-more")
    assert resp.status_code == 404
