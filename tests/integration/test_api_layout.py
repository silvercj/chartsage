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


@pytest.fixture
def client_with_report(sales):
    chart_calls = [
        tool_use("frequency_bar_chart",
                 {"column": "region", "title": f"Chart {i}", "intent": f"intent {i}"},
                 id_=f"tu_{i}")
        for i in range(10)
    ]
    fake_claude = FakeClaude([
        {"tool_calls": chart_calls},
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "Sales.", "captions": [f"cap{i}" for i in range(10)], "data_quality": []})]},
    ])
    fake_db = FakeDB()
    fake_storage = FakeStorage()
    fake_posthog = FakePostHog()

    from main import app, get_claude_client, get_db, get_storage, get_posthog
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=fake_claude)
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[get_storage] = lambda: fake_storage
    app.dependency_overrides[get_posthog] = lambda: fake_posthog

    tc = TestClient(app)
    anon = str(uuid4())

    resp = tc.post("/generate-report",
                   files={"file": ("sales.csv", _csv_bytes(sales), "text/csv")},
                   headers={"X-Anon-Id": anon})
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    report = fake_db.get_report(session_id)["report_json"]

    yield tc, session_id, report, anon, fake_db, fake_storage, fake_posthog
    app.dependency_overrides.clear()


def test_patch_layout_valid_returns_204(client_with_report):
    tc, session_id, report, anon, *_ = client_with_report
    layout = report["layout"]
    layout[0], layout[1] = layout[1], layout[0]
    layout[0]["order"], layout[1]["order"] = 0, 1
    resp = tc.patch(f"/report/{session_id}/layout",
                    json=layout, headers={"X-Anon-Id": anon})
    assert resp.status_code == 204


def test_patch_layout_unknown_chart_id_returns_400(client_with_report):
    tc, session_id, _, anon, *_ = client_with_report
    bad = [{"chart_id": "does-not-exist", "position": "main", "order": 0}]
    resp = tc.patch(f"/report/{session_id}/layout",
                    json=bad, headers={"X-Anon-Id": anon})
    assert resp.status_code == 400
    assert "does-not-exist" in resp.text or "chart_id" in resp.text.lower()


def test_patch_layout_unknown_session_returns_404(client_with_report):
    tc, _, _, anon, *_ = client_with_report
    resp = tc.patch("/report/no-such-session/layout",
                    json=[], headers={"X-Anon-Id": anon})
    assert resp.status_code == 404


def test_patch_layout_persists(client_with_report):
    tc, session_id, report, anon, db, *_ = client_with_report
    layout = report["layout"]
    chart = layout[5]
    chart["position"] = "main"
    chart["order"] = 5
    resp = tc.patch(f"/report/{session_id}/layout",
                    json=layout, headers={"X-Anon-Id": anon})
    assert resp.status_code == 204

    persisted = db.get_report(session_id)["report_json"]["layout"]
    moved = next(e for e in persisted if e["chart_id"] == chart["chart_id"])
    assert moved["position"] == "main"


def test_generate_more_appends_charts(client_with_report, sales):
    tc, session_id, report, anon, db, _, ph = client_with_report

    new_calls = [
        tool_use("histogram_chart",
                 {"column": "revenue", "title": f"More {i}", "intent": f"new {i}"},
                 id_=f"more_{i}")
        for i in range(5)
    ]
    new_fake = FakeClaude([
        {"tool_calls": new_calls},
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "Updated.", "captions": [f"nc{i}" for i in range(5)], "data_quality": []})]},
    ])

    from main import app, get_claude_client
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=new_fake)

    initial_count = len(report["charts"])
    initial_sidebar = sum(1 for e in report["layout"] if e["position"] == "sidebar")

    resp = tc.post(f"/report/{session_id}/generate-more",
                   headers={"X-Anon-Id": anon})
    assert resp.status_code == 200
    updated = resp.json()
    assert len(updated["charts"]) == initial_count + 5

    new_sidebar = sum(1 for e in updated["layout"] if e["position"] == "sidebar")
    assert new_sidebar == initial_sidebar + 5
    # PostHog: success event fired
    assert len(ph.find("generate_more_succeeded")) == 1


def test_generate_more_unknown_session(client_with_report):
    tc, _, _, anon, *_ = client_with_report
    resp = tc.post("/report/nope/generate-more", headers={"X-Anon-Id": anon})
    assert resp.status_code == 404
