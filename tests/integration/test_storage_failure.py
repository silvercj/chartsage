import io
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


def test_storage_upload_failure_returns_502_and_leaves_no_row(sales):
    chart_calls = [tool_use("frequency_bar_chart",
                            {"column": "region", "title": f"T{i}", "intent": f"i{i}"},
                            id_=f"tu_{i}")
                   for i in range(10)]
    fake_claude = FakeClaude([
        {"tool_calls": chart_calls},
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "S.", "captions": [f"c{i}" for i in range(10)], "data_quality": []})]},
    ])
    fake_db = FakeDB()
    fake_storage = FakeStorage()
    fake_storage.fail_next_upload()
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
    assert resp.status_code == 502
    assert "STORAGE_UNAVAILABLE" in resp.text
    assert fake_db.count_anon_reports(anon) == 0     # no orphan row

    app.dependency_overrides.clear()
