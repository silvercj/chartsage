"""Deep analysis — the iterative-deepening loop in ReportGenerator.deepen.

The loop proposes follow-up charts round by round and stops when a round adds
nothing (the AI's "done" signal), hard-capped by MAX_DEEP_ROUNDS / MAX_DEEP_CHARTS.

Also covers the /deepen endpoint money path: an empty round must serve the report
unchanged with NO debit, and the enriched set must never exceed MAX_DEEP_CHARTS.
"""
import io
import types

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from uuid import uuid4

from report_generator import ReportGenerator, MAX_DEEP_CHARTS
from credits import DEEP_ANALYSIS_COST
from profile import profile_dataframe
from schemas import ChartSpec
from tests.helpers.fake_claude import FakeClaude, tool_use
from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_storage import FakeStorage
from tests.helpers.fake_posthog import FakePostHog
from tests.helpers.fake_auth import auth_identity


def test_deepen_adds_then_stops(sales):
    fc = FakeClaude([
        {"tool_calls": [tool_use("histogram_chart", {"column": "revenue", "title": "Rev dist", "intent": "i"})]},  # round 1: 1 new
        {"tool_calls": []},                                                                                          # round 2: none -> stop
    ])
    gen = ReportGenerator(
        profile=profile_dataframe(sales), df=sales,
        claude=types.SimpleNamespace(messages_create=fc),
        model_selection="m", model_narrative="m",
    )
    seed = []  # pretend no charts yet for the test
    extra = gen.deepen(seed)
    assert len(extra) == 1            # added one, then stopped on the empty round


def test_deepen_respects_chart_cap(sales):
    """The enriched set must never exceed MAX_DEEP_CHARTS even if the AI keeps
    proposing more. Seed near the cap, then script a round that proposes more
    new charts than there is room for — deepen must clamp via specs[:room]."""
    seed_count = MAX_DEEP_CHARTS - 2   # 18: only 2 slots left
    seed = [
        ChartSpec(kind="bar", title=f"seed {i}", intent="seed",
                  source_columns=["region"], data_point_count=4)
        for i in range(seed_count)
    ]
    # Round 1 proposes 5 valid histograms — far more than the 2 remaining slots.
    overflow = [
        tool_use("histogram_chart", {"column": "revenue", "title": f"H{i}", "intent": "i"}, id_=f"h{i}")
        for i in range(5)
    ]
    fc = FakeClaude([{"tool_calls": overflow}])   # one round is enough; cap stops the loop
    gen = ReportGenerator(
        profile=profile_dataframe(sales), df=sales,
        claude=types.SimpleNamespace(messages_create=fc),
        model_selection="m", model_narrative="m",
    )
    extra = gen.deepen(seed)
    assert len(extra) == 2                              # clamped to the remaining room
    assert seed_count + len(extra) == MAX_DEEP_CHARTS   # total never exceeds the cap (20)


# --- /deepen endpoint money path -------------------------------------------------

def _csv_bytes(df):
    buf = io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode("utf-8")


def _ten_chart_fake():
    calls = [tool_use("frequency_bar_chart", {"column": "region", "title": f"T{i}", "intent": f"i{i}"}, id_=f"tu_{i}")
             for i in range(10)]
    return FakeClaude([
        {"tool_calls": calls},
        {"tool_calls": [tool_use("submit_narrative", {"summary": "S.", "captions": [f"c{i}" for i in range(10)], "data_quality": []})]},
    ])


class _Holder:
    def __init__(self): self.current = None
    def __call__(self): return self.current


@pytest.fixture
def ctx(sales):
    fake_db, fake_storage, fake_posthog = FakeDB(), FakeStorage(), FakePostHog()
    holder = _Holder()
    from main import app, get_claude_client, get_db, get_storage, get_posthog, get_identity
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=_ten_chart_fake())
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[get_storage] = lambda: fake_storage
    app.dependency_overrides[get_posthog] = lambda: fake_posthog
    app.dependency_overrides[get_identity] = holder
    yield TestClient(app), fake_db, holder
    app.dependency_overrides.clear()


def _post_report(tc, sales):
    return tc.post("/generate-report", files={"file": ("s.csv", _csv_bytes(sales), "text/csv")})


def test_deepen_empty_round_does_not_debit(ctx, sales):
    """deepen() returns [] on round 1 (AI proposes nothing) -> the report is served
    unchanged and the user is NOT charged DEEP_ANALYSIS_COST. No narrative pass runs."""
    tc, db, holder = ctx
    user = str(uuid4()); holder.current = auth_identity(user)
    resp = _post_report(tc, sales)                       # creates report; balance now 200 (300 - 100)
    sid = resp.json()["session_id"]
    # Top up so the balance comfortably clears DEEP_ANALYSIS_COST — this forces the
    # gate to pass, so an unchanged balance afterward proves the debit was skipped
    # (not merely blocked by the OUT_OF_CREDITS gate).
    db.grant_credits(user, DEEP_ANALYSIS_COST, "test_topup")
    before = db.get_balance(user)
    assert before >= DEEP_ANALYSIS_COST
    report_before = db.get_report(sid)["report_json"]

    # First (and only) deepen round returns no tool calls -> deepen() == [] -> early return.
    empty = FakeClaude([{"tool_calls": []}])
    from main import app, get_claude_client
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=empty)

    resp = tc.post(f"/report/{sid}/deepen")
    assert resp.status_code == 200
    assert resp.json() == report_before                  # report unchanged
    assert db.get_balance(user) == before                # no debit
    assert not any(t["reason"] == "deep_analysis" for t in db.list_transactions(user))  # no ledger entry
