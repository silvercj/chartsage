"""run_report mirrors the production endpoint assembly. Uses FakeClaude (offline)."""
import pandas as pd
import types

from qa.pipeline import run_report, RunResult
from tests.helpers.fake_claude import FakeClaude, tool_use


def _fake_claude_for_sales():
    return FakeClaude([
        {"tool_calls": [
            tool_use("frequency_bar_chart", {"column": "region", "title": "By region", "intent": "mix"}),
            tool_use("aggregation_bar_chart", {"value_col": "revenue", "group_col": "region",
                                               "agg": "sum", "title": "Rev by region", "intent": "compare"}),
            tool_use("line_chart", {"date_col": "order_date", "value_col": "revenue", "agg": "sum",
                                    "granularity": "month", "title": "Rev over time", "intent": "trend"}),
        ]},
        {"tool_calls": []},  # reach-for-more proposes nothing
        {"tool_calls": [tool_use("submit_narrative", {
            "summary": "Revenue rises over time.",
            "captions": ["A.", "B.", "C."], "data_quality": []})]},
    ])


def _sales_df():
    return pd.DataFrame({
        "order_id": list(range(1, 16)),
        "region": ["north", "south", "east", "west", "north", "south", "east", "west",
                   "north", "south", "east", "west", "north", "south", "east"],
        "revenue": [1200.0, 850.0, 2100.0, 1500.0, 1800.0, 900.0, 2400.0, 1750.0,
                    1300.0, 950.0, 2200.0, 1650.0, 1400.0, 1000.0, 2300.0],
        "order_date": pd.date_range("2024-01-01", periods=15, freq="3D"),
    })


def test_run_report_returns_prod_report_shape(monkeypatch):
    # Inject FakeClaude as the shared client so no network call happens.
    fake = _fake_claude_for_sales()
    monkeypatch.setattr(
        "qa.pipeline._build_claude",
        lambda: types.SimpleNamespace(messages_create=fake),
    )
    result = run_report(_sales_df())
    assert isinstance(result, RunResult)
    assert result.error is None
    rep = result.report
    # Exact production keys from report_generator.build_report / Report model.
    for key in ("generated_at", "summary", "data_quality", "key_metrics",
                "charts", "layout", "metadata"):
        assert key in rep, f"missing report key {key!r}"
    assert rep["summary"] == "Revenue rises over time."
    assert len(rep["charts"]) == 3
    # Layout: first 5 main / rest sidebar (here all 3 are main).
    assert all(e["position"] == "main" for e in rep["layout"])
    assert result.elapsed_ms >= 0


def test_run_report_captures_exception(monkeypatch):
    def boom():
        raise RuntimeError("claude down")
    monkeypatch.setattr("qa.pipeline._build_claude", boom)
    result = run_report(_sales_df())
    assert result.report is None
    assert result.error is not None
    assert "claude down" in result.error
