import pandas as pd
import pytest
from unittest.mock import MagicMock
from report_generator import ReportGenerator
from profile import profile_dataframe
from schemas import ChartSpec
from tests.helpers.fake_claude import FakeClaude, tool_use


def _make_generator(df, fake):
    profile = profile_dataframe(df)
    client = MagicMock()
    client.messages_create = fake
    gen = ReportGenerator(profile=profile, df=df, claude=client,
                          model_selection="m1", model_narrative="m2")
    return gen


def test_happy_path_returns_chart_specs(activities):
    fake = FakeClaude([
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "activity_type", "title": "Type counts", "intent": "show mix"}),
            tool_use("histogram_chart", {
                "column": "duration_minutes", "title": "Duration", "intent": "show spread"}),
            tool_use("aggregation_bar_chart", {
                "value_col": "duration_minutes", "group_col": "activity_type",
                "agg": "median", "title": "Median by type", "intent": "compare"}),
        ]},
    ])
    gen = _make_generator(activities, fake)
    specs = gen.generate_charts()
    assert len(specs) == 3
    assert all(isinstance(s, ChartSpec) for s in specs)


def test_caps_at_ten_charts(activities):
    """Even if Claude returns 15 tool calls, we only keep the first 10."""
    fake = FakeClaude([
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "activity_type", "title": f"t{i}", "intent": f"i{i}"})
            for i in range(15)
        ]},
    ])
    gen = _make_generator(activities, fake)
    specs = gen.generate_charts()
    assert len(specs) <= 10


def test_full_report_includes_narrative(activities):
    fake = FakeClaude([
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "activity_type", "title": "T1", "intent": "i1"}),
            tool_use("histogram_chart", {
                "column": "duration_minutes", "title": "T2", "intent": "i2"}),
            tool_use("aggregation_bar_chart", {
                "value_col": "duration_minutes", "group_col": "activity_type",
                "agg": "median", "title": "T3", "intent": "i3"}),
        ]},
        {"tool_calls": [
            tool_use("submit_narrative", {
                "summary": "Three findings emerged.",
                "captions": ["Caption A.", "Caption B.", "Caption C."],
                "data_quality": ["Note: some negative durations."],
            }),
        ]},
    ])
    gen = _make_generator(activities, fake)
    report = gen.build_report()
    assert report.summary == "Three findings emerged."
    assert len(report.charts) == 3
    assert report.charts[0].caption == "Caption A."
    assert "negative" in report.data_quality[0]


def test_key_metrics_routed_to_report_not_charts(sales):
    """A key_metrics tool_use populates report.key_metrics and is NOT a chart.

    Pass #1 emits key_metrics + 3 real chart tool_uses. The 3 charts clear
    MIN_CHARTS_FOR_NO_FALLBACK so no heuristic fallback fires, leaving charts
    determined solely by the chart tool_uses. If key_metrics were wrongly
    treated as a chart, charts would be 4 — pinning the exclusion contract.
    """
    fake = FakeClaude([
        {"tool_calls": [
            tool_use("key_metrics", {"metrics": [
                {"label": "Total revenue", "column": "revenue", "agg": "sum", "format": "currency"},
                {"label": "Regions", "column": "region", "agg": "nunique", "format": "number"},
            ]}),
            tool_use("frequency_bar_chart", {
                "column": "region", "title": "Orders by region", "intent": "show mix"}),
            tool_use("aggregation_bar_chart", {
                "value_col": "revenue", "group_col": "region",
                "agg": "sum", "title": "Revenue by region", "intent": "compare"}),
            tool_use("line_chart", {
                "date_col": "order_date", "value_col": "revenue", "agg": "sum",
                "granularity": "month", "title": "Revenue over time", "intent": "trend"}),
        ]},
        {"tool_calls": [
            tool_use("submit_narrative", {
                "summary": "Revenue concentrates in a few regions.",
                "captions": ["Caption A.", "Caption B.", "Caption C."],
                "data_quality": [],
            }),
        ]},
    ])
    gen = _make_generator(sales, fake)
    report = gen.build_report()

    # key_metrics populated and values computed from the dataframe (not the AI)
    assert len(report.key_metrics) == 2
    by_label = {m.label: m for m in report.key_metrics}
    assert by_label["Total revenue"].value == float(sales["revenue"].sum())
    assert by_label["Total revenue"].format == "currency"
    assert by_label["Regions"].value == float(sales["region"].nunique())

    # the key_metrics call is NOT a chart: only the 3 chart tool_uses landed
    assert len(report.charts) == 3
    assert all(c.spec.kind != "key_metrics" for c in report.charts)
    # and it never entered the layout
    assert len(report.layout) == 3


def test_narrative_template_fallback_on_failure(activities):
    """If pass #2 returns nothing usable, we fill in a template."""
    fake = FakeClaude([
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "activity_type", "title": "T1", "intent": "i1"}),
            tool_use("histogram_chart", {
                "column": "duration_minutes", "title": "T2", "intent": "i2"}),
            tool_use("aggregation_bar_chart", {
                "value_col": "duration_minutes", "group_col": "activity_type",
                "agg": "median", "title": "T3", "intent": "i3"}),
        ]},
        # Pass #2: no submit_narrative call (degraded)
        {"tool_calls": [], "text": "I cannot."},
    ])
    gen = _make_generator(activities, fake)
    report = gen.build_report()
    assert report.summary  # non-empty fallback
    assert len(report.charts) == 3
    # captions degrade to spec.intent
    assert report.charts[0].caption == "i1"
