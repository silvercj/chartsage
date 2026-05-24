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
