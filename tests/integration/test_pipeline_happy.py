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
