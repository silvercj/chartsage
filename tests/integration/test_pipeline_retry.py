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
    return ReportGenerator(profile=profile, df=df, claude=client,
                           model_selection="m1", model_narrative="m2")


def test_retry_recovers_from_bad_column(activities):
    fake = FakeClaude([
        # First call: one tool fails (bad column)
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "nope", "title": "Bad", "intent": "fail"}, id_="bad1"),
            tool_use("frequency_bar_chart", {
                "column": "activity_type", "title": "Good", "intent": "good"}, id_="good1"),
        ]},
        # Retry call: Claude corrects the bad one
        {"tool_calls": [
            tool_use("histogram_chart", {
                "column": "duration_minutes", "title": "Fixed", "intent": "fixed"}, id_="fix1"),
        ]},
    ])
    gen = _make_generator(activities, fake)
    specs = gen.generate_charts()
    titles = {s.title for s in specs}
    assert "Good" in titles
    assert "Fixed" in titles


def test_retry_failures_are_dropped(activities):
    fake = FakeClaude([
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "nope1", "title": "t1", "intent": "i"}, id_="b1"),
            tool_use("frequency_bar_chart", {
                "column": "activity_type", "title": "Good", "intent": "good"}, id_="g1"),
        ]},
        # Retry returns more broken tools
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "still_bad", "title": "t2", "intent": "i"}, id_="b2"),
        ]},
    ])
    gen = _make_generator(activities, fake)
    specs = gen.generate_charts()
    titles = {s.title for s in specs}
    assert "Good" in titles
    assert "t1" not in titles
    assert "t2" not in titles


def test_no_retry_when_no_errors(activities):
    fake = FakeClaude([
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "activity_type", "title": "Good", "intent": "good"}),
            tool_use("histogram_chart", {
                "column": "duration_minutes", "title": "Hist", "intent": "spread"}),
            tool_use("aggregation_bar_chart", {
                "value_col": "duration_minutes", "group_col": "activity_type",
                "agg": "mean", "title": "Mean", "intent": "compare"}),
        ]},
        # If retry happens, this will be hit and tests fail
    ])
    gen = _make_generator(activities, fake)
    specs = gen.generate_charts()
    assert len(specs) == 3
    assert len(fake.calls) == 1
