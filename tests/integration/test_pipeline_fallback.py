import pandas as pd
import pytest
from unittest.mock import MagicMock
from report_generator import ReportGenerator
from profile import profile_dataframe
from tests.helpers.fake_claude import FakeClaude, tool_use


def _make_generator(df, fake):
    profile = profile_dataframe(df)
    client = MagicMock()
    client.messages_create = fake
    return ReportGenerator(profile=profile, df=df, claude=client,
                           model_selection="m1", model_narrative="m2")


def test_fallback_when_claude_returns_nothing(activities):
    fake = FakeClaude([
        {"tool_calls": []},  # pass #1 first call: nothing
        # No retry call expected because there are no errors to send back.
        # Narrative is still called because we have fallback charts.
        {"tool_calls": [
            tool_use("submit_narrative", {
                "summary": "Auto summary.",
                "captions": ["c1", "c2", "c3"],
                "data_quality": [],
            })
        ]},
    ])
    gen = _make_generator(activities, fake)
    report = gen.build_report()
    assert len(report.charts) >= 3
    assert all("fallback" in c.spec.intent.lower() for c in report.charts)


def test_fallback_when_all_tool_calls_error(activities):
    fake = FakeClaude([
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "nope", "title": "x", "intent": "x"}, id_="e1"),
        ]},
        # Retry returns more errors
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "still_nope", "title": "x", "intent": "x"}, id_="e2"),
        ]},
        # Narrative
        {"tool_calls": [
            tool_use("submit_narrative", {
                "summary": "s", "captions": ["c1", "c2", "c3"], "data_quality": [],
            })
        ]},
    ])
    gen = _make_generator(activities, fake)
    report = gen.build_report()
    assert len(report.charts) >= 3
