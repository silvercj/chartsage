"""Deep analysis — the iterative-deepening loop in ReportGenerator.deepen.

The loop proposes follow-up charts round by round and stops when a round adds
nothing (the AI's "done" signal), hard-capped by MAX_DEEP_ROUNDS / MAX_DEEP_CHARTS.
"""
import types

from report_generator import ReportGenerator
from profile import profile_dataframe
from tests.helpers.fake_claude import FakeClaude, tool_use


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
