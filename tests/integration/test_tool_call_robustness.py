"""A single malformed chart spec must never 500 the whole report.

Executors validate their input and return ToolError for *known*-bad cases, but an
unexpected exception (e.g. the pandas "Grouper for X not 1-dimensional" raised by a
duplicate-column groupby) used to propagate out of `_execute_tool_calls` and 500 the
request (Sentry: ValueError on /report/{id}/generate-more). It must instead be caught,
logged, and recorded as an error so generation continues and the retry loop can react.
"""
import types

import report_generator as rg
from report_generator import ReportGenerator
from profile import profile_dataframe


def test_execute_tool_calls_swallows_unexpected_executor_exception(sales, monkeypatch):
    def boom(df, params):
        raise RuntimeError("kaboom")

    monkeypatch.setitem(rg.TOOL_EXECUTORS, "frequency_bar_chart", boom)
    gen = ReportGenerator(
        profile=profile_dataframe(sales),
        df=sales,
        claude=None,
        model_selection="m1",
        model_narrative="m2",
    )
    block = types.SimpleNamespace(
        type="tool_use",
        name="frequency_bar_chart",
        id="x1",
        input={"column": "region", "title": "t", "intent": "i"},
    )

    # Must NOT raise — the exception is converted into a recorded error.
    specs, errors = gen._execute_tool_calls([block])

    assert specs == []
    assert len(errors) == 1
    assert errors[0]["id"] == "x1"
