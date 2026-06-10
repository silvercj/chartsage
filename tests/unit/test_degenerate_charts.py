"""A chart that executes successfully but renders as nothing — a flat all-1s bar, a
single-point line, a one-slice pie — must be rejected like a schema error, so the
existing retry round asks the model for a better chart. This is the general mechanism
for the 'successful but meaningless' bug class (flat hurricanes bar, blowouts tangle)."""
import pandas as pd
from unittest.mock import MagicMock

from chart_executor import (
    degenerate_reason,
    execute_frequency_bar_chart,
    execute_line_chart,
    execute_aggregation_bar_chart,
)
from schemas import ChartSpec
from profile import profile_dataframe
from report_generator import ReportGenerator


def _spec(**kw) -> ChartSpec:
    base = dict(kind="bar", title="t", intent="i", source_columns=["a"], data_point_count=10)
    base.update(kw)
    return ChartSpec(**base)


def test_all_ones_frequency_bar_is_degenerate():
    # Every label unique -> counts all 1 -> flat bar (the hurricanes 'decade' bug class).
    df = pd.DataFrame({"decade": ["1990s", "2000s", "2010s", "2020s"], "v": [1.0, 2.0, 3.0, 4.0]})
    spec = execute_frequency_bar_chart(df, {"column": "decade", "title": "t", "intent": "i"})
    reason = degenerate_reason(spec)
    assert reason is not None
    assert "aggregation_bar_chart" in reason     # actionable: tells the model what to do instead


def test_normal_frequency_bar_is_fine():
    df = pd.DataFrame({"cat": ["a", "a", "a", "b", "b", "c"]})
    spec = execute_frequency_bar_chart(df, {"column": "cat", "title": "t", "intent": "i"})
    assert degenerate_reason(spec) is None


def test_single_category_bar_is_degenerate():
    df = pd.DataFrame({"cat": ["a"] * 6, "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    spec = execute_aggregation_bar_chart(
        df, {"value_col": "v", "group_col": "cat", "agg": "sum", "title": "t", "intent": "i"})
    assert degenerate_reason(spec) is not None


def test_short_line_is_degenerate():
    # All rows in one period -> a single-point 'trend'.
    df = pd.DataFrame({"d": ["2024-01-05", "2024-01-20", "2024-01-28"], "v": [1.0, 2.0, 3.0]})
    spec = execute_line_chart(df, {
        "date_col": "d", "value_col": "v", "agg": "sum", "granularity": "year",
        "title": "t", "intent": "i"})
    assert degenerate_reason(spec) is not None


def test_real_line_is_fine():
    df = pd.DataFrame({"d": [f"2024-{m:02d}-01" for m in range(1, 7)], "v": [1.0] * 6})
    spec = execute_line_chart(df, {
        "date_col": "d", "value_col": "v", "agg": "sum", "granularity": "month",
        "title": "t", "intent": "i"})
    assert degenerate_reason(spec) is None


def test_tiny_scatter_is_degenerate():
    spec = _spec(kind="scatter", x=[1.0, 2.0], y=[3.0, 4.0], source_columns=["a", "b"])
    assert degenerate_reason(spec) is not None


def test_one_slice_pie_is_degenerate():
    spec = _spec(kind="pie", x=["only"], y=[10.0])
    assert degenerate_reason(spec) is not None


def test_one_by_one_heatmap_is_degenerate():
    spec = _spec(kind="heatmap", x=["a"], y=["b"],
                 series=[{"row": "b", "col": "a", "value": 1.0}], source_columns=["a", "b"])
    assert degenerate_reason(spec) is not None


def test_one_node_treemap_is_degenerate():
    spec = _spec(kind="treemap", nodes=[{"name": "only", "value": 10.0}])
    assert degenerate_reason(spec) is not None


def test_multi_series_of_lone_points_is_degenerate():
    # One point per series = the 'spiky tangle' class.
    spec = _spec(kind="line", series=[
        {"name": "a", "x": ["2020"], "y": [1.0]},
        {"name": "b", "x": ["2021"], "y": [2.0]},
    ])
    assert degenerate_reason(spec) is not None


def test_execute_tool_calls_turns_degenerate_into_retryable_error():
    # The wiring: a degenerate chart must land in `errors` (feeding the retry round),
    # not in `specs` (the report).
    df = pd.DataFrame({"decade": ["1990s", "2000s", "2010s", "2020s"], "v": [1.0, 2.0, 3.0, 4.0]})
    gen = ReportGenerator(
        profile=profile_dataframe(df), df=df, claude=MagicMock(),
        model_selection="m", model_narrative="m",
    )
    block = MagicMock()
    block.type = "tool_use"
    block.id = "tu_bad"
    block.name = "frequency_bar_chart"
    block.input = {"column": "decade", "title": "t", "intent": "i"}
    specs, errors = gen._execute_tool_calls([block])
    assert specs == []
    assert len(errors) == 1 and errors[0]["id"] == "tu_bad"
