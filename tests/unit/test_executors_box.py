import pandas as pd
import pytest
from chart_executor import execute_box_plot
from schemas import ChartSpec, ToolError


def _params(value_col="duration_minutes", group_col=None, title="t", intent="i"):
    p = {"value_col": value_col, "title": title, "intent": intent}
    if group_col is not None:
        p["group_col"] = group_col
    return p


def test_single_box(activities):
    result = execute_box_plot(activities, _params())
    assert isinstance(result, ChartSpec)
    assert result.kind == "box"
    assert result.series is not None
    assert len(result.series) == 1
    stats = result.series[0]
    for k in ("min", "q1", "median", "q3", "max"):
        assert k in stats


def test_grouped_box(activities):
    result = execute_box_plot(activities, _params(group_col="activity_type"))
    assert isinstance(result, ChartSpec)
    assert len(result.series) == activities["activity_type"].nunique()


def test_non_numeric_value(activities):
    result = execute_box_plot(activities, _params(value_col="activity_type"))
    assert isinstance(result, ToolError)
    assert "numeric" in result.reason.lower()


def test_missing_value_col(activities):
    result = execute_box_plot(activities, _params(value_col="nope"))
    assert isinstance(result, ToolError)


def test_group_too_many():
    df = pd.DataFrame({"v": list(range(50)), "g": [f"g{i}" for i in range(50)]})
    result = execute_box_plot(df, _params(value_col="v", group_col="g"))
    assert isinstance(result, ToolError)
