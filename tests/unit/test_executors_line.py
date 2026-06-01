import pandas as pd
import pytest
from chart_executor import execute_line_chart
from schemas import ChartSpec, ToolError


def _params(date_col="activity_date", value_col="duration_minutes",
            agg="count", granularity="month", group_by=None, title="t", intent="i"):
    p = {"date_col": date_col, "value_col": value_col, "agg": agg,
         "granularity": granularity, "title": title, "intent": intent}
    if group_by is not None:
        p["group_by"] = group_by
    return p


def test_count_by_month(activities):
    result = execute_line_chart(activities, _params(agg="count"))
    assert isinstance(result, ChartSpec)
    assert result.kind == "line"
    assert len(result.x) == len(result.y)
    assert sum(result.y) == len(activities)


def test_mean_by_month(activities):
    result = execute_line_chart(activities, _params(agg="mean"))
    assert isinstance(result, ChartSpec)
    assert all(isinstance(v, float) for v in result.y)


def test_sum_by_quarter(activities):
    result = execute_line_chart(activities, _params(agg="sum", granularity="quarter"))
    assert isinstance(result, ChartSpec)


def test_with_group_by(activities):
    result = execute_line_chart(activities, _params(group_by="activity_type"))
    assert isinstance(result, ChartSpec)
    assert result.series is not None
    assert len(result.series) >= 1


def test_invalid_granularity(activities):
    result = execute_line_chart(activities, _params(granularity="century"))
    assert isinstance(result, ToolError)


def test_invalid_agg(activities):
    result = execute_line_chart(activities, _params(agg="nope"))
    assert isinstance(result, ToolError)


def test_non_date_column(activities):
    result = execute_line_chart(activities, _params(date_col="activity_type"))
    assert isinstance(result, ToolError)
    assert "date" in result.reason.lower()


def test_value_col_must_be_numeric_when_not_count(activities):
    result = execute_line_chart(activities, _params(value_col="activity_type", agg="mean"))
    assert isinstance(result, ToolError)
    assert "numeric" in result.reason.lower()


def test_area_defaults_false(activities):
    result = execute_line_chart(activities, _params(agg="count"))
    assert isinstance(result, ChartSpec)
    assert result.area is False


def test_area_flag_sets_spec_area(activities):
    params = _params(agg="count")
    params["area"] = True
    result = execute_line_chart(activities, params)
    assert isinstance(result, ChartSpec)
    assert result.area is True
