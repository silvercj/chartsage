import pandas as pd
import pytest
from chart_executor import execute_pie_chart
from schemas import ChartSpec, ToolError


def _params(category_col="activity_type", value_col=None, agg="sum", title="t", intent="i"):
    p = {"category_col": category_col, "agg": agg, "title": title, "intent": intent}
    if value_col is not None:
        p["value_col"] = value_col
    return p


def test_count_no_value_col(activities):
    result = execute_pie_chart(activities, _params(agg="count"))
    assert isinstance(result, ChartSpec)
    assert result.kind == "pie"
    assert sum(result.y) == len(activities)


def test_sum_with_value_col(activities):
    result = execute_pie_chart(activities, _params(value_col="duration_minutes", agg="sum"))
    assert isinstance(result, ChartSpec)
    expected_total = activities["duration_minutes"].sum()
    assert sum(result.y) == pytest.approx(expected_total)


def test_caps_to_max_slices_plus_other():
    df = pd.DataFrame({"cat": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"] * 5})
    result = execute_pie_chart(df, _params(category_col="cat", agg="count"))
    assert isinstance(result, ChartSpec)
    assert len(result.x) <= 9
    if len(result.x) == 9:
        assert "Other" in result.x


def test_invalid_agg(activities):
    result = execute_pie_chart(activities, _params(agg="median"))
    assert isinstance(result, ToolError)


def test_missing_category_col(activities):
    result = execute_pie_chart(activities, _params(category_col="nope"))
    assert isinstance(result, ToolError)


def test_value_col_required_for_non_count():
    df = pd.DataFrame({"cat": ["a", "b"]})
    result = execute_pie_chart(df, _params(category_col="cat", agg="sum"))
    assert isinstance(result, ToolError)
    assert "value_col" in result.reason.lower()


def test_same_category_and_value_col_rejected():
    # The model occasionally picks the SAME column for both roles. df[[col, col]]
    # then has duplicate labels and groupby(col) raises the pandas
    # "Grouper for X not 1-dimensional" ValueError (a production 500 on generate-more).
    # The executor must reject it cleanly as a ToolError instead.
    df = pd.DataFrame({"global_sales": [1.0, 2.0, 3.0, 4.0]})
    result = execute_pie_chart(
        df, _params(category_col="global_sales", value_col="global_sales", agg="sum")
    )
    assert isinstance(result, ToolError)
    assert "global_sales" in result.reason
