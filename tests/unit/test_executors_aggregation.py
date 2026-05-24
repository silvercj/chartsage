import pandas as pd
import pytest
from chart_executor import execute_aggregation_bar_chart
from schemas import ChartSpec, ToolError


def _params(value_col="revenue", group_col="region", agg="sum", title="t", intent="i"):
    return {"value_col": value_col, "group_col": group_col, "agg": agg, "title": title, "intent": intent}


def test_sum_happy_path(sales):
    result = execute_aggregation_bar_chart(sales, _params())
    assert isinstance(result, ChartSpec)
    assert set(result.x) == {"north", "south", "east", "west"}
    # Verify exact sum for one region
    idx = result.x.index("north")
    expected = sales[sales["region"] == "north"]["revenue"].sum()
    assert result.y[idx] == pytest.approx(expected)


def test_mean_aggregation(sales):
    result = execute_aggregation_bar_chart(sales, _params(agg="mean"))
    assert isinstance(result, ChartSpec)
    idx = result.x.index("north")
    expected = sales[sales["region"] == "north"]["revenue"].mean()
    assert result.y[idx] == pytest.approx(expected)


def test_median_aggregation(sales):
    result = execute_aggregation_bar_chart(sales, _params(agg="median"))
    assert isinstance(result, ChartSpec)


def test_min_max_aggregations(sales):
    rmin = execute_aggregation_bar_chart(sales, _params(agg="min"))
    rmax = execute_aggregation_bar_chart(sales, _params(agg="max"))
    assert isinstance(rmin, ChartSpec) and isinstance(rmax, ChartSpec)
    for region in rmin.x:
        i_min = rmin.x.index(region)
        i_max = rmax.x.index(region)
        assert rmin.y[i_min] <= rmax.y[i_max]


def test_count_agg_rejected():
    """count is for frequency_bar_chart, not aggregation_bar_chart."""
    df = pd.DataFrame({"v": [1, 2, 3], "g": ["a", "b", "a"]})
    result = execute_aggregation_bar_chart(df, _params(value_col="v", group_col="g", agg="count"))
    assert isinstance(result, ToolError)
    assert "frequency_bar_chart" in result.reason


def test_value_col_must_be_numeric():
    df = pd.DataFrame({"v": ["x", "y", "z"], "g": ["a", "b", "a"]})
    result = execute_aggregation_bar_chart(df, _params(value_col="v", group_col="g"))
    assert isinstance(result, ToolError)
    assert "numeric" in result.reason.lower()


def test_missing_value_col(sales):
    result = execute_aggregation_bar_chart(sales, _params(value_col="nope"))
    assert isinstance(result, ToolError)
    assert "nope" in result.reason


def test_missing_group_col(sales):
    result = execute_aggregation_bar_chart(sales, _params(group_col="nope"))
    assert isinstance(result, ToolError)
    assert "nope" in result.reason


def test_too_many_groups():
    df = pd.DataFrame({"v": list(range(50)), "g": [f"g{i}" for i in range(50)]})
    result = execute_aggregation_bar_chart(df, _params(value_col="v", group_col="g"))
    assert isinstance(result, ToolError)
