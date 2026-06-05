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


def test_many_groups_ranks_top_n():
    """Many groups -> rank by value and show the top N, not an error.
    This is the 'rank entities by a metric' bar (e.g. pole-to-win % by circuit)."""
    from chart_executor import MAX_CATEGORIES
    df = pd.DataFrame({"v": list(range(50)), "g": [f"g{i}" for i in range(50)]})
    result = execute_aggregation_bar_chart(df, _params(value_col="v", group_col="g", agg="mean"))
    assert isinstance(result, ChartSpec)
    assert len(result.x) == MAX_CATEGORIES                  # top-N, not an error
    assert result.y == sorted(result.y, reverse=True)       # ranked descending
    assert result.x[0] == "g49" and result.y[0] == 49.0     # highest value first


def test_one_row_per_entity_ranks():
    """Rank entities by a value when there's one row each (agg of a single value = itself)."""
    df = pd.DataFrame({"circuit": ["a", "b", "c"], "rate": [70.0, 30.0, 50.0]})
    r = execute_aggregation_bar_chart(df, _params(value_col="rate", group_col="circuit", agg="mean"))
    assert isinstance(r, ChartSpec)
    assert r.x == ["a", "c", "b"]          # sorted by rate desc: 70, 50, 30
    assert r.y == [70.0, 50.0, 30.0]
