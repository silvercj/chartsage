"""Month/weekday/quarter names on a category axis must render chronologically, not
alphabetically (Apr, Aug, Dec…) and not by value — 'sales by month' sorted by revenue
reads as noise. natural_category_order detects a known cyclic sequence; the category
executors apply it to their final axis order."""
import pandas as pd

from data_processing_utils import natural_category_order
from chart_executor import (
    execute_aggregation_bar_chart,
    execute_frequency_bar_chart,
    execute_grouped_bar_chart,
    execute_dual_axis_chart,
    execute_heatmap_chart,
)


def test_detects_month_names_and_abbreviations():
    assert natural_category_order(["Mar", "Jan", "Feb"]) == ["Jan", "Feb", "Mar"]
    assert natural_category_order(["September", "January", "June"]) == \
        ["January", "June", "September"]
    assert natural_category_order(["Sept", "Jan", "Jun"]) == ["Jan", "Jun", "Sept"]


def test_detects_weekdays_and_quarters():
    assert natural_category_order(["Wed", "Mon", "Fri"]) == ["Mon", "Wed", "Fri"]
    assert natural_category_order(["Q4", "Q1", "Q2"]) == ["Q1", "Q2", "Q4"]


def test_plain_categories_are_left_alone():
    assert natural_category_order(["West", "East", "North"]) is None
    # Too few to be confident they're months ('May'/'Mar' could be names).
    assert natural_category_order(["May", "Mar"]) is None


def _month_df():
    return pd.DataFrame({
        "month": ["Mar", "Jan", "Feb", "Mar", "Jan", "Feb"],
        "region": ["a", "a", "a", "b", "b", "b"],
        "sales": [30.0, 10.0, 20.0, 35.0, 15.0, 25.0],
    })


def test_aggregation_bar_orders_months_chronologically():
    spec = execute_aggregation_bar_chart(_month_df(), {
        "value_col": "sales", "group_col": "month", "agg": "sum",
        "title": "t", "intent": "i"})
    assert spec.x == ["Jan", "Feb", "Mar"]
    assert spec.y == [25.0, 45.0, 65.0]      # values follow their categories


def test_frequency_bar_orders_months_chronologically():
    df = pd.DataFrame({"month": ["Mar", "Mar", "Mar", "Jan", "Feb", "Feb"]})
    spec = execute_frequency_bar_chart(df, {"column": "month", "title": "t", "intent": "i"})
    assert spec.x == ["Jan", "Feb", "Mar"]
    assert spec.y == [1, 2, 3]


def test_grouped_bar_orders_months_chronologically():
    spec = execute_grouped_bar_chart(_month_df(), {
        "category_col": "month", "breakdown_col": "region", "value_col": "sales",
        "agg": "sum", "mode": "grouped", "title": "t", "intent": "i"})
    assert spec.x == ["Jan", "Feb", "Mar"]


def test_dual_axis_orders_months_chronologically():
    spec = execute_dual_axis_chart(_month_df(), {
        "x_col": "month", "bar_value_col": "sales", "line_value_col": "sales",
        "bar_agg": "sum", "line_agg": "mean", "title": "t", "intent": "i"})
    assert spec.x == ["Jan", "Feb", "Mar"]


def test_heatmap_pivot_orders_month_axis_chronologically():
    spec = execute_heatmap_chart(_month_df(), {
        "mode": "pivot", "row_col": "region", "col_col": "month",
        "value_col": "sales", "agg": "sum", "title": "t", "intent": "i"})
    assert spec.x == ["Jan", "Feb", "Mar"]


def test_value_sort_kept_for_plain_categories():
    df = pd.DataFrame({"team": ["x", "y", "z"], "score": [5.0, 9.0, 7.0]})
    spec = execute_aggregation_bar_chart(df, {
        "value_col": "score", "group_col": "team", "agg": "sum",
        "title": "t", "intent": "i"})
    assert spec.x == ["y", "z", "x"]         # ranking by value still the default
