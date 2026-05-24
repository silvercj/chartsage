"""Regression net for the original frequency-chart bug.

The old system computed value_counts on the counts themselves, producing a
'histogram of frequencies' instead of an actual frequency chart. These tests
assert the EXACT x/y values that the correct implementation produces.
"""
import pandas as pd
import pytest
from chart_executor import execute_frequency_bar_chart
from schemas import ChartSpec, ToolError


def _params(column="activity_type", title="t", intent="i"):
    return {"column": column, "title": title, "intent": intent}


def test_happy_path_exact_values():
    df = pd.DataFrame({"activity_type": ["a", "b", "a", "a", "b", "c"]})
    result = execute_frequency_bar_chart(df, _params())
    assert isinstance(result, ChartSpec)
    assert result.x == ["a", "b", "c"]
    assert result.y == [3, 2, 1]


def test_sorted_descending_by_count():
    df = pd.DataFrame({"activity_type": ["c"] * 5 + ["a"] * 2 + ["b"] * 3})
    result = execute_frequency_bar_chart(df, _params())
    assert result.x == ["c", "b", "a"]
    assert result.y == [5, 3, 2]


def test_with_real_fixture(activities):
    result = execute_frequency_bar_chart(activities, _params())
    assert isinstance(result, ChartSpec)
    assert "consultation" in result.x
    assert sum(result.y) == len(activities)


def test_missing_column_returns_error(activities):
    result = execute_frequency_bar_chart(activities, _params(column="not_a_column"))
    assert isinstance(result, ToolError)
    assert "not_a_column" in result.reason
    assert "activity_type" in result.reason  # offers alternatives


def test_too_many_categories_returns_error():
    df = pd.DataFrame({"high_card": [f"v{i}" for i in range(40)]})
    result = execute_frequency_bar_chart(df, _params(column="high_card"))
    assert isinstance(result, ToolError)
    assert "40" in result.reason or "too many" in result.reason.lower()


def test_all_null_returns_error():
    df = pd.DataFrame({"col": [None, None, None]})
    result = execute_frequency_bar_chart(df, _params(column="col"))
    assert isinstance(result, ToolError)


def test_chartspec_metadata_populated():
    df = pd.DataFrame({"activity_type": ["a", "b", "a"]})
    result = execute_frequency_bar_chart(df, _params())
    assert result.kind == "bar"
    assert result.source_columns == ["activity_type"]
    assert result.data_point_count == 3
    assert result.y_display_type == "count"
