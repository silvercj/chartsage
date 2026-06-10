import pandas as pd
import pytest
from chart_executor import execute_heatmap_chart
from schemas import ChartSpec, ToolError


def test_correlation_mode_happy_path():
    df = pd.DataFrame({
        "a": list(range(20)),
        "b": [i * 2 for i in range(20)],
        "c": [20 - i for i in range(20)],
        "label": ["x"] * 20,
    })
    params = {"mode": "correlation", "title": "t", "intent": "i"}
    result = execute_heatmap_chart(df, params)
    assert isinstance(result, ChartSpec)
    assert result.kind == "heatmap"
    assert result.series is not None
    assert len(result.x) == 3
    assert len(result.y) == 3


def test_correlation_mode_needs_two_numeric():
    df = pd.DataFrame({"a": list(range(5)), "label": ["x"] * 5})
    params = {"mode": "correlation", "title": "t", "intent": "i"}
    result = execute_heatmap_chart(df, params)
    assert isinstance(result, ToolError)
    assert "numeric" in result.reason.lower()


def test_pivot_mode_happy_path():
    df = pd.DataFrame({
        "row_cat": ["a", "a", "b", "b", "c", "c"],
        "col_cat": ["x", "y", "x", "y", "x", "y"],
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    params = {
        "mode": "pivot", "title": "t", "intent": "i",
        "row_col": "row_cat", "col_col": "col_cat",
        "value_col": "value", "agg": "sum",
    }
    result = execute_heatmap_chart(df, params)
    assert isinstance(result, ChartSpec)
    assert sorted(result.x) == ["x", "y"]
    assert sorted(result.y) == ["a", "b", "c"]


def test_pivot_mode_count():
    df = pd.DataFrame({
        "row_cat": ["a", "a", "b", "b"],
        "col_cat": ["x", "y", "x", "y"],
    })
    params = {
        "mode": "pivot", "title": "t", "intent": "i",
        "row_col": "row_cat", "col_col": "col_cat", "agg": "count",
    }
    result = execute_heatmap_chart(df, params)
    assert isinstance(result, ChartSpec)


def test_invalid_mode():
    df = pd.DataFrame({"a": [1, 2, 3]})
    params = {"mode": "wrong", "title": "t", "intent": "i"}
    result = execute_heatmap_chart(df, params)
    assert isinstance(result, ToolError)


def test_pivot_missing_value_col_for_non_count():
    df = pd.DataFrame({"r": ["a", "b"], "c": ["x", "y"]})
    params = {
        "mode": "pivot", "title": "t", "intent": "i",
        "row_col": "r", "col_col": "c", "agg": "sum",
    }
    result = execute_heatmap_chart(df, params)
    assert isinstance(result, ToolError)
    assert "value_col" in result.reason.lower()


def test_correlation_excludes_identifier_columns():
    # A row-ID is numeric by dtype but meaningless in a correlation matrix — the prompt
    # rule 'never use an identifier as a metric' is now enforced at the executor too.
    df = pd.DataFrame({
        "order_id": list(range(1, 13)),
        "revenue": [100.0, 120.0, 90.0, 200.0, 150.0, 130.0,
                    170.0, 110.0, 95.0, 180.0, 140.0, 160.0],
        "units": [1.0, 2.0, 1.0, 4.0, 3.0, 2.0, 3.0, 1.0, 1.0, 4.0, 2.0, 3.0],
    })
    result = execute_heatmap_chart(df, {"mode": "correlation", "title": "t", "intent": "i"})
    assert isinstance(result, ChartSpec)
    assert "order_id" not in result.x


def test_correlation_errors_when_only_ids_and_one_metric():
    df = pd.DataFrame({
        "order_id": list(range(1, 13)),
        "revenue": [float(v) for v in range(12)],
    })
    result = execute_heatmap_chart(df, {"mode": "correlation", "title": "t", "intent": "i"})
    assert isinstance(result, ToolError)
