import numpy as np
import pandas as pd
import pytest
from chart_executor import execute_histogram_chart
from schemas import ChartSpec, ToolError


def _params(column="duration_minutes", title="t", intent="i"):
    return {"column": column, "title": title, "intent": intent}


def test_happy_path_returns_spec(activities):
    result = execute_histogram_chart(activities, _params())
    assert isinstance(result, ChartSpec)
    assert result.kind == "histogram"
    assert len(result.x) == len(result.y)
    assert len(result.x) >= 5
    assert len(result.x) <= 20


def test_y_values_sum_to_input_count():
    df = pd.DataFrame({"x": list(np.random.RandomState(0).normal(100, 15, 200))})
    result = execute_histogram_chart(df, _params(column="x"))
    assert isinstance(result, ChartSpec)
    assert sum(result.y) == 200


def test_outlier_trimmed_bins_are_useful():
    """One extreme outlier should NOT destroy bin distribution."""
    values = list(np.random.RandomState(0).normal(100, 15, 200))
    values.append(100_000.0)  # extreme outlier
    df = pd.DataFrame({"x": values})
    result = execute_histogram_chart(df, _params(column="x"))
    assert isinstance(result, ChartSpec)
    non_empty_bins = sum(1 for v in result.y if v > 0)
    assert non_empty_bins >= 5, f"only {non_empty_bins} non-empty bins; outlier trimming failed"


def test_constant_column_returns_error():
    df = pd.DataFrame({"x": [5.0] * 50})
    result = execute_histogram_chart(df, _params(column="x"))
    assert isinstance(result, ToolError)
    assert "constant" in result.reason.lower() or "no variance" in result.reason.lower()


def test_non_numeric_returns_error():
    df = pd.DataFrame({"x": ["a", "b", "c"]})
    result = execute_histogram_chart(df, _params(column="x"))
    assert isinstance(result, ToolError)
    assert "numeric" in result.reason.lower()


def test_missing_column_returns_error(activities):
    result = execute_histogram_chart(activities, _params(column="nope"))
    assert isinstance(result, ToolError)


def test_all_null_returns_error():
    df = pd.DataFrame({"x": [None, None, None]})
    result = execute_histogram_chart(df, _params(column="x"))
    assert isinstance(result, ToolError)
