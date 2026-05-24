import numpy as np
import pandas as pd
import pytest
from chart_executor import execute_scatter_chart
from schemas import ChartSpec, ToolError


def _params(x_col="duration_minutes", y_col="activity_id", color_by=None, title="t", intent="i"):
    p = {"x_col": x_col, "y_col": y_col, "title": title, "intent": intent}
    if color_by is not None:
        p["color_by"] = color_by
    return p


def test_happy_path(activities):
    result = execute_scatter_chart(activities, _params())
    assert isinstance(result, ChartSpec)
    assert result.kind == "scatter"
    assert len(result.x) == len(result.y)
    assert len(result.x) <= len(activities)


def test_with_color_by(activities):
    result = execute_scatter_chart(activities, _params(color_by="activity_type"))
    assert isinstance(result, ChartSpec)
    assert result.series is not None
    assert len(result.series) > 0


def test_drops_nan_pairs():
    # 20 rows, 18 unique x values (above MIN_SCATTER_X_CARDINALITY=15) with two NaN rows.
    df = pd.DataFrame({
        "x": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
              11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        "y": [10.0, None, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
              110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0],
    })
    result = execute_scatter_chart(df, _params(x_col="x", y_col="y"))
    assert isinstance(result, ChartSpec)
    assert len(result.x) == 18   # 20 rows minus 2 NaN pairs


def test_rejects_low_cardinality_x():
    """Discount-like data with few unique values should be redirected to box/agg."""
    df = pd.DataFrame({
        "discount": [0.0, 0.1, 0.2, 0.0, 0.1, 0.2, 0.0, 0.1, 0.2, 0.0],
        "profit": [10.0, -5.0, -20.0, 12.0, -3.0, -15.0, 8.0, -7.0, -25.0, 15.0],
    })
    result = execute_scatter_chart(df, _params(x_col="discount", y_col="profit"))
    assert isinstance(result, ToolError)
    assert "box_plot" in result.reason or "aggregation" in result.reason


def test_samples_when_over_max():
    np.random.seed(0)
    n = 6000
    df = pd.DataFrame({"x": np.random.rand(n), "y": np.random.rand(n)})
    result = execute_scatter_chart(df, _params(x_col="x", y_col="y"))
    assert isinstance(result, ChartSpec)
    assert len(result.x) <= 5000


def test_non_numeric_x_returns_error():
    df = pd.DataFrame({"x": ["a", "b"], "y": [1, 2]})
    result = execute_scatter_chart(df, _params(x_col="x", y_col="y"))
    assert isinstance(result, ToolError)
    assert "numeric" in result.reason.lower()


def test_missing_column(activities):
    result = execute_scatter_chart(activities, _params(x_col="nope"))
    assert isinstance(result, ToolError)


def test_color_by_must_be_categorical(activities):
    result = execute_scatter_chart(activities, _params(color_by="duration_minutes"))
    assert isinstance(result, ToolError)
    assert "color_by" in result.reason.lower() or "categorical" in result.reason.lower()


def test_color_by_high_cardinality_rejected():
    df = pd.DataFrame({
        "x": list(range(40)),
        "y": list(range(40)),
        "cat": [f"c{i}" for i in range(40)],
    })
    result = execute_scatter_chart(df, _params(x_col="x", y_col="y", color_by="cat"))
    assert isinstance(result, ToolError)
