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
    df = pd.DataFrame({
        "x": [1.0, 2.0, None, 4.0, 5.0],
        "y": [10.0, None, 30.0, 40.0, 50.0],
    })
    result = execute_scatter_chart(df, _params(x_col="x", y_col="y"))
    assert isinstance(result, ChartSpec)
    # Only rows 0 (1,10), 3 (4,40), 4 (5,50) have both
    assert len(result.x) == 3


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
