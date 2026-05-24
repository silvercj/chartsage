"""Chart executors: pure functions from (df, tool_params) to ChartSpec or ToolError.

One executor per Anthropic tool. Each must:
- Validate columns exist with correct roles
- Validate cardinality constraints
- Compute the chart data and return a fully-populated ChartSpec
- On any failure, return ToolError with an actionable, specific reason
"""
from typing import Any, Callable
import numpy as np
import pandas as pd
from schemas import ChartSpec, ToolError


MAX_CATEGORIES = 30
MAX_PIE_SLICES = 8
MAX_SCATTER_POINTS = 5000


def _err(reason: str) -> ToolError:
    return ToolError(reason=reason)


def _available_columns_by_role(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group columns by inferred role for error messages."""
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if df[c].dtype == "object" and df[c].nunique(dropna=True) <= 50]
    return {"numeric": numeric, "categorical": categorical, "all": list(df.columns)}


def execute_frequency_bar_chart(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    column = params["column"]
    title = params["title"]
    intent = params["intent"]

    if column not in df.columns:
        cats = _available_columns_by_role(df)["categorical"]
        return _err(f"'{column}' is not a column. Available categorical columns: {cats}")

    series = df[column].dropna()
    if len(series) == 0:
        return _err(f"'{column}' has no non-null values.")

    counts = series.value_counts()
    if len(counts) > MAX_CATEGORIES:
        return _err(
            f"'{column}' has {len(counts)} unique values, more than the max ({MAX_CATEGORIES}). "
            f"Use frequency charts on lower-cardinality columns; this column may be an identifier."
        )

    x = [str(v) for v in counts.index.tolist()]
    y = [int(v) for v in counts.values.tolist()]

    return ChartSpec(
        kind="bar",
        title=title,
        intent=intent,
        x=x,
        y=y,
        x_label=column,
        y_label="Count",
        x_display_type="category",
        y_display_type="count",
        source_columns=[column],
        data_point_count=int(len(series)),
    )


TOOL_EXECUTORS: dict[str, Callable[[pd.DataFrame, dict], ChartSpec | ToolError]] = {
    "frequency_bar_chart": execute_frequency_bar_chart,
}
