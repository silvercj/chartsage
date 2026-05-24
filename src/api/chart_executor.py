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


_AGGREGATION_BAR_AGGS = {"sum", "mean", "median", "min", "max"}


def execute_aggregation_bar_chart(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    value_col = params["value_col"]
    group_col = params["group_col"]
    agg = params["agg"]
    title = params["title"]
    intent = params["intent"]

    if agg == "count":
        return _err("aggregation_bar_chart does not support agg='count'. Use frequency_bar_chart for counts by category.")

    if agg not in _AGGREGATION_BAR_AGGS:
        return _err(f"agg='{agg}' is not allowed. Allowed: {sorted(_AGGREGATION_BAR_AGGS)}.")

    avail = _available_columns_by_role(df)
    if value_col not in df.columns:
        return _err(f"'{value_col}' is not a column. Available numeric columns: {avail['numeric']}")

    if group_col not in df.columns:
        return _err(f"'{group_col}' is not a column. Available categorical columns: {avail['categorical']}")

    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return _err(f"'{value_col}' is not numeric. Available numeric columns: {avail['numeric']}")

    groups = df[group_col].dropna().nunique()
    if groups > MAX_CATEGORIES:
        return _err(f"'{group_col}' has {groups} unique values, more than max ({MAX_CATEGORIES}).")

    if groups == 0:
        return _err(f"'{group_col}' has no non-null values.")

    work = df[[group_col, value_col]].dropna()
    grouped = work.groupby(group_col)[value_col].agg(agg)
    grouped = grouped.sort_values(ascending=False)

    return ChartSpec(
        kind="bar",
        title=title,
        intent=intent,
        x=[str(k) for k in grouped.index.tolist()],
        y=[float(v) for v in grouped.values.tolist()],
        x_label=group_col,
        y_label=f"{agg.capitalize()} of {value_col}",
        x_display_type="category",
        y_display_type="number",
        source_columns=[value_col, group_col],
        data_point_count=int(len(work)),
    )


def execute_histogram_chart(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    from data_processing_utils import compute_histogram_bins_and_freqs

    column = params["column"]
    title = params["title"]
    intent = params["intent"]

    if column not in df.columns:
        return _err(f"'{column}' is not a column. Available numeric columns: {_available_columns_by_role(df)['numeric']}")

    if not pd.api.types.is_numeric_dtype(df[column]):
        return _err(f"'{column}' is not numeric. Histograms require numeric columns.")

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(series) == 0:
        return _err(f"'{column}' has no non-null numeric values.")

    if series.nunique() < 2:
        return _err(f"'{column}' is constant (or has no variance); can't build a histogram.")

    labels, freqs = compute_histogram_bins_and_freqs(series)
    if not labels:
        return _err(f"'{column}' did not produce usable histogram bins.")

    return ChartSpec(
        kind="histogram",
        title=title,
        intent=intent,
        x=labels,
        y=[int(v) for v in freqs],
        x_label=column,
        y_label="Frequency",
        x_display_type="text",
        y_display_type="count",
        source_columns=[column],
        data_point_count=int(len(series)),
    )


def execute_scatter_chart(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    x_col = params["x_col"]
    y_col = params["y_col"]
    color_by = params.get("color_by")
    title = params["title"]
    intent = params["intent"]

    avail = _available_columns_by_role(df)

    for col, label in [(x_col, "x_col"), (y_col, "y_col")]:
        if col not in df.columns:
            return _err(f"{label}='{col}' is not a column. Available numeric columns: {avail['numeric']}")
        if not pd.api.types.is_numeric_dtype(df[col]):
            return _err(f"{label}='{col}' is not numeric. Scatter charts need two numeric columns.")

    cols = [x_col, y_col]
    if color_by is not None:
        if color_by not in df.columns:
            return _err(f"color_by='{color_by}' is not a column.")
        if pd.api.types.is_numeric_dtype(df[color_by]):
            return _err(f"color_by='{color_by}' is numeric or high-cardinality; pass a categorical column with ≤{MAX_CATEGORIES} groups.")
        if df[color_by].nunique() > MAX_CATEGORIES:
            return _err(f"color_by='{color_by}' has {df[color_by].nunique()} unique values; max is {MAX_CATEGORIES}.")
        cols.append(color_by)

    work = df[cols].dropna()
    if len(work) == 0:
        return _err(f"No rows where both '{x_col}' and '{y_col}' are non-null.")

    if len(work) > MAX_SCATTER_POINTS:
        work = work.sample(n=MAX_SCATTER_POINTS, random_state=42)

    if color_by is None:
        return ChartSpec(
            kind="scatter",
            title=title,
            intent=intent,
            x=[float(v) for v in work[x_col].tolist()],
            y=[float(v) for v in work[y_col].tolist()],
            x_label=x_col,
            y_label=y_col,
            x_display_type="number",
            y_display_type="number",
            source_columns=[x_col, y_col],
            data_point_count=int(len(work)),
        )

    series_list: list[dict] = []
    for group_value, group_df in work.groupby(color_by):
        series_list.append({
            "name": str(group_value),
            "x": [float(v) for v in group_df[x_col].tolist()],
            "y": [float(v) for v in group_df[y_col].tolist()],
        })

    return ChartSpec(
        kind="scatter",
        title=title,
        intent=intent,
        series=series_list,
        x_label=x_col,
        y_label=y_col,
        x_display_type="number",
        y_display_type="number",
        source_columns=[x_col, y_col, color_by],
        data_point_count=int(len(work)),
    )


TOOL_EXECUTORS: dict[str, Callable[[pd.DataFrame, dict], ChartSpec | ToolError]] = {
    "frequency_bar_chart": execute_frequency_bar_chart,
    "aggregation_bar_chart": execute_aggregation_bar_chart,
    "histogram_chart": execute_histogram_chart,
    "scatter_chart": execute_scatter_chart,
}
