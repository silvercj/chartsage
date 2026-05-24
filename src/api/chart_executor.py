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


_LINE_AGGS = {"count", "sum", "mean", "median", "min", "max"}
_GRANULARITIES = {
    "day": "D", "week": "W", "month": "M",
    "quarter": "Q", "year": "Y",
}


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def execute_line_chart(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    date_col = params["date_col"]
    value_col = params["value_col"]
    agg = params["agg"]
    granularity = params["granularity"]
    group_by = params.get("group_by")
    title = params["title"]
    intent = params["intent"]

    if agg not in _LINE_AGGS:
        return _err(f"agg='{agg}' is not allowed. Allowed: {sorted(_LINE_AGGS)}.")

    if granularity not in _GRANULARITIES:
        return _err(f"granularity='{granularity}' is not allowed. Allowed: {sorted(_GRANULARITIES.keys())}.")

    if date_col not in df.columns:
        return _err(f"date_col='{date_col}' is not a column.")

    parsed_dates = _to_datetime(df[date_col])
    if parsed_dates.notna().sum() < 0.5 * len(df):
        return _err(f"'{date_col}' could not be parsed as a date column.")

    if agg != "count":
        if value_col not in df.columns:
            return _err(f"value_col='{value_col}' is not a column.")
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            return _err(f"value_col='{value_col}' is not numeric (required when agg != 'count').")

    work = df.copy()
    work["_date"] = parsed_dates
    work = work.dropna(subset=["_date"])
    work["_period"] = work["_date"].dt.to_period(_GRANULARITIES[granularity])

    def _agg_one(g: pd.DataFrame) -> float:
        if agg == "count":
            return float(len(g))
        return float(g[value_col].agg(agg))

    if group_by is None:
        grouped = work.groupby("_period").apply(_agg_one).sort_index()
        return ChartSpec(
            kind="line",
            title=title,
            intent=intent,
            x=[str(p) for p in grouped.index.tolist()],
            y=[float(v) for v in grouped.values.tolist()],
            x_label=f"{granularity.capitalize()} ({date_col})",
            y_label=f"{agg.capitalize()}" + (f" of {value_col}" if agg != "count" else ""),
            x_display_type="date",
            y_display_type="count" if agg == "count" else "number",
            source_columns=[date_col] + ([value_col] if agg != "count" else []),
            data_point_count=int(len(work)),
        )

    if group_by not in df.columns:
        return _err(f"group_by='{group_by}' is not a column.")
    if work[group_by].nunique() > MAX_CATEGORIES:
        return _err(f"group_by='{group_by}' has {work[group_by].nunique()} unique values; max is {MAX_CATEGORIES}.")

    series_list: list[dict] = []
    all_periods: set = set()
    per_group: dict[str, pd.Series] = {}
    for gv, sub in work.groupby(group_by):
        s = sub.groupby("_period").apply(_agg_one).sort_index()
        per_group[str(gv)] = s
        all_periods.update(s.index.tolist())

    periods_sorted = sorted(all_periods)
    period_labels = [str(p) for p in periods_sorted]
    for name, s in per_group.items():
        aligned = [float(s.get(p, 0.0)) for p in periods_sorted]
        series_list.append({"name": name, "x": period_labels, "y": aligned})

    return ChartSpec(
        kind="line",
        title=title,
        intent=intent,
        series=series_list,
        x_label=f"{granularity.capitalize()} ({date_col})",
        y_label=f"{agg.capitalize()}" + (f" of {value_col}" if agg != "count" else ""),
        x_display_type="date",
        y_display_type="count" if agg == "count" else "number",
        source_columns=[date_col, group_by] + ([value_col] if agg != "count" else []),
        data_point_count=int(len(work)),
    )


_PIE_AGGS = {"sum", "mean", "count"}


def execute_pie_chart(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    category_col = params["category_col"]
    value_col = params.get("value_col")
    agg = params["agg"]
    title = params["title"]
    intent = params["intent"]

    if agg not in _PIE_AGGS:
        return _err(f"agg='{agg}' is not allowed for pie_chart. Allowed: {sorted(_PIE_AGGS)}.")

    if category_col not in df.columns:
        return _err(f"category_col='{category_col}' is not a column.")

    if agg != "count":
        if value_col is None:
            return _err("value_col is required when agg is not 'count'.")
        if value_col not in df.columns:
            return _err(f"value_col='{value_col}' is not a column.")
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            return _err(f"value_col='{value_col}' is not numeric.")

    cols = [category_col] + ([value_col] if agg != "count" else [])
    work = df[cols].dropna(subset=[category_col])
    if agg != "count":
        work = work.dropna(subset=[value_col])

    if len(work) == 0:
        return _err(f"No usable rows for pie chart on '{category_col}'.")

    if agg == "count":
        s = work[category_col].value_counts()
    else:
        s = work.groupby(category_col)[value_col].agg(agg)

    s = s.sort_values(ascending=False)
    if len(s) > MAX_PIE_SLICES:
        top = s.head(MAX_PIE_SLICES)
        other_val = float(s.iloc[MAX_PIE_SLICES:].sum())
        x = [str(k) for k in top.index.tolist()] + ["Other"]
        y = [float(v) for v in top.values.tolist()] + [other_val]
    else:
        x = [str(k) for k in s.index.tolist()]
        y = [float(v) for v in s.values.tolist()]

    return ChartSpec(
        kind="pie",
        title=title,
        intent=intent,
        x=x,
        y=y,
        x_label=category_col,
        y_label=f"{agg.capitalize()}" + (f" of {value_col}" if agg != "count" else ""),
        x_display_type="category",
        y_display_type="count" if agg == "count" else "number",
        source_columns=cols,
        data_point_count=int(len(work)),
    )


def _box_stats(values: pd.Series) -> dict:
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lo = max(values.min(), q1 - 1.5 * iqr)
    hi = min(values.max(), q3 + 1.5 * iqr)
    outliers = values[(values < lo) | (values > hi)].tolist()
    return {
        "min": float(lo), "q1": float(q1), "median": float(median),
        "q3": float(q3), "max": float(hi),
        "outliers": [float(v) for v in outliers],
    }


def execute_box_plot(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    value_col = params["value_col"]
    group_col = params.get("group_col")
    title = params["title"]
    intent = params["intent"]

    if value_col not in df.columns:
        return _err(f"value_col='{value_col}' is not a column.")
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return _err(f"value_col='{value_col}' is not numeric. Box plots need a numeric value column.")

    values = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if len(values) < 5:
        return _err(f"'{value_col}' has only {len(values)} non-null values; need at least 5 for a meaningful box plot.")

    if group_col is None:
        return ChartSpec(
            kind="box",
            title=title,
            intent=intent,
            series=[{"name": value_col, **_box_stats(values)}],
            x_label=value_col,
            y_label="",
            x_display_type="category",
            y_display_type="number",
            source_columns=[value_col],
            data_point_count=int(len(values)),
        )

    if group_col not in df.columns:
        return _err(f"group_col='{group_col}' is not a column.")
    if df[group_col].nunique() > MAX_CATEGORIES:
        return _err(f"group_col='{group_col}' has {df[group_col].nunique()} unique values; max is {MAX_CATEGORIES}.")

    work = df[[value_col, group_col]].dropna()
    series_list: list[dict] = []
    for gv, sub in work.groupby(group_col):
        v = pd.to_numeric(sub[value_col], errors="coerce").dropna()
        if len(v) < 5:
            continue
        series_list.append({"name": str(gv), **_box_stats(v)})

    if not series_list:
        return _err(f"No group in '{group_col}' has ≥5 non-null values for '{value_col}'.")

    return ChartSpec(
        kind="box",
        title=title,
        intent=intent,
        series=series_list,
        x_label=group_col,
        y_label=value_col,
        x_display_type="category",
        y_display_type="number",
        source_columns=[value_col, group_col],
        data_point_count=int(len(work)),
    )


TOOL_EXECUTORS: dict[str, Callable[[pd.DataFrame, dict], ChartSpec | ToolError]] = {
    "frequency_bar_chart": execute_frequency_bar_chart,
    "aggregation_bar_chart": execute_aggregation_bar_chart,
    "histogram_chart": execute_histogram_chart,
    "scatter_chart": execute_scatter_chart,
    "line_chart": execute_line_chart,
    "pie_chart": execute_pie_chart,
    "box_plot": execute_box_plot,
}
