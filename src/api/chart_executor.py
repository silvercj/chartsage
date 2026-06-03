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
from schemas import ChartSpec, KeyMetric, ToolError
from multi_value import detect_multi_value, explode_multi_value


MAX_CATEGORIES = 30
MAX_PIE_SLICES = 8
MAX_SCATTER_POINTS = 5000
MIN_SCATTER_X_CARDINALITY = 15  # below this, scatter degenerates to vertical stripes
                                # (Superstore discount has 11 unique values — must be excluded)

_CURRENCY_KEYWORDS = (
    "revenue", "sales", "price", "cost", "amount", "profit",
    "fee", "expense", "income", "payment", "salary", "wage",
    "gross", "net", "dollar", "usd", "earnings", "spend",
)
_PERCENTAGE_KEYWORDS = ("rate", "ratio", "percent", "pct", "share", "margin")


def _err(reason: str) -> ToolError:
    return ToolError(reason=reason)


def _available_columns_by_role(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group columns by inferred role for error messages."""
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if df[c].dtype == "object" and df[c].nunique(dropna=True) <= 50]
    return {"numeric": numeric, "categorical": categorical, "all": list(df.columns)}


def _infer_display_type(col_name: str, agg: str = None) -> str:
    """Pick y_display_type from a value column name + aggregation.

    Counts always display as 'count'. Otherwise we sniff for currency
    or percentage keywords in the column name; default is 'number'.
    """
    if agg == "count":
        return "count"
    lower = col_name.lower()
    if any(kw in lower for kw in _CURRENCY_KEYWORDS):
        return "currency"
    if any(kw in lower for kw in _PERCENTAGE_KEYWORDS):
        return "percentage"
    return "number"


def execute_frequency_bar_chart(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    column = params["column"]
    title = params["title"]
    intent = params["intent"]

    if column not in df.columns:
        cats = _available_columns_by_role(df)["categorical"]
        return _err(f"'{column}' is not a column. Available categorical columns: {cats}")

    if df[column].dropna().empty:
        return _err(f"'{column}' has no non-null values.")

    delim = detect_multi_value(df[column])
    if delim is not None:
        series = explode_multi_value(df[column], delim)
        counts = series.value_counts().head(MAX_CATEGORIES)   # top-N atoms; no cardinality error
    else:
        series = df[column].dropna()
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
        y_display_type=_infer_display_type(value_col, agg),
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

    if x_col == y_col:
        return _err(f"x_col and y_col are both '{x_col}'; a scatter needs two different numeric columns.")

    # Reject scatter when x is a low-cardinality numeric (e.g., discount ∈ {0, 0.1, 0.2, ...}).
    # On such data the scatter degenerates to vertical stripes; box_plot or aggregation_bar_chart shows the same insight clearly.
    x_card = int(df[x_col].dropna().nunique())
    if x_card < MIN_SCATTER_X_CARDINALITY:
        return _err(
            f"x_col='{x_col}' has only {x_card} unique values; scatter degenerates to vertical stripes. "
            f"Use box_plot (value_col='{y_col}', group_col='{x_col}') or aggregation_bar_chart "
            f"(value_col='{y_col}', group_col='{x_col}', agg='mean') instead."
        )

    cols = [x_col, y_col]
    if color_by is not None:
        if color_by not in df.columns:
            return _err(f"color_by='{color_by}' is not a column.")
        if pd.api.types.is_numeric_dtype(df[color_by]):
            return _err(f"color_by='{color_by}' is numeric or high-cardinality; pass a categorical column with ≤{MAX_CATEGORIES} groups.")
        if df[color_by].nunique() > MAX_CATEGORIES:
            return _err(f"color_by='{color_by}' has {df[color_by].nunique()} unique values; max is {MAX_CATEGORIES}.")
        if color_by not in cols:
            cols.append(color_by)

    work = df[cols].dropna()
    if len(work) == 0:
        return _err(f"No rows where both '{x_col}' and '{y_col}' are non-null.")

    if len(work) > MAX_SCATTER_POINTS:
        work = work.sample(n=MAX_SCATTER_POINTS, random_state=42)

    y_display = _infer_display_type(y_col)

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
            y_display_type=y_display,
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
        y_display_type=y_display,
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
    area = params.get("area", False)
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
            y_display_type=_infer_display_type(value_col, agg) if value_col else "count",
            area=area,
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
        y_display_type=_infer_display_type(value_col, agg) if value_col else "count",
        area=area,
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
        if value_col == category_col:
            return _err(
                f"category_col and value_col must be different columns (both are '{category_col}')."
            )

    cols = [category_col] + ([value_col] if agg != "count" else [])
    work = df[cols].dropna(subset=[category_col])
    if agg != "count":
        work = work.dropna(subset=[value_col])

    if len(work) == 0:
        return _err(f"No usable rows for pie chart on '{category_col}'.")

    cat_delim = detect_multi_value(df[category_col])
    if cat_delim is not None:
        # Multi-value category: split each cell into atoms (one row per atom).
        # The existing top-MAX_PIE_SLICES + "Other" rollup then runs on atom counts/values.
        work = work.assign(
            **{category_col: work[category_col].astype(str).str.split(cat_delim, regex=False)}
        ).explode(category_col)
        work[category_col] = work[category_col].str.strip()
        work = work[work[category_col] != ""]
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
        y_display_type=_infer_display_type(value_col, agg) if value_col else "count",
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
            y_display_type=_infer_display_type(value_col),
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
        y_display_type=_infer_display_type(value_col),
        source_columns=[value_col, group_col],
        data_point_count=int(len(work)),
    )


_HEATMAP_AGGS = {"sum", "mean", "count"}


def execute_heatmap_chart(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    mode = params["mode"]
    title = params["title"]
    intent = params["intent"]

    if mode == "correlation":
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) < 2:
            return _err(f"Correlation heatmap needs ≥2 numeric columns; found {len(numeric_cols)}.")
        corr = df[numeric_cols].corr(numeric_only=True).round(3)
        series = []
        for i, row in enumerate(numeric_cols):
            for j, col in enumerate(numeric_cols):
                v = corr.loc[row, col]
                if pd.notna(v):
                    series.append({"row": row, "col": col, "value": float(v)})
        return ChartSpec(
            kind="heatmap",
            title=title,
            intent=intent,
            x=numeric_cols,
            y=numeric_cols,
            series=series,
            x_label="",
            y_label="",
            x_display_type="category",
            y_display_type="number",
            source_columns=numeric_cols,
            data_point_count=int(df.shape[0]),
        )

    if mode != "pivot":
        return _err(f"mode='{mode}' is not allowed. Allowed: 'correlation', 'pivot'.")

    row_col = params.get("row_col")
    col_col = params.get("col_col")
    agg = params.get("agg", "count")
    value_col = params.get("value_col")

    if agg not in _HEATMAP_AGGS:
        return _err(f"agg='{agg}' is not allowed for pivot. Allowed: {sorted(_HEATMAP_AGGS)}.")
    if not row_col or row_col not in df.columns:
        return _err(f"row_col='{row_col}' is not a column.")
    if not col_col or col_col not in df.columns:
        return _err(f"col_col='{col_col}' is not a column.")
    if agg != "count":
        if not value_col:
            return _err("value_col is required when agg is not 'count'.")
        if value_col not in df.columns:
            return _err(f"value_col='{value_col}' is not a column.")
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            return _err(f"value_col='{value_col}' is not numeric.")

    cols_needed = [row_col, col_col] + ([value_col] if agg != "count" else [])
    work = df[cols_needed].dropna()

    if agg == "count":
        pivot = work.groupby([row_col, col_col]).size().unstack(fill_value=0)
    else:
        pivot = work.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc=agg, fill_value=0)

    if pivot.shape[0] > MAX_CATEGORIES or pivot.shape[1] > MAX_CATEGORIES:
        return _err(f"Heatmap dimensions {pivot.shape} exceed {MAX_CATEGORIES} × {MAX_CATEGORIES} max.")

    series = []
    for r in pivot.index:
        for c in pivot.columns:
            series.append({"row": str(r), "col": str(c), "value": float(pivot.loc[r, c])})

    return ChartSpec(
        kind="heatmap",
        title=title,
        intent=intent,
        x=[str(c) for c in pivot.columns.tolist()],
        y=[str(r) for r in pivot.index.tolist()],
        series=series,
        x_label=col_col,
        y_label=row_col,
        x_display_type="category",
        y_display_type=_infer_display_type(value_col, agg) if value_col else "count",
        source_columns=cols_needed,
        data_point_count=int(len(work)),
    )


_GROUPED_BAR_AGGS = {"sum", "mean", "median", "min", "max"}
MAX_GROUPED_BREAKDOWN = 6  # multi-series bars stay legible only with a small breakdown


def execute_grouped_bar_chart(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    category_col = params["category_col"]
    breakdown_col = params["breakdown_col"]
    value_col = params["value_col"]
    agg = params["agg"]
    mode = params["mode"]
    title = params["title"]
    intent = params["intent"]

    if agg not in _GROUPED_BAR_AGGS:
        return _err(f"agg='{agg}' is not allowed. Allowed: {sorted(_GROUPED_BAR_AGGS)}.")

    avail = _available_columns_by_role(df)
    if category_col not in df.columns:
        return _err(f"category_col='{category_col}' is not a column. Available categorical columns: {avail['categorical']}")
    if breakdown_col not in df.columns:
        return _err(f"breakdown_col='{breakdown_col}' is not a column. Available categorical columns: {avail['categorical']}")
    if value_col not in df.columns:
        return _err(f"value_col='{value_col}' is not a column. Available numeric columns: {avail['numeric']}")
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return _err(f"value_col='{value_col}' is not numeric. Available numeric columns: {avail['numeric']}")

    breakdowns = int(df[breakdown_col].dropna().nunique())
    if breakdowns == 0:
        return _err(f"breakdown_col='{breakdown_col}' has no non-null values.")
    if breakdowns > MAX_GROUPED_BREAKDOWN:
        return _err(
            f"breakdown_col='{breakdown_col}' has {breakdowns} unique values, more than the max ({MAX_GROUPED_BREAKDOWN}) "
            f"for a grouped/stacked bar. Use aggregation_bar_chart (value_col='{value_col}', group_col='{breakdown_col}', "
            f"agg='{agg}') instead, or pick a lower-cardinality breakdown."
        )

    categories = int(df[category_col].dropna().nunique())
    if categories == 0:
        return _err(f"category_col='{category_col}' has no non-null values.")
    if categories > MAX_CATEGORIES:
        return _err(f"category_col='{category_col}' has {categories} unique values, more than max ({MAX_CATEGORIES}).")

    work = df[[category_col, breakdown_col, value_col]].dropna()
    if len(work) == 0:
        return _err(f"No rows where '{category_col}', '{breakdown_col}', and '{value_col}' are all non-null.")

    pivot = work.pivot_table(
        index=category_col, columns=breakdown_col, values=value_col, aggfunc=agg
    )
    # x order: categories by total value descending (matches aggregation_bar_chart).
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    x = [str(c) for c in pivot.index.tolist()]
    series_list: list[dict] = []
    for breakdown_value in pivot.columns.tolist():
        col = pivot[breakdown_value]
        data = [None if pd.isna(v) else float(v) for v in col.tolist()]
        series_list.append({"name": str(breakdown_value), "data": data})

    return ChartSpec(
        kind="grouped_bar",
        title=title,
        intent=intent,
        x=x,
        series=series_list,
        x_label=category_col,
        y_label=f"{agg.capitalize()} of {value_col}",
        x_display_type="category",
        y_display_type=_infer_display_type(value_col, agg),
        stacked=(mode == "stacked"),
        source_columns=[category_col, breakdown_col, value_col],
        data_point_count=int(len(work)),
    )


_DUAL_AXIS_AGGS = {"sum", "mean", "median", "min", "max", "count"}


def execute_dual_axis_chart(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    x_col = params["x_col"]
    bar_value_col = params["bar_value_col"]
    line_value_col = params["line_value_col"]
    bar_agg = params["bar_agg"]
    line_agg = params["line_agg"]
    title = params["title"]
    intent = params["intent"]

    if bar_agg not in _DUAL_AXIS_AGGS:
        return _err(f"bar_agg='{bar_agg}' is not allowed. Allowed: {sorted(_DUAL_AXIS_AGGS)}.")
    if line_agg not in _DUAL_AXIS_AGGS:
        return _err(f"line_agg='{line_agg}' is not allowed. Allowed: {sorted(_DUAL_AXIS_AGGS)}.")

    avail = _available_columns_by_role(df)
    if x_col not in df.columns:
        return _err(f"x_col='{x_col}' is not a column. Available categorical columns: {avail['categorical']}")
    if bar_value_col not in df.columns:
        return _err(f"bar_value_col='{bar_value_col}' is not a column. Available numeric columns: {avail['numeric']}")
    if line_value_col not in df.columns:
        return _err(f"line_value_col='{line_value_col}' is not a column. Available numeric columns: {avail['numeric']}")

    # Value columns must be numeric unless their aggregation is a count.
    if bar_agg != "count" and not pd.api.types.is_numeric_dtype(df[bar_value_col]):
        return _err(f"bar_value_col='{bar_value_col}' is not numeric (required when bar_agg != 'count'). "
                    f"Available numeric columns: {avail['numeric']}")
    if line_agg != "count" and not pd.api.types.is_numeric_dtype(df[line_value_col]):
        return _err(f"line_value_col='{line_value_col}' is not numeric (required when line_agg != 'count'). "
                    f"Available numeric columns: {avail['numeric']}")

    categories = int(df[x_col].dropna().nunique())
    if categories == 0:
        return _err(f"x_col='{x_col}' has no non-null values.")
    if categories > MAX_CATEGORIES:
        return _err(f"x_col='{x_col}' has {categories} unique values, more than max ({MAX_CATEGORIES}).")

    cols = [x_col] + list({bar_value_col, line_value_col})
    work = df[cols].dropna(subset=[x_col])
    if len(work) == 0:
        return _err(f"No rows where '{x_col}' is non-null.")

    def _agg_col(value_col: str, agg: str) -> pd.Series:
        if agg == "count":
            return work.groupby(x_col)[value_col].count()
        sub = work[[x_col, value_col]].dropna(subset=[value_col])
        return sub.groupby(x_col)[value_col].agg(agg)

    bar_series = _agg_col(bar_value_col, bar_agg)
    line_series = _agg_col(line_value_col, line_agg)

    # Shared x: union of categories present in either aggregation, bar order first.
    x_index = list(bar_series.index)
    for k in line_series.index:
        if k not in bar_series.index:
            x_index.append(k)

    x = [str(k) for k in x_index]
    bar_data = [None if pd.isna(bar_series.get(k)) else float(bar_series.get(k)) for k in x_index]
    line_data = [None if pd.isna(line_series.get(k)) else float(line_series.get(k)) for k in x_index]

    series_list = [
        {"name": bar_value_col, "type": "bar", "yAxisIndex": 0, "data": bar_data},
        {"name": line_value_col, "type": "line", "yAxisIndex": 1, "data": line_data},
    ]

    return ChartSpec(
        kind="dual_axis",
        title=title,
        intent=intent,
        x=x,
        series=series_list,
        x_label=x_col,
        y_label=bar_value_col,
        y_label_secondary=line_value_col,
        x_display_type="category",
        y_display_type=_infer_display_type(bar_value_col, bar_agg),
        source_columns=cols,
        data_point_count=int(len(work)),
    )


_TREEMAP_AGGS = {"sum", "mean", "count"}


def execute_treemap_chart(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    category_col = params["category_col"]
    subcategory_col = params.get("subcategory_col")
    value_col = params["value_col"]
    agg = params["agg"]
    title = params["title"]
    intent = params["intent"]

    if agg not in _TREEMAP_AGGS:
        return _err(f"agg='{agg}' is not allowed for treemap_chart. Allowed: {sorted(_TREEMAP_AGGS)}.")

    avail = _available_columns_by_role(df)
    if category_col not in df.columns:
        return _err(f"category_col='{category_col}' is not a column. Available categorical columns: {avail['categorical']}")
    if subcategory_col is not None and subcategory_col not in df.columns:
        return _err(f"subcategory_col='{subcategory_col}' is not a column. Available categorical columns: {avail['categorical']}")
    if value_col not in df.columns:
        return _err(f"value_col='{value_col}' is not a column. Available numeric columns: {avail['numeric']}")
    if agg != "count" and not pd.api.types.is_numeric_dtype(df[value_col]):
        return _err(f"value_col='{value_col}' is not numeric (required when agg != 'count'). "
                    f"Available numeric columns: {avail['numeric']}")

    if df[category_col].dropna().empty:
        return _err(f"category_col='{category_col}' has no non-null values.")

    cat_delim = detect_multi_value(df[category_col])
    if cat_delim is None:
        categories = int(df[category_col].dropna().nunique())
        if categories > MAX_CATEGORIES:
            return _err(f"category_col='{category_col}' has {categories} unique values, more than max ({MAX_CATEGORIES}).")

    group_cols = [category_col] + ([subcategory_col] if subcategory_col else [])
    subset = group_cols + ([value_col] if agg != "count" else [])
    work = df[list(dict.fromkeys(subset))].dropna(subset=group_cols)
    if agg != "count":
        work = work.dropna(subset=[value_col])
    if len(work) == 0:
        return _err(f"No usable rows for treemap on '{category_col}'.")

    if cat_delim is not None:
        # Multi-value category: split each cell into atoms (one row per atom),
        # then keep the top MAX_CATEGORIES atoms by aggregated value.
        work = work.assign(
            **{category_col: work[category_col].astype(str).str.split(cat_delim, regex=False)}
        ).explode(category_col)
        work[category_col] = work[category_col].str.strip()
        work = work[work[category_col] != ""]
        if len(work) == 0:
            return _err(f"No usable rows for treemap on '{category_col}'.")

    def _agg(g: pd.DataFrame) -> float:
        if agg == "count":
            return float(len(g))
        return float(g[value_col].agg(agg))

    if cat_delim is not None:
        # Keep the top MAX_CATEGORIES atoms by their (top-level) aggregated value.
        if agg == "count":
            totals = work[category_col].value_counts()
        else:
            totals = work.groupby(category_col)[value_col].agg(agg).sort_values(ascending=False)
        top_atoms = set(totals.head(MAX_CATEGORIES).index)
        work = work[work[category_col].isin(top_atoms)]

    nodes: list[dict] = []
    if subcategory_col:
        # 2-level hierarchy: parent value is the sum of its children's aggregated values.
        parents: list[tuple[str, float, list[dict]]] = []
        for parent_value, parent_df in work.groupby(category_col):
            children: list[dict] = []
            for child_value, child_df in parent_df.groupby(subcategory_col):
                children.append({"name": str(child_value), "value": _agg(child_df)})
            total = float(sum(c["value"] for c in children))
            children.sort(key=lambda c: c["value"], reverse=True)
            parents.append((str(parent_value), total, children))
        parents.sort(key=lambda p: p[1], reverse=True)
        nodes = [{"name": name, "value": total, "children": children} for name, total, children in parents]
    else:
        flat: list[dict] = []
        for cat_value, cat_df in work.groupby(category_col):
            flat.append({"name": str(cat_value), "value": _agg(cat_df)})
        flat.sort(key=lambda n: n["value"], reverse=True)
        nodes = flat

    def _all_values(ns: list[dict]):
        for n in ns:
            yield n["value"]
            if n.get("children"):
                yield from _all_values(n["children"])

    if any(v < 0 for v in _all_values(nodes)):
        return _err(
            f"treemap_chart can't show negative values ('{value_col}' produced a negative {agg}). "
            f"Use aggregation_bar_chart instead, which can show negatives."
        )

    return ChartSpec(
        kind="treemap",
        title=title,
        intent=intent,
        nodes=nodes,
        x_label=category_col,
        y_label=(f"{agg.capitalize()}" if agg == "count" else f"{agg.capitalize()} of {value_col}"),
        x_display_type="category",
        y_display_type=_infer_display_type(value_col, agg),
        source_columns=group_cols + ([value_col] if agg != "count" else []),
        data_point_count=int(len(work)),
    )


def execute_key_metrics(df: pd.DataFrame, params: dict) -> list[KeyMetric] | ToolError:
    out: list[KeyMetric] = []
    for m in (params.get("metrics") or [])[:5]:
        col, agg = m.get("column"), m.get("agg")
        label, fmt = m.get("label") or col, m.get("format", "number")
        if col not in df.columns:
            continue
        s = df[col]
        try:
            if agg == "count":
                val = float(s.dropna().shape[0])
            elif agg == "nunique":
                val = float(s.dropna().nunique())
            else:  # sum/mean/median/min/max
                nums = pd.to_numeric(s, errors="coerce").dropna()
                if nums.empty:
                    continue
                val = float(getattr(nums, agg)())
        except Exception:
            continue
        if fmt not in ("number", "currency", "percent"):
            fmt = "number"
        out.append(KeyMetric(label=str(label)[:60], value=val, format=fmt))
    if not out:
        return _err("no valid metrics could be computed")
    return out


# NOTE: execute_key_metrics is intentionally NOT registered below — it returns
# list[KeyMetric] | ToolError (not a ChartSpec) and is routed by name in
# report_generator._execute_tool_calls so it never counts as a chart.
TOOL_EXECUTORS: dict[str, Callable[[pd.DataFrame, dict], ChartSpec | ToolError]] = {
    "frequency_bar_chart": execute_frequency_bar_chart,
    "aggregation_bar_chart": execute_aggregation_bar_chart,
    "histogram_chart": execute_histogram_chart,
    "scatter_chart": execute_scatter_chart,
    "line_chart": execute_line_chart,
    "pie_chart": execute_pie_chart,
    "box_plot": execute_box_plot,
    "heatmap_chart": execute_heatmap_chart,
    "grouped_bar_chart": execute_grouped_bar_chart,
    "dual_axis_chart": execute_dual_axis_chart,
    "treemap_chart": execute_treemap_chart,
}
