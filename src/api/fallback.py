"""Heuristic chart picker used when Claude pass #1 produces fewer than 3 usable charts.

Deliberately diversified: at most ONE frequency bar, then a box plot (distribution by
group), histograms, and a scatter — so the safety net never produces an all-bar report.
"""
from typing import Any
import pandas as pd
from schemas import ChartSpec, DataProfile
from chart_executor import (
    execute_frequency_bar_chart,
    execute_aggregation_bar_chart,
    execute_histogram_chart,
    execute_scatter_chart,
    execute_box_plot,
    _infer_display_type,
)


# Marker on the `intent` of any spec this heuristic safety net produced (vs. a chart
# the model selected). Analytics + tests key off this to compute the fallback rate.
FALLBACK_INTENT = "fallback: Claude pass #1 didn't pick this; chosen by heuristic."


def is_fallback_spec(spec) -> bool:
    """True when a chart came from the heuristic fallback rather than model selection."""
    return (getattr(spec, "intent", "") or "").lower().startswith("fallback")


def chart_composition(charts) -> dict:
    """Model-output composition for analytics, from a report's charts (each having a
    ``.spec``): how many charts the model picked vs. the heuristic fallback, plus the
    chart-kind mix. Lets us track the fallback rate and what the model actually makes."""
    n = len(charts)
    fb = sum(1 for c in charts if is_fallback_spec(c.spec))
    return {
        "chartCount": n,
        "fallbackChartCount": fb,
        "modelChartCount": n - fb,
        "usedFallback": fb > 0,
        "allFallback": n > 0 and fb == n,
        "fallbackRatio": round(fb / n, 3) if n else 0.0,
        "chartKinds": [c.spec.kind for c in charts],
    }


def chart_signature(spec) -> tuple:
    """A chart's identity for dedup: its kind + the set of columns it draws from. Two specs
    with the same signature render the same chart (regardless of title)."""
    return (getattr(spec, "kind", None), tuple(sorted(getattr(spec, "source_columns", None) or [])))


def drop_duplicates(existing, candidates) -> list:
    """Drop candidate specs that re-make a chart already in ``existing`` (same signature).
    The heuristic fallback re-derives charts from the same dataframe, so when it supplements
    a few model-selected charts it can duplicate them — keep the model's (better-titled) one."""
    seen = {chart_signature(s) for s in existing}
    kept = []
    for s in candidates:
        sig = chart_signature(s)
        if sig in seen:
            continue
        seen.add(sig)
        kept.append(s)
    return kept


# A column whose name IS one of these (e.g. "Year", "Decade") is a time/ordinal axis.
# Matched on the WHOLE name (letters only) — not as a substring — so a rate like
# "majors_per_year" or "goals_per_month", which merely *contains* a temporal word, is
# never mistaken for the axis (that bug made the hurricanes report flat-line).
_TEMPORAL_NAMES = {
    "year", "years", "yr", "date", "dates", "decade", "decades", "season", "seasons",
    "period", "periods", "quarter", "quarters", "month", "months", "week", "weeks",
}


def _ordinal_index(nums, df: pd.DataFrame) -> str | None:
    """A numeric column that's a temporal/ordinal index — a year/decade/… axis with one
    row per value (the x of a time series). Returns its name, or None. Lets the fallback
    chart a metric OVER time instead of histogramming the year itself."""
    for c in nums:
        norm = "".join(ch for ch in c.name.lower() if ch.isalpha())
        if norm not in _TEMPORAL_NAMES:
            continue
        s = pd.to_numeric(df[c.name], errors="coerce").dropna()
        if len(s) >= 4 and s.nunique() >= 0.9 * len(s):
            return c.name
    return None


def _timeseries_line(df: pd.DataFrame, idx_col: str, value_col: str, intent: str) -> ChartSpec:
    """A chronological line of `value_col` over the ordinal `idx_col` (e.g. a metric by year)."""
    work = df[[idx_col, value_col]].dropna().sort_values(idx_col)
    return ChartSpec(
        kind="line",
        title=f"{value_col} over {idx_col}",
        intent=intent,
        x=[str(v) for v in work[idx_col].tolist()],
        y=[float(v) for v in work[value_col].tolist()],
        x_label=idx_col,
        y_label=value_col,
        x_display_type="category",
        y_display_type=_infer_display_type(value_col),
        source_columns=[idx_col, value_col],
        data_point_count=int(len(work)),
    )


def pick_fallback_charts(profile: DataProfile, df: pd.DataFrame, max_charts: int = 5) -> list[ChartSpec]:
    """Pure heuristic chart selection, biased toward VARIETY rather than bars.

    Order (each capped, stops at max_charts):
    1. one frequency_bar_chart for the top categorical column (2–30 unique)
    2. one box_plot of the top numeric across that categorical (distribution by group)
    3. histogram_chart for the top 2 numeric columns
    4. scatter_chart for the strongest |correlation| ≥ 0.3 pair (or any numeric pair)
    """
    specs: list[ChartSpec] = []
    intent = FALLBACK_INTENT

    cats = [c for c in profile.columns if c.role == "categorical" and 2 <= c.cardinality <= 30]
    nums = [c for c in profile.columns if c.role == "numeric"]
    idx_name = _ordinal_index(nums, df)                 # a year/ordinal time axis, if any
    metrics = [c for c in nums if c.name != idx_name]   # numerics minus the time axis

    def add(result: Any) -> bool:
        """Append if it's a real chart; return True when we've hit max_charts."""
        if isinstance(result, ChartSpec):
            specs.append(result)
        return len(specs) >= max_charts

    # 1. ONE lead chart — the shape the table actually wants.
    lead = None
    if idx_name and metrics:
        # Time-series table (a year/ordinal axis + metrics): chart the metric OVER time,
        # not a histogram of the year. Fixes "Blowout_pct by year" rendering as a flat
        # "year — distribution" histogram when the model under-picks (World Cup blowouts).
        lead = _timeseries_line(df, idx_name, metrics[0].name, intent)
    elif cats:
        top_cat = cats[0]
        # A categorical whose every value is unique is a row label/key (one row each), so a
        # frequency bar of it is all 1s — a flat, useless chart (this is what made the
        # hurricanes-by-decade report render as a flat "decade — distribution" bar). When
        # numerics are present it's a "label + metrics" table, so chart the most prominent
        # metric BY the label instead — for one row per label the agg is that value.
        if metrics and top_cat.cardinality >= profile.row_count:
            lead = execute_aggregation_bar_chart(df, {
                "value_col": metrics[0].name,
                "group_col": top_cat.name,
                "agg": "sum",
                "title": f"{metrics[0].name} by {top_cat.name}",
                "intent": intent,
            })
        else:
            lead = execute_frequency_bar_chart(df, {
                "column": top_cat.name,
                "title": f"{top_cat.name} — distribution",
                "intent": intent,
            })
    if lead is not None and add(lead):
        return specs

    # 1b. A second trend line — the next metric over time (time-series tables only), so an
    #     all-fallback report leads with TWO trends rather than a trend + a weak histogram.
    if idx_name and len(metrics) >= 2:
        if add(_timeseries_line(df, idx_name, metrics[1].name, intent)):
            return specs

    # 2. A box plot — top metric across the top categorical (a different way to read the data).
    if metrics and cats:
        if add(execute_box_plot(df, {
            "value_col": metrics[0].name,
            "group_col": cats[0].name,
            "title": f"{metrics[0].name} by {cats[0].name}",
            "intent": intent,
        })):
            return specs

    # 3. Histograms — the top 2 metrics, skipping any already drawn as trend lines above.
    hist_metrics = metrics[2:] if idx_name else metrics[:2]
    for col in hist_metrics[:2]:
        if add(execute_histogram_chart(df, {
            "column": col.name,
            "title": f"{col.name} — distribution",
            "intent": intent,
        })):
            return specs

    # 4. Strongest correlation as scatter — among metrics, skipping the time axis (so we
    #    don't reduce a trend to a generic "year vs matches" scatter).
    scatter_added = False
    metric_corrs = {k: v for k, v in (profile.correlations or {}).items()
                    if idx_name not in k.split("||")}
    if metric_corrs:
        best_pair, _ = max(metric_corrs.items(), key=lambda kv: abs(kv[1]))
        x_col, y_col = best_pair.split("||")
        result = execute_scatter_chart(df, {
            "x_col": x_col,
            "y_col": y_col,
            "title": f"{x_col} vs {y_col}",
            "intent": intent,
        })
        if isinstance(result, ChartSpec):
            specs.append(result)
            scatter_added = True
            if len(specs) >= max_charts:
                return specs

    # 5. If still fewer than 3, try a scatter from any two numeric-dtype columns.
    if not scatter_added and len(specs) < 3:
        numeric_dtype_cols = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and col != idx_name
        ]
        if len(numeric_dtype_cols) >= 2:
            x_col, y_col = numeric_dtype_cols[0], numeric_dtype_cols[1]
            add(execute_scatter_chart(df, {
                "x_col": x_col,
                "y_col": y_col,
                "title": f"{x_col} vs {y_col}",
                "intent": intent,
            }))

    return specs
