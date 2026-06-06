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

    def add(result: Any) -> bool:
        """Append if it's a real chart; return True when we've hit max_charts."""
        if isinstance(result, ChartSpec):
            specs.append(result)
        return len(specs) >= max_charts

    # 1. ONE lead bar (top categorical) — capped at one to avoid a bar-heavy fallback.
    if cats:
        top_cat = cats[0]
        # A categorical whose every value is unique is a row label/key (one row each),
        # so a frequency bar of it is all 1s — a flat, useless chart (this is what made
        # the hurricanes-by-decade report render as a flat "decade — distribution" bar).
        # When numerics are present it's a "label + metrics" table, so chart the most
        # prominent metric BY the label instead — for one row per label the agg of a
        # single value is that value, giving the bar the data actually wants.
        if nums and top_cat.cardinality >= profile.row_count:
            lead = execute_aggregation_bar_chart(df, {
                "value_col": nums[0].name,
                "group_col": top_cat.name,
                "agg": "sum",
                "title": f"{nums[0].name} by {top_cat.name}",
                "intent": intent,
            })
        else:
            lead = execute_frequency_bar_chart(df, {
                "column": top_cat.name,
                "title": f"{top_cat.name} — distribution",
                "intent": intent,
            })
        if add(lead):
            return specs

    # 2. A box plot — top numeric across the top categorical (a different way to read the data).
    if nums and cats:
        if add(execute_box_plot(df, {
            "value_col": nums[0].name,
            "group_col": cats[0].name,
            "title": f"{nums[0].name} by {cats[0].name}",
            "intent": intent,
        })):
            return specs

    # 3. Histograms for the top 2 numerics.
    for col in nums[:2]:
        if add(execute_histogram_chart(df, {
            "column": col.name,
            "title": f"{col.name} — distribution",
            "intent": intent,
        })):
            return specs

    # 4. Strongest correlation as scatter (correlated pair or best available numeric pair).
    scatter_added = False
    if profile.correlations:
        best_pair, _ = max(profile.correlations.items(), key=lambda kv: abs(kv[1]))
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
            if pd.api.types.is_numeric_dtype(df[col])
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
