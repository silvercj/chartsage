"""Heuristic chart picker used when Claude pass #1 produces fewer than 3 usable charts."""
from typing import Any
import pandas as pd
from schemas import ChartSpec, DataProfile
from chart_executor import (
    execute_frequency_bar_chart,
    execute_histogram_chart,
    execute_scatter_chart,
)


def pick_fallback_charts(profile: DataProfile, df: pd.DataFrame, max_charts: int = 5) -> list[ChartSpec]:
    """Pure heuristic chart selection.

    Rules:
    - frequency_bar_chart for top 2 categorical columns (cardinality 2–30)
    - histogram_chart for top 2 numeric columns
    - scatter_chart for the strongest |correlation| ≥ 0.3 pair
    """
    specs: list[ChartSpec] = []
    intent = "fallback: Claude pass #1 didn't pick this; chosen by heuristic."

    # 1. Top 2 categoricals
    cats = [c for c in profile.columns if c.role == "categorical" and 2 <= c.cardinality <= 30]
    for col in cats[:2]:
        result = execute_frequency_bar_chart(df, {
            "column": col.name,
            "title": f"{col.name} — distribution",
            "intent": intent,
        })
        if isinstance(result, ChartSpec):
            specs.append(result)
            if len(specs) >= max_charts:
                return specs

    # 2. Top 2 numerics
    nums = [c for c in profile.columns if c.role == "numeric"]
    for col in nums[:2]:
        result = execute_histogram_chart(df, {
            "column": col.name,
            "title": f"{col.name} — distribution",
            "intent": intent,
        })
        if isinstance(result, ChartSpec):
            specs.append(result)
            if len(specs) >= max_charts:
                return specs

    # 3. Strongest correlation as scatter (correlated pair or best available numeric pair)
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

    # 4. If still fewer than 3, try a scatter from any two numeric-dtype columns
    #    (including identifier columns that happen to be numeric)
    if not scatter_added and len(specs) < 3:
        numeric_dtype_cols = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col])
        ]
        if len(numeric_dtype_cols) >= 2:
            x_col, y_col = numeric_dtype_cols[0], numeric_dtype_cols[1]
            result = execute_scatter_chart(df, {
                "x_col": x_col,
                "y_col": y_col,
                "title": f"{x_col} vs {y_col}",
                "intent": intent,
            })
            if isinstance(result, ChartSpec):
                specs.append(result)

    return specs
