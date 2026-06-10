"""The model can declare how a charted value displays (y_format: number/currency/percent).
Column-name keyword sniffing stays as the fallback, but the model's call wins — it knows
from context that a football goal 'margin' is a plain number, not a percentage, and that
'gross' in 'gross_margin_pct' isn't currency. Kills the keyword-collision bug class."""
import pandas as pd

from chart_executor import (
    execute_aggregation_bar_chart,
    execute_line_chart,
    execute_box_plot,
)
from chart_tools import CHART_TOOLS


def _margin_df():
    # 'margin' name-sniffs as a percentage — but here it's a goal margin (a count-like number).
    return pd.DataFrame({
        "team": ["a", "a", "b", "b", "c", "c"],
        "margin": [1.0, 3.0, 2.0, 4.0, 1.0, 2.0],
    })


def test_y_format_number_overrides_percentage_sniff():
    spec = execute_aggregation_bar_chart(_margin_df(), {
        "value_col": "margin", "group_col": "team", "agg": "mean",
        "y_format": "number", "title": "t", "intent": "i"})
    assert spec.y_display_type == "number"


def test_y_format_percent_overrides_plain_name():
    df = pd.DataFrame({"team": ["a", "a", "b", "b"], "wins": [0.4, 0.6, 0.5, 0.7]})
    spec = execute_aggregation_bar_chart(df, {
        "value_col": "wins", "group_col": "team", "agg": "mean",
        "y_format": "percent", "title": "t", "intent": "i"})
    assert spec.y_display_type == "percentage"


def test_omitted_y_format_keeps_name_inference():
    spec = execute_aggregation_bar_chart(_margin_df(), {
        "value_col": "margin", "group_col": "team", "agg": "mean",
        "title": "t", "intent": "i"})
    assert spec.y_display_type == "percentage"


def test_count_agg_always_displays_as_count():
    df = pd.DataFrame({"d": [f"2024-{m:02d}-01" for m in range(1, 7)], "v": [1.0] * 6})
    spec = execute_line_chart(df, {
        "date_col": "d", "value_col": "v", "agg": "count", "granularity": "month",
        "y_format": "currency", "title": "t", "intent": "i"})
    assert spec.y_display_type == "count"


def test_line_and_box_honor_y_format():
    df = pd.DataFrame({"d": [f"2024-{m:02d}-01" for m in range(1, 7)], "v": [1.0] * 6})
    line = execute_line_chart(df, {
        "date_col": "d", "value_col": "v", "agg": "sum", "granularity": "month",
        "y_format": "currency", "title": "t", "intent": "i"})
    assert line.y_display_type == "currency"

    box = execute_box_plot(_margin_df(), {
        "value_col": "margin", "y_format": "number", "title": "t", "intent": "i"})
    assert box.y_display_type == "number"


def test_value_charting_tools_expose_y_format():
    by_name = {t["name"]: t for t in CHART_TOOLS}
    for name in ("aggregation_bar_chart", "scatter_chart", "line_chart", "pie_chart",
                 "box_plot", "grouped_bar_chart", "dual_axis_chart", "treemap_chart",
                 "heatmap_chart"):
        props = by_name[name]["input_schema"]["properties"]
        assert "y_format" in props, f"{name} is missing y_format"
        assert "y_format" not in by_name[name]["input_schema"]["required"]
    # Pure count charts don't need it.
    for name in ("frequency_bar_chart", "histogram_chart"):
        assert "y_format" not in by_name[name]["input_schema"]["properties"]
