import pandas as pd
from chart_executor import (
    execute_frequency_bar_chart,
    execute_pie_chart,
    execute_treemap_chart,
)


def _country_df():
    # "United States" is in 3 of every 4 rows; raw value_counts would never see it as 3/4.
    return pd.DataFrame({"country": [
        "United States, India", "United States, UK", "India", "United States, Japan, India",
    ] * 30})


def test_frequency_bar_explodes_multi_value():
    df = _country_df()
    spec = execute_frequency_bar_chart(df, {"column": "country", "title": "Top countries", "intent": "x"})
    assert not hasattr(spec, "reason"), getattr(spec, "reason", "")   # not a ToolError
    pairs = dict(zip(spec.x, spec.y))
    assert pairs["United States"] == 90    # 3 per 4-row block * 30 blocks
    assert pairs["India"] == 90
    assert pairs["Japan"] == 30


def test_frequency_bar_single_value_unchanged():
    df = pd.DataFrame({"rating": ["TV-MA", "PG-13", "R", "TV-14"] * 25})
    spec = execute_frequency_bar_chart(df, {"column": "rating", "title": "Ratings", "intent": "x"})
    assert not hasattr(spec, "reason")
    assert dict(zip(spec.x, spec.y))["TV-MA"] == 25


def test_treemap_explodes_multi_value():
    df = _country_df()
    spec = execute_treemap_chart(
        df, {"category_col": "country", "value_col": "country", "agg": "count", "title": "t", "intent": "x"}
    )
    assert not hasattr(spec, "reason"), getattr(spec, "reason", "")   # not a ToolError
    values = {n["name"]: n["value"] for n in spec.nodes}
    assert values["United States"] == 90    # atom count, not raw combo count
    assert values["India"] == 90
    assert values["Japan"] == 30


def test_treemap_single_value_unchanged():
    df = pd.DataFrame({"region": ["W", "W", "E", "S"] * 25, "rev": [10, 5, 20, 8] * 25})
    spec = execute_treemap_chart(
        df, {"category_col": "region", "value_col": "rev", "agg": "sum", "title": "t", "intent": "x"}
    )
    assert not hasattr(spec, "reason")
    values = {n["name"]: n["value"] for n in spec.nodes}
    assert values["W"] == (10 + 5) * 25


def test_pie_explodes_multi_value():
    df = _country_df()
    spec = execute_pie_chart(
        df, {"category_col": "country", "agg": "count", "title": "t", "intent": "x"}
    )
    assert not hasattr(spec, "reason"), getattr(spec, "reason", "")   # not a ToolError
    pairs = dict(zip(spec.x, spec.y))
    assert pairs["United States"] == 90    # atom count; only 4 atoms so no "Other" rollup
    assert pairs["Japan"] == 30
    assert "Other" not in spec.x


def test_pie_single_value_unchanged():
    df = pd.DataFrame({"rating": ["TV-MA", "PG-13", "R", "TV-14"] * 25})
    spec = execute_pie_chart(
        df, {"category_col": "rating", "agg": "count", "title": "t", "intent": "x"}
    )
    assert not hasattr(spec, "reason")
    assert dict(zip(spec.x, spec.y))["TV-MA"] == 25
