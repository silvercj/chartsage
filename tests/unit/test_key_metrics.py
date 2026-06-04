import pandas as pd
from chart_executor import execute_key_metrics
from schemas import KeyMetric, Report, ToolError

def test_keymetric_defaults_number():
    m = KeyMetric(label="Revenue", value=1240.5)
    assert m.format == "number"

def test_report_key_metrics_defaults_empty():
    r = Report(generated_at="t", summary="s", data_quality=[], charts=[])
    assert r.key_metrics == []

def test_key_metrics_computes_values():
    df = pd.DataFrame({"revenue": [100, 200, 300], "region": ["W", "E", "W"]})
    res = execute_key_metrics(df, {"metrics": [
        {"label": "Total revenue", "column": "revenue", "agg": "sum", "format": "currency"},
        {"label": "Regions", "column": "region", "agg": "nunique", "format": "number"},
    ]})
    assert isinstance(res, list)
    by_label = {m.label: m for m in res}
    assert by_label["Total revenue"].value == 600.0
    assert by_label["Total revenue"].format == "currency"
    assert by_label["Regions"].value == 2.0

def test_key_metrics_drops_invalid_and_errors_when_empty():
    df = pd.DataFrame({"revenue": [1, 2]})
    # bad column dropped; if none valid -> ToolError
    res = execute_key_metrics(df, {"metrics": [{"label": "X", "column": "nope", "agg": "sum"}]})
    assert isinstance(res, ToolError)


def test_key_metrics_filter_subsets_before_agg():
    # The host-advantage bug: a metric for one group must aggregate ONLY that
    # group's rows, not the whole column. Overall mean here is 2/6 = 0.333.
    df = pd.DataFrame({
        "home_win": [1, 1, 0, 0, 0, 0],
        "venue": ["Host nation", "Host nation", "Host nation",
                  "Neutral site", "Neutral site", "Neutral site"],
    })
    res = execute_key_metrics(df, {"metrics": [
        {"label": "Host win rate", "column": "home_win", "agg": "mean",
         "filter": {"column": "venue", "value": "Host nation"}, "format": "percent"},
        {"label": "Neutral win rate", "column": "home_win", "agg": "mean",
         "filter": {"column": "venue", "value": "Neutral site"}, "format": "percent"},
    ]})
    by = {m.label: m.value for m in res}
    assert round(by["Host win rate"], 3) == 0.667     # 2/3, NOT the 0.333 overall
    assert round(by["Neutral win rate"], 3) == 0.0    # 0/3


def test_key_metrics_filter_count_subsets():
    df = pd.DataFrame({"venue": ["Host nation"] * 3 + ["Neutral site"] * 7})
    res = execute_key_metrics(df, {"metrics": [
        {"label": "Host matches", "column": "venue", "agg": "count",
         "filter": {"column": "venue", "value": "Host nation"}},
    ]})
    assert res[0].value == 3.0   # NOT 10 (the whole-column count)


def test_key_metrics_filter_numeric_value_coerced():
    # LLM passes the filter value as a string; the column is int — must still match.
    df = pd.DataFrame({"year": [2018, 2018, 2022], "goals": [2, 4, 6]})
    res = execute_key_metrics(df, {"metrics": [
        {"label": "2022 goals", "column": "goals", "agg": "sum",
         "filter": {"column": "year", "value": "2022"}},
    ]})
    assert res[0].value == 6.0


def test_key_metrics_filter_no_match_dropped():
    df = pd.DataFrame({"home_win": [1, 0], "venue": ["Host nation", "Neutral site"]})
    res = execute_key_metrics(df, {"metrics": [
        {"label": "Mars win rate", "column": "home_win", "agg": "mean",
         "filter": {"column": "venue", "value": "Mars"}},     # matches nothing
        {"label": "Overall", "column": "home_win", "agg": "mean"},
    ]})
    by = {m.label: m.value for m in res}
    assert "Mars win rate" not in by      # dropped, not a misleading NaN/0
    assert by["Overall"] == 0.5


def test_key_metrics_filter_missing_column_dropped():
    df = pd.DataFrame({"home_win": [1, 0]})
    res = execute_key_metrics(df, {"metrics": [
        {"label": "Bad filter", "column": "home_win", "agg": "mean",
         "filter": {"column": "nope", "value": "x"}},
        {"label": "Overall", "column": "home_win", "agg": "mean"},
    ]})
    by = {m.label: m.value for m in res}
    assert "Bad filter" not in by
    assert by["Overall"] == 0.5
