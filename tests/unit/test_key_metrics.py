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
