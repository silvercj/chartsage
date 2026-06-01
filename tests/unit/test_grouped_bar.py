import pandas as pd
from chart_executor import execute_grouped_bar_chart
from schemas import ChartSpec


def test_grouped_bar_multi_series():
    df = pd.DataFrame({
        "region": ["W", "W", "E", "E"],
        "product": ["A", "B", "A", "B"],
        "rev": [10, 20, 30, 40],
    })
    res = execute_grouped_bar_chart(df, {
        "category_col": "region", "breakdown_col": "product",
        "value_col": "rev", "agg": "sum", "mode": "grouped",
        "title": "Rev by region × product", "intent": "x",
    })
    assert isinstance(res, ChartSpec)
    assert res.kind == "grouped_bar"
    assert res.stacked is False
    assert len(res.series) == 2          # one per product
    assert res.x == ["W", "E"] or set(res.x) == {"W", "E"}
