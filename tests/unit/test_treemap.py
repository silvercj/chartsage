import pandas as pd
from chart_executor import execute_treemap_chart
from schemas import ChartSpec


def test_treemap_flat():
    df = pd.DataFrame({"cat": ["A", "A", "B", "C"], "rev": [10, 5, 20, 8]})
    res = execute_treemap_chart(df, {"category_col": "cat", "value_col": "rev", "agg": "sum", "title": "t", "intent": "x"})
    assert isinstance(res, ChartSpec)
    assert res.kind == "treemap"
    names = {n["name"] for n in res.nodes}
    assert names == {"A", "B", "C"}
    assert next(n for n in res.nodes if n["name"] == "A")["value"] == 15


def test_treemap_hierarchy():
    df = pd.DataFrame({"region": ["W", "W", "E"], "product": ["A", "B", "A"], "rev": [10, 20, 30]})
    res = execute_treemap_chart(df, {"category_col": "region", "subcategory_col": "product", "value_col": "rev", "agg": "sum", "title": "t", "intent": "x"})
    w = next(n for n in res.nodes if n["name"] == "W")
    assert {c["name"] for c in w["children"]} == {"A", "B"}
