import pandas as pd
from chart_executor import execute_dual_axis_chart
from schemas import ChartSpec


def test_dual_axis_two_series():
    df = pd.DataFrame({"month": ["Jan", "Jan", "Feb", "Feb"], "rev": [10, 20, 30, 40], "rate": [0.1, 0.2, 0.3, 0.4]})
    res = execute_dual_axis_chart(df, {
        "x_col": "month", "bar_value_col": "rev", "line_value_col": "rate",
        "bar_agg": "sum", "line_agg": "mean", "title": "Rev & rate", "intent": "x",
    })
    assert isinstance(res, ChartSpec)
    assert res.kind == "dual_axis"
    assert res.y_label_secondary is not None
    assert len(res.series) == 2
    # series carry an axis/type hint for the frontend
    kinds = {s.get("type") for s in res.series}
    assert kinds == {"bar", "line"}
