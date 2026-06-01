"""Request-a-chart: ReportGenerator.add_chart focused single-chart selection.

Two modes:
  - "type": force a specific tool (the chart_type), model picks columns.
  - "describe": let the model choose the tool from a free-text prompt.

Returns a ChartWithCaption, or None when no spec is produced (the endpoint
turns None into a 422 with no debit).
"""
import types
from report_generator import ReportGenerator
from profile import profile_dataframe
from tests.helpers.fake_claude import FakeClaude, tool_use


def test_add_chart_pick_type_returns_one(sales):
    # sales_df numeric columns are order_id (15 unique) and revenue — a valid scatter.
    fc = FakeClaude([
        {"tool_calls": [tool_use("scatter_chart", {
            "x_col": "order_id", "y_col": "revenue", "title": "U vs R", "intent": "i"})]},
    ])
    gen = ReportGenerator(
        profile=profile_dataframe(sales), df=sales,
        claude=types.SimpleNamespace(messages_create=fc),
        model_selection="m", model_narrative="m",
    )
    cwc = gen.add_chart(mode="type", chart_type="scatter_chart", prompt=None)
    assert cwc is not None and cwc.spec.kind == "scatter"


def test_add_chart_no_chart_returns_none(sales):
    fc = FakeClaude([{"tool_calls": []}])
    gen = ReportGenerator(
        profile=profile_dataframe(sales), df=sales,
        claude=types.SimpleNamespace(messages_create=fc),
        model_selection="m", model_narrative="m",
    )
    assert gen.add_chart(mode="describe", chart_type=None, prompt="show me x") is None
