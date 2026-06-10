"""The narrative pass is the only model call that sees the charts' computed data, so it
curates the report: `chart_order` picks the hero and ranks the rest (layout was 'first 5
tool calls -> main'), and `drop_charts` removes charts whose data turned out to show
nothing. Captions move with their charts."""
import pandas as pd
from unittest.mock import MagicMock

from report_generator import ReportGenerator
from profile import profile_dataframe
from tests.helpers.fake_claude import FakeClaude, tool_use


def _make_generator(df, fake):
    client = MagicMock()
    client.messages_create = fake
    return ReportGenerator(
        profile=profile_dataframe(df), df=df, claude=client,
        model_selection="m1", model_narrative="m2",
    )


def _df():
    return pd.DataFrame({
        "activity_type": ["run", "run", "ride", "ride", "swim", "swim"] * 4,
        "duration": [30.0, 45.0, 60.0, 90.0, 25.0, 40.0] * 4,
    })


def _six_chart_calls():
    """Six charts with six distinct signatures (kind + columns), titled Chart 0..5."""
    calls = [
        tool_use("frequency_bar_chart", {"column": "activity_type"}),
        tool_use("aggregation_bar_chart",
                 {"value_col": "duration", "group_col": "activity_type", "agg": "mean"}),
        tool_use("histogram_chart", {"column": "duration"}),
        tool_use("box_plot", {"value_col": "duration", "group_col": "activity_type"}),
        tool_use("pie_chart", {"category_col": "activity_type", "agg": "count"}),
        tool_use("box_plot", {"value_col": "duration"}),
    ]
    for i, c in enumerate(calls):
        c["input"].update({"title": f"Chart {i}", "intent": f"i{i}"})
    return calls


def _narrative(captions_n, **extra):
    return tool_use("submit_narrative", {
        "summary": "s",
        "captions": [f"c{i}" for i in range(captions_n)],
        "data_quality": [],
        **extra,
    })


def test_chart_order_reorders_charts_and_captions():
    fake = FakeClaude([
        {"tool_calls": _six_chart_calls()[:5]},        # 5 charts (no dedup concerns: differing agg)
        {"tool_calls": []},                            # reach-for-more proposes nothing
        {"tool_calls": [_narrative(5, chart_order=[3, 1, 2, 4, 5])]},
    ])
    gen = _make_generator(_df(), fake)
    report = gen.build_report()
    assert [c.spec.title for c in report.charts] == \
        ["Chart 2", "Chart 0", "Chart 1", "Chart 3", "Chart 4"]
    assert [c.caption for c in report.charts] == ["c2", "c0", "c1", "c3", "c4"]
    # The hero (layout main, order 0) is the curated #1.
    main0 = next(e for e in report.layout if e.position == "main" and e.order == 0)
    assert main0.chart_id == report.charts[0].chart_id


def test_drop_charts_removes_duds():
    fake = FakeClaude([
        {"tool_calls": _six_chart_calls()},
        {"tool_calls": [_narrative(6, drop_charts=[2, 6])]},
    ])
    gen = _make_generator(_df(), fake)
    report = gen.build_report()
    titles = [c.spec.title for c in report.charts]
    assert titles == ["Chart 0", "Chart 2", "Chart 3", "Chart 4"]


def test_drops_never_reduce_report_below_three_charts():
    fake = FakeClaude([
        {"tool_calls": _six_chart_calls()[:4]},
        {"tool_calls": []},                            # reach-for-more proposes nothing
        {"tool_calls": [_narrative(4, drop_charts=[1, 2, 3, 4])]},
    ])
    gen = _make_generator(_df(), fake)
    report = gen.build_report()
    assert len(report.charts) == 3


def test_bogus_indices_are_ignored():
    fake = FakeClaude([
        {"tool_calls": _six_chart_calls()[:5]},
        {"tool_calls": []},
        {"tool_calls": [_narrative(5, chart_order=[99, 0, 2, 2], drop_charts=[42])]},
    ])
    gen = _make_generator(_df(), fake)
    report = gen.build_report()
    # Only the valid '2' applies; everything else keeps original order, nothing dropped.
    assert [c.spec.title for c in report.charts] == \
        ["Chart 1", "Chart 0", "Chart 2", "Chart 3", "Chart 4"]


def test_no_curation_fields_keeps_original_order():
    fake = FakeClaude([
        {"tool_calls": _six_chart_calls()[:5]},
        {"tool_calls": []},
        {"tool_calls": [_narrative(5)]},
    ])
    gen = _make_generator(_df(), fake)
    report = gen.build_report()
    assert [c.spec.title for c in report.charts] == [f"Chart {i}" for i in range(5)]
