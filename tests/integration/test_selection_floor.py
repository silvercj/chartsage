"""Min-charts 'reach for more' selection retry (Fix 1 for the under-selection bug).

When pass-1 selection under-selects (fewer than MIN_CHARTS_TARGET charts) with NO
execution errors, generate_charts() must fire one extra _call_selection_more round to
push toward MAX_CHARTS — but only when there's headroom. The fallback floor stays the
last resort. A healthy selection (>= MIN_CHARTS_TARGET) must NOT trigger the extra call.
"""
import types

from report_generator import ReportGenerator, MIN_CHARTS_TARGET, MAX_CHARTS
from profile import profile_dataframe
from tests.helpers.fake_claude import FakeClaude, tool_use, ten_distinct_chart_calls


def _make_generator(df, fake):
    profile = profile_dataframe(df)
    client = types.SimpleNamespace(messages_create=fake)
    return ReportGenerator(profile=profile, df=df, claude=client,
                           model_selection="m1", model_narrative="m2")


def test_under_selection_triggers_reach_for_more(sales):
    """Initial selection returns 3 valid charts (no errors). Because 3 < MIN_CHARTS_TARGET
    and there's headroom, _call_selection_more fires once and returns 4 more distinct
    charts -> 7 total. No fallback is needed (7 >= MIN_CHARTS_FOR_NO_FALLBACK)."""
    fake = FakeClaude([
        # Initial selection: 3 valid charts, no errors.
        {"tool_calls": [
            tool_use("frequency_bar_chart", {"column": "region", "title": "Region freq", "intent": "i"}, id_="a1"),
            tool_use("histogram_chart", {"column": "revenue", "title": "Revenue dist", "intent": "i"}, id_="a2"),
            tool_use("aggregation_bar_chart", {"value_col": "revenue", "group_col": "region",
                                               "agg": "mean", "title": "Mean rev by region", "intent": "i"}, id_="a3"),
        ]},
        # Reach-for-more round: 4 ADDITIONAL distinct charts.
        {"tool_calls": [
            tool_use("line_chart", {"date_col": "order_date", "value_col": "revenue", "agg": "sum",
                                    "granularity": "week", "title": "Revenue over time", "intent": "i"}, id_="b1"),
            tool_use("pie_chart", {"category_col": "region", "agg": "count",
                                   "title": "Region share", "intent": "i"}, id_="b2"),
            tool_use("scatter_chart", {"x_col": "order_id", "y_col": "revenue",
                                       "title": "Order id vs revenue", "intent": "i"}, id_="b3"),
            tool_use("treemap_chart", {"category_col": "region", "value_col": "revenue", "agg": "sum",
                                       "title": "Revenue treemap", "intent": "i"}, id_="b4"),
        ]},
    ])
    gen = _make_generator(sales, fake)
    specs = gen.generate_charts()
    assert len(specs) == 7
    assert len(fake.calls) == 2   # initial + one reach-for-more


def test_healthy_selection_does_not_trigger_reach_for_more(sales):
    """Initial selection returns 8 valid charts (>= MIN_CHARTS_TARGET). _call_selection_more
    must NOT be called: the count stays 8 (all distinct signatures) and FakeClaude's single
    scripted response is the only one consumed (a second call would raise AssertionError)."""
    calls = ten_distinct_chart_calls()[:8]
    fake = FakeClaude([{"tool_calls": calls}])   # only ONE scripted response
    gen = _make_generator(sales, fake)
    specs = gen.generate_charts()
    assert len(specs) == 8
    assert len(fake.calls) == 1   # no reach-for-more call was made


def test_reach_for_more_dedupes_and_caps(sales):
    """_call_selection_more must drop charts that repeat an existing angle
    (same kind + same source columns) and never exceed the remaining room."""
    existing = [
        # bar over region (frequency); bar over region+revenue (aggregation)
        tool_use("frequency_bar_chart", {"column": "region", "title": "Region freq", "intent": "i"}, id_="e1"),
        tool_use("aggregation_bar_chart", {"value_col": "revenue", "group_col": "region",
                                           "agg": "mean", "title": "Mean rev", "intent": "i"}, id_="e2"),
    ]
    fake = FakeClaude([
        {"tool_calls": existing},
        {"tool_calls": [
            # repeat of e1's angle (bar over region) — must be deduped out
            tool_use("frequency_bar_chart", {"column": "region", "title": "Dup region", "intent": "i"}, id_="d1"),
            # repeat of e2's angle (bar over region+revenue, different agg) — deduped out
            tool_use("aggregation_bar_chart", {"value_col": "revenue", "group_col": "region",
                                               "agg": "max", "title": "Dup agg", "intent": "i"}, id_="d2"),
            # genuinely new angles
            tool_use("line_chart", {"date_col": "order_date", "value_col": "revenue", "agg": "sum",
                                    "granularity": "week", "title": "Rev trend", "intent": "i"}, id_="d3"),
            tool_use("pie_chart", {"category_col": "region", "agg": "count",
                                   "title": "Region share", "intent": "i"}, id_="d4"),
        ]},
    ])
    gen = _make_generator(sales, fake)
    specs = gen.generate_charts()
    titles = [s.title for s in specs]
    assert "Dup region" not in titles
    assert "Dup agg" not in titles
    assert "Rev trend" in titles
    assert "Region share" in titles
    assert len(specs) == 4   # 2 existing + 2 new (dups dropped)


def test_min_charts_target_constant():
    assert MIN_CHARTS_TARGET == 6
    assert MIN_CHARTS_TARGET < MAX_CHARTS
