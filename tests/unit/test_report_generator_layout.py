import pandas as pd
import pytest
from unittest.mock import MagicMock
from report_generator import ReportGenerator
from profile import profile_dataframe
from tests.helpers.fake_claude import FakeClaude, tool_use


def _make_generator(df, fake):
    profile = profile_dataframe(df)
    client = MagicMock()
    client.messages_create = fake
    return ReportGenerator(
        profile=profile, df=df, claude=client,
        model_selection="m1", model_narrative="m2",
    )


def _ten_charts_response(activities):
    """Build a FakeClaude with 10 chart tool calls + a narrative call."""
    chart_calls = []
    for i in range(10):
        chart_calls.append(tool_use(
            "frequency_bar_chart",
            {"column": "activity_type", "title": f"Chart {i}", "intent": f"i{i}"},
        ))
    return FakeClaude([
        {"tool_calls": chart_calls},
        {"tool_calls": [tool_use(
            "submit_narrative",
            {"summary": "s", "captions": [f"c{i}" for i in range(10)], "data_quality": []},
        )]},
    ])


def test_chart_ids_unique(activities):
    fake = _ten_charts_response(activities)
    gen = _make_generator(activities, fake)
    report = gen.build_report()
    ids = [c.chart_id for c in report.charts]
    assert len(ids) == len(set(ids)), "chart_ids must be unique"
    assert all(isinstance(i, str) and len(i) > 0 for i in ids)


def test_default_layout_splits_5_main_5_sidebar(activities):
    fake = _ten_charts_response(activities)
    gen = _make_generator(activities, fake)
    report = gen.build_report()

    assert len(report.layout) == 10
    main = [e for e in report.layout if e.position == "main"]
    sidebar = [e for e in report.layout if e.position == "sidebar"]
    assert len(main) == 5
    assert len(sidebar) == 5

    # First 5 charts go to main, next 5 to sidebar
    main_ids_in_order = [e.chart_id for e in sorted(main, key=lambda e: e.order)]
    sidebar_ids_in_order = [e.chart_id for e in sorted(sidebar, key=lambda e: e.order)]
    assert main_ids_in_order == [c.chart_id for c in report.charts[:5]]
    assert sidebar_ids_in_order == [c.chart_id for c in report.charts[5:]]

    # Orders are 0-indexed and dense
    assert [e.order for e in sorted(main, key=lambda e: e.order)] == [0, 1, 2, 3, 4]
    assert [e.order for e in sorted(sidebar, key=lambda e: e.order)] == [0, 1, 2, 3, 4]


def test_fewer_than_5_charts_all_main(activities):
    """Edge case: if only 3 charts come back, all go to main."""
    chart_calls = [tool_use(
        "frequency_bar_chart",
        {"column": "activity_type", "title": f"c{i}", "intent": "i"},
    ) for i in range(3)]
    fake = FakeClaude([
        {"tool_calls": chart_calls},
        {"tool_calls": []},  # reach-for-more (under target, no errors) proposes nothing
        {"tool_calls": [tool_use(
            "submit_narrative",
            {"summary": "s", "captions": ["c1", "c2", "c3"], "data_quality": []},
        )]},
    ])
    gen = _make_generator(activities, fake)
    report = gen.build_report()
    assert all(e.position == "main" for e in report.layout)
    assert len(report.layout) == 3
