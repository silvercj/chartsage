import pytest
from pydantic import ValidationError
from schemas import ChartLayoutEntry, ChartSpec, ChartWithCaption, Report


def _spec() -> ChartSpec:
    return ChartSpec(
        kind="bar", title="t", intent="i",
        x=["a"], y=[1],
        x_label="X", y_label="Y",
        x_display_type="category", y_display_type="count",
        source_columns=["c"], data_point_count=1,
    )


def test_layout_entry_minimum():
    e = ChartLayoutEntry(chart_id="abc", position="main", order=0)
    assert e.chart_id == "abc"
    assert e.position == "main"
    assert e.order == 0


def test_layout_entry_rejects_invalid_position():
    with pytest.raises(ValidationError):
        ChartLayoutEntry(chart_id="abc", position="trash", order=0)


def test_chart_with_caption_has_chart_id():
    c = ChartWithCaption(chart_id="abc", spec=_spec(), caption="cap")
    assert c.chart_id == "abc"


def test_report_has_layout_field_with_default_empty():
    r = Report(
        generated_at="2026-05-24T00:00:00",
        summary="...", data_quality=[], charts=[], metadata={},
    )
    assert r.layout == []


def test_report_round_trips_with_layout():
    r = Report(
        generated_at="2026-05-24T00:00:00",
        summary="s", data_quality=[],
        charts=[ChartWithCaption(chart_id="c1", spec=_spec(), caption="cap")],
        layout=[ChartLayoutEntry(chart_id="c1", position="main", order=0)],
        metadata={},
    )
    payload = r.model_dump_json()
    r2 = Report.model_validate_json(payload)
    assert r2.layout[0].chart_id == "c1"
    assert r2.charts[0].chart_id == "c1"
