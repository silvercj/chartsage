import pytest
from pydantic import ValidationError
from schemas import ChartSpec, ToolError, ColumnInfo, DataProfile, Report, ReportNarrative


def test_chart_spec_minimum_required():
    spec = ChartSpec(
        kind="bar",
        title="t",
        intent="i",
        x=["a", "b"],
        y=[1, 2],
        x_label="X",
        y_label="Y",
        x_display_type="category",
        y_display_type="count",
        source_columns=["col1"],
        data_point_count=2,
    )
    assert spec.kind == "bar"


def test_chart_spec_rejects_invalid_kind():
    with pytest.raises(ValidationError):
        ChartSpec(
            kind="banana",  # not in literal
            title="t", intent="i",
            x=["a"], y=[1],
            x_label="X", y_label="Y",
            x_display_type="category", y_display_type="count",
            source_columns=["col1"], data_point_count=1,
        )


def test_tool_error_holds_reason():
    err = ToolError(reason="'revenue' is not a column")
    assert "revenue" in err.reason


def test_column_info_categorical():
    col = ColumnInfo(
        name="region", dtype="object", role="categorical",
        cardinality=4, null_count=0,
        top_values=[("north", 5), ("south", 4)],
    )
    assert col.role == "categorical"


def test_data_profile_basic():
    profile = DataProfile(
        row_count=100,
        columns=[ColumnInfo(name="x", dtype="int64", role="numeric",
                            cardinality=50, null_count=0, min=0, max=100)],
        correlations={},
        anomalies=[],
    )
    assert profile.row_count == 100
    text = profile.to_text()
    assert "x" in text


def test_report_round_trips_json():
    r = Report(
        generated_at="2026-05-23T12:00:00",
        summary="...",
        data_quality=[],
        charts=[],
        metadata={"model": "haiku-4-5"},
    )
    payload = r.model_dump_json()
    r2 = Report.model_validate_json(payload)
    assert r2.summary == "..."
