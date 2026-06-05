"""Stored percentage values must be 0–1 fractions (the display layer multiplies
by 100). normalize_percentage_spec() rescales specs whose values arrived on the
0–100 scale, across every chart geometry, while leaving genuine fractions,
non-percentage specs, and a dual-axis secondary series untouched.
"""
import pytest
from chart_executor import normalize_percentage_spec
from schemas import ChartSpec


def _spec(**kw):
    base = dict(kind="bar", title="t", intent="i",
                source_columns=["margin"], data_point_count=3,
                y_display_type="percentage")
    base.update(kw)
    return ChartSpec(**base)


def test_single_series_y_rescaled():        # bar / line / pie / histogram / scatter
    out = normalize_percentage_spec(_spec(x=["a", "b", "c"], y=[30.0, 35.0, 40.0]))
    assert out.y == pytest.approx([0.30, 0.35, 0.40])


def test_grouped_bar_series_data_rescaled():
    out = normalize_percentage_spec(_spec(
        kind="grouped_bar",
        series=[{"name": "A", "data": [20.0, 40.0]}, {"name": "B", "data": [60.0, 80.0]}],
    ))
    assert out.series[0]["data"] == pytest.approx([0.20, 0.40])
    assert out.series[1]["data"] == pytest.approx([0.60, 0.80])


def test_grouped_line_and_scatter_series_y_rescaled():
    out = normalize_percentage_spec(_spec(
        kind="line",
        series=[{"name": "A", "x": ["q1", "q2"], "y": [50.0, 70.0]}],
    ))
    assert out.series[0]["y"] == pytest.approx([0.50, 0.70])
    assert out.series[0]["x"] == ["q1", "q2"]      # labels untouched


def test_heatmap_cell_values_rescaled():
    out = normalize_percentage_spec(_spec(
        kind="heatmap",
        y=["a", "b"],                               # category labels, not numbers
        series=[{"row": "a", "col": "b", "value": 50.0}],
    ))
    assert out.series[0]["value"] == pytest.approx(0.50)
    assert out.y == ["a", "b"]                      # string axis untouched


def test_box_stats_and_outliers_rescaled():
    out = normalize_percentage_spec(_spec(
        kind="box",
        series=[{"name": "x", "min": 10.0, "q1": 20.0, "median": 30.0,
                 "q3": 40.0, "max": 50.0, "outliers": [90.0]}],
    ))
    s = out.series[0]
    assert [s["min"], s["q1"], s["median"], s["q3"], s["max"]] == pytest.approx([0.10, 0.20, 0.30, 0.40, 0.50])
    assert s["outliers"] == pytest.approx([0.90])


def test_treemap_nested_node_values_rescaled():
    out = normalize_percentage_spec(_spec(
        kind="treemap",
        nodes=[{"name": "P", "value": 60.0,
                "children": [{"name": "c1", "value": 25.0}, {"name": "c2", "value": 35.0}]}],
    ))
    assert out.nodes[0]["value"] == pytest.approx(0.60)
    assert [c["value"] for c in out.nodes[0]["children"]] == pytest.approx([0.25, 0.35])


def test_dual_axis_rescales_primary_axis_only():
    out = normalize_percentage_spec(_spec(
        kind="dual_axis",
        series=[{"name": "rate", "type": "bar", "yAxisIndex": 0, "data": [30.0, 40.0]},
                {"name": "count", "type": "line", "yAxisIndex": 1, "data": [5000.0, 6000.0]}],
    ))
    assert out.series[0]["data"] == pytest.approx([0.30, 0.40])   # primary % axis
    assert out.series[1]["data"] == [5000.0, 6000.0]              # secondary untouched


def test_genuine_fractions_left_unchanged():
    out = normalize_percentage_spec(_spec(x=["a", "b"], y=[0.3, 0.8]))
    assert out.y == [0.3, 0.8]


def test_non_percentage_spec_untouched():
    out = normalize_percentage_spec(_spec(y_display_type="number", x=["a"], y=[5000.0]))
    assert out.y == [5000.0]
