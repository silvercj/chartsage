"""Deterministic validators: each check fires on a crafted bad report and passes a good one."""
import pandas as pd

from qa.validators import validate, Issue


# ---- helpers ---------------------------------------------------------------

def _df_sales():
    return pd.DataFrame({
        "region": ["north", "south", "east", "north", "south", "east"],
        "revenue": [100.0, 200.0, 300.0, 150.0, 250.0, 350.0],
    })


def _good_bar_report():
    """A clean frequency bar over region with a real groupby-consistent y."""
    return {
        "generated_at": "2026-06-04T00:00:00",
        "summary": "Region counts are even across the three regions.",
        "data_quality": [],
        "key_metrics": [{"label": "Total revenue", "value": 1350.0, "format": "currency"}],
        "charts": [{
            "chart_id": "c1",
            "caption": "Counts by region.",
            "spec": {
                "kind": "bar", "title": "By region", "intent": "mix",
                "x": ["north", "south", "east"], "y": [2, 2, 2],
                "x_label": "region", "y_label": "Count",
                "x_display_type": "category", "y_display_type": "count",
                "source_columns": ["region"], "data_point_count": 6,
            },
        }],
        "layout": [{"chart_id": "c1", "position": "main", "order": 0}],
        "metadata": {},
    }


def _codes(issues: list[Issue]) -> set[str]:
    return {i.code for i in issues}


# ---- generation error ------------------------------------------------------

def test_generation_error_when_report_none():
    issues = validate(_df_sales(), None)
    assert any(i.code == "generation_error" and i.severity == "fail" for i in issues)


def test_good_report_has_no_fail_issues():
    issues = validate(_df_sales(), _good_bar_report())
    fails = [i for i in issues if i.severity == "fail"]
    assert fails == [], f"unexpected fails: {fails}"


# ---- degenerate chart ------------------------------------------------------

def test_empty_xy_flagged():
    rep = _good_bar_report()
    rep["charts"][0]["spec"]["x"] = []
    rep["charts"][0]["spec"]["y"] = []
    issues = validate(_df_sales(), rep)
    assert "degenerate_chart" in _codes(issues)


def test_line_with_one_distinct_x_flagged():
    rep = _good_bar_report()
    spec = rep["charts"][0]["spec"]
    spec["kind"] = "line"
    spec["x"] = ["2020", "2020", "2020"]
    spec["y"] = [1.0, 2.0, 3.0]
    issues = validate(_df_sales(), rep)
    assert any(i.code == "degenerate_chart" and i.chart_id == "c1" for i in issues)


def test_all_identical_y_flagged():
    rep = _good_bar_report()
    rep["charts"][0]["spec"]["y"] = [5, 5, 5]
    issues = validate(_df_sales(), rep)
    assert "degenerate_chart" in _codes(issues)


def test_all_nan_y_flagged():
    rep = _good_bar_report()
    rep["charts"][0]["spec"]["y"] = [None, None, None]
    issues = validate(_df_sales(), rep)
    assert "degenerate_chart" in _codes(issues)


def test_all_zero_y_flagged():
    rep = _good_bar_report()
    rep["charts"][0]["spec"]["y"] = [0, 0, 0]
    issues = validate(_df_sales(), rep)
    assert "degenerate_chart" in _codes(issues)


# ---- chart-data consistency ------------------------------------------------

def test_frequency_bar_consistent_passes():
    # _good_bar_report's y=[2,2,2] matches value_counts of region.
    issues = validate(_df_sales(), _good_bar_report())
    assert "chart_data_mismatch" not in _codes(issues)


def test_frequency_bar_inconsistent_flagged():
    rep = _good_bar_report()
    rep["charts"][0]["spec"]["y"] = [2, 2, 99]   # east is really 2, not 99
    issues = validate(_df_sales(), rep)
    assert "chart_data_mismatch" in _codes(issues)


def test_aggregation_sum_bar_consistent_passes():
    rep = _good_bar_report()
    spec = rep["charts"][0]["spec"]
    spec["title"] = "Revenue by region"
    spec["y_display_type"] = "currency"
    spec["y_label"] = "Sum of revenue"
    spec["source_columns"] = ["revenue", "region"]
    # real sums: north=250, south=450, east=650
    spec["x"] = ["east", "south", "north"]
    spec["y"] = [650.0, 450.0, 250.0]
    issues = validate(_df_sales(), rep)
    assert "chart_data_mismatch" not in _codes(issues)


def test_aggregation_sum_bar_inconsistent_flagged():
    rep = _good_bar_report()
    spec = rep["charts"][0]["spec"]
    spec["title"] = "Revenue by region"
    spec["y_label"] = "Sum of revenue"
    spec["source_columns"] = ["revenue", "region"]
    spec["x"] = ["east", "south", "north"]
    spec["y"] = [650.0, 450.0, 999.0]    # north sum is 250, not 999
    issues = validate(_df_sales(), rep)
    assert "chart_data_mismatch" in _codes(issues)


# ---- KPI sanity ------------------------------------------------------------

def test_kpi_non_finite_flagged():
    rep = _good_bar_report()
    rep["key_metrics"] = [{"label": "Avg", "value": float("inf"), "format": "number"}]
    issues = validate(_df_sales(), rep)
    assert any(i.code == "kpi_non_finite" and i.severity == "fail" for i in issues)


def test_kpi_year_lookalike_warns():
    rep = _good_bar_report()
    rep["key_metrics"] = [{"label": "Years covered", "value": 2022.0, "format": "number"}]
    issues = validate(_df_sales(), rep)
    assert any(i.code == "kpi_year_lookalike" and i.severity == "warn" for i in issues)


def test_kpi_total_2020_warns():
    rep = _good_bar_report()
    rep["key_metrics"] = [{"label": "Total", "value": 2020.0, "format": "number"}]
    issues = validate(_df_sales(), rep)
    assert "kpi_year_lookalike" in {i.code for i in issues}


def test_kpi_legit_count_not_a_year_passes():
    rep = _good_bar_report()
    rep["key_metrics"] = [{"label": "Total rows", "value": 1500.0, "format": "number"}]
    issues = validate(_df_sales(), rep)
    assert "kpi_year_lookalike" not in {i.code for i in issues}


def test_kpi_revenue_2020_not_flagged_label_mismatch():
    # 'revenue' isn't a year/count/total label, so 2020.0 is fine.
    rep = _good_bar_report()
    rep["key_metrics"] = [{"label": "Revenue", "value": 2020.0, "format": "currency"}]
    issues = validate(_df_sales(), rep)
    assert "kpi_year_lookalike" not in {i.code for i in issues}


# ---- narrative / floor / labels --------------------------------------------

def test_empty_summary_flagged():
    rep = _good_bar_report()
    rep["summary"] = "   "
    issues = validate(_df_sales(), rep)
    assert any(i.code == "narrative_missing" and i.severity == "fail" for i in issues)


def test_chart_count_below_floor_flagged():
    rep = _good_bar_report()   # only 1 chart; floor is MIN_CHARTS_FOR_NO_FALLBACK (3)
    issues = validate(_df_sales(), rep)
    assert any(i.code == "chart_count_below_floor" for i in issues)


def test_missing_axis_labels_warn():
    rep = _good_bar_report()
    rep["charts"][0]["spec"]["x_label"] = ""
    rep["charts"][0]["spec"]["y_label"] = ""
    issues = validate(_df_sales(), rep)
    assert any(i.code == "missing_axis_label" and i.severity == "warn" for i in issues)


# ---- the motivating regression ---------------------------------------------

def test_year_as_int_dataset_flags_if_line_regresses():
    """Build the year-as-int generator df and feed a DEGENERATE line report (what a
    broken _to_datetime would yield: every year collapsed to one period). The
    degenerate-chart check must fire. This is the harness's proof-of-catch."""
    from qa.generators import gen_year_as_int
    _name, df = gen_year_as_int()
    df.columns = [c.lower() for c in df.columns]
    degenerate = {
        "generated_at": "2026-06-04T00:00:00",
        "summary": "Revenue over time.",
        "data_quality": [],
        "key_metrics": [],
        "charts": [
            {"chart_id": "L", "caption": "trend", "spec": {
                "kind": "line", "title": "Revenue over time", "intent": "trend",
                # broken: every point lands on the SAME period label
                "x": ["1970", "1970", "1970", "1970", "1970"],
                "y": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
                "x_label": "Year (year)", "y_label": "Sum of revenue",
                "x_display_type": "date", "y_display_type": "currency",
                "source_columns": ["year", "revenue"], "data_point_count": 360,
            }},
            # pad to clear the floor so we isolate the degenerate-line signal
            {"chart_id": "B1", "caption": "c", "spec": {
                "kind": "bar", "title": "Region mix", "intent": "mix",
                "x": ["north", "south"], "y": [180, 180],
                "x_label": "region", "y_label": "Count",
                "x_display_type": "category", "y_display_type": "count",
                "source_columns": ["region"], "data_point_count": 360}},
            {"chart_id": "B2", "caption": "c", "spec": {
                "kind": "bar", "title": "Rev by region", "intent": "x",
                "x": ["north", "south"], "y": [50000.0, 49000.0],
                "x_label": "region", "y_label": "Sum of revenue",
                "x_display_type": "category", "y_display_type": "currency",
                "source_columns": ["revenue", "region"], "data_point_count": 360}},
        ],
        "layout": [{"chart_id": "L", "position": "main", "order": 0}],
        "metadata": {},
    }
    issues = validate(df, degenerate)
    assert any(i.code == "degenerate_chart" and i.chart_id == "L" for i in issues), \
        "the collapsed year-as-int line must be flagged degenerate"
