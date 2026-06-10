"""Temporal-ordinal columns (a Year/Decade/Quarter axis stored as numbers) must be
detected ONCE in the profile and consumed everywhere: the model's prompt text, the
fallback's lead-line pick, and the key-metrics role guard. Previously the detection
lived only in fallback.py, so the model was told 'role=numeric' for a Year column and
kept histogramming it / reporting max(year) as a KPI."""
import pandas as pd
from unittest.mock import MagicMock

from profile import profile_dataframe
from fallback import pick_fallback_charts
from report_generator import ReportGenerator


def _years_df(n=24, start=1990):
    return pd.DataFrame({
        "year": list(range(start, start + n)),
        "goals": [float(50 + (i * 7) % 40) for i in range(n)],
    })


def _col(profile, name):
    return next(c for c in profile.columns if c.name == name)


def test_year_column_flagged_temporal_ordinal():
    profile = profile_dataframe(_years_df())
    col = _col(profile, "year")
    assert col.role == "numeric"
    assert col.temporal_ordinal is True
    assert _col(profile, "goals").temporal_ordinal is False


def test_rate_named_per_year_is_not_temporal():
    # 'majors_per_year' merely CONTAINS a temporal word — it's a rate, not the axis.
    df = pd.DataFrame({
        "decade": ["1990s", "2000s", "2010s", "2020s"],
        "majors_per_year": [2.5, 3.6, 3.0, 4.33],
    })
    profile = profile_dataframe(df)
    assert _col(profile, "majors_per_year").temporal_ordinal is False


def test_repeating_year_is_not_an_ordinal_axis():
    # Many rows per year (an event log) — year is a grouping dimension, not a unique axis.
    df = pd.DataFrame({
        "year": [2020] * 10 + [2021] * 10,
        "value": list(range(20)),
    })
    profile = profile_dataframe(df)
    assert _col(profile, "year").temporal_ordinal is False


def test_long_unique_year_column_is_not_misread_as_identifier():
    # 74 unique integer years would previously trip the all-unique-ints identifier rule,
    # hiding the time axis from the model entirely.
    profile = profile_dataframe(_years_df(n=74, start=1950))
    col = _col(profile, "year")
    assert col.role == "numeric"
    assert col.temporal_ordinal is True


def test_profile_text_tells_model_to_use_it_as_time_axis():
    text = profile_dataframe(_years_df()).to_text()
    assert "time axis" in text


def test_fallback_leads_with_metric_over_the_flagged_axis():
    df = _years_df()
    profile = profile_dataframe(df)
    specs = pick_fallback_charts(profile, df)
    assert specs[0].kind == "line"
    assert specs[0].source_columns == ["year", "goals"]


def test_key_metrics_guard_covers_temporal_ordinal_numerics():
    # max(year) as a 'Current year' KPI must be dropped even though year's role is numeric.
    df = _years_df()
    gen = ReportGenerator(
        profile=profile_dataframe(df), df=df, claude=MagicMock(),
        model_selection="m", model_narrative="m",
    )
    block = MagicMock()
    block.type = "tool_use"
    block.id = "tu_km"
    block.name = "key_metrics"
    block.input = {"metrics": [
        {"label": "Latest year", "column": "year", "agg": "max", "format": "number"},
        {"label": "Total goals", "column": "goals", "agg": "sum", "format": "number"},
    ]}
    gen._execute_tool_calls([block])
    labels = [m.label for m in gen._key_metrics]
    assert labels == ["Total goals"]
