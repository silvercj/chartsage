import pandas as pd
import pytest
from fallback import pick_fallback_charts
from profile import profile_dataframe
from schemas import ChartSpec


def test_picks_at_least_three_for_normal_data(activities):
    profile = profile_dataframe(activities)
    specs = pick_fallback_charts(profile, activities)
    assert len(specs) >= 3


def test_uses_frequency_for_categoricals(sales):
    profile = profile_dataframe(sales)
    specs = pick_fallback_charts(profile, sales)
    kinds = {s.kind for s in specs}
    assert "bar" in kinds


def test_uses_histogram_for_numerics(activities):
    profile = profile_dataframe(activities)
    specs = pick_fallback_charts(profile, activities)
    kinds = {s.kind for s in specs}
    assert "histogram" in kinds or "bar" in kinds


def test_uses_scatter_for_correlated_pair():
    df = pd.DataFrame({
        "x": list(range(20)),
        "y": [i * 2 + 1 for i in range(20)],
        "label": ["a"] * 20,
    })
    profile = profile_dataframe(df)
    specs = pick_fallback_charts(profile, df)
    kinds = {s.kind for s in specs}
    assert "scatter" in kinds


def test_handles_degenerate(degenerate):
    profile = profile_dataframe(degenerate)
    specs = pick_fallback_charts(profile, degenerate)
    # Degenerate data may not produce any charts; that's OK
    assert isinstance(specs, list)


def test_specs_marked_as_fallback(activities):
    profile = profile_dataframe(activities)
    specs = pick_fallback_charts(profile, activities)
    if specs:
        assert all("fallback" in s.intent.lower() for s in specs)
