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


def test_fallback_caps_bars_and_diversifies():
    # Several categoricals + numerics: the OLD heuristic produced 2 frequency bars first.
    # The rebalanced one caps bars at 1 and mixes in other kinds.
    df = pd.DataFrame({
        "region":  (["West", "East", "North", "South"] * 25),
        "tier":    (["A", "B", "C", "D", "E"] * 20),
        "channel": (["Direct", "Online", "Partner"] * 33 + ["Direct"]),
        "revenue": [i * 1.5 for i in range(100)],
        "orders":  [i % 30 for i in range(100)],
    })
    profile = profile_dataframe(df)
    specs = pick_fallback_charts(profile, df)
    kinds = [s.kind for s in specs]
    assert kinds.count("bar") <= 1          # at most one frequency bar
    assert len(set(kinds)) >= 2             # not an all-one-kind report


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


def test_unique_label_table_charts_metric_not_frequency():
    """A 'label + metrics' table — one unique-label categorical + numeric(s) — must NOT
    fall back to a frequency bar of the label. Every label appears once, so the counts
    are all 1 and the bar renders flat/useless. The fallback must chart the metric BY
    the label instead (the chart the data actually wants). Regression for the hurricanes
    report that came out as a flat 'decade — distribution' bar of all 1s."""
    df = pd.DataFrame({
        "Decade": ["1990s", "2000s", "2010s", "2020s"],
        "Major_per_year": [2.5, 3.6, 3.0, 4.33],
    })
    profile = profile_dataframe(df)
    specs = pick_fallback_charts(profile, df)
    assert specs, "fallback produced no charts"
    lead = specs[0]
    assert lead.kind == "bar"
    assert len(set(lead.y)) > 1, f"lead bar is a degenerate frequency chart: y={lead.y}"
    assert max(lead.y) == pytest.approx(4.33), f"lead bar charts counts, not the metric: y={lead.y}"


def test_chart_composition_and_fallback_detection():
    """The analytics helper that powers the fallback-rate event: counts model vs.
    fallback charts and the kind mix off the spec `intent` marker."""
    from fallback import is_fallback_spec, chart_composition, FALLBACK_INTENT

    class _Spec:
        def __init__(self, kind, intent):
            self.kind, self.intent = kind, intent

    class _Chart:
        def __init__(self, spec):
            self.spec = spec

    charts = [
        _Chart(_Spec("bar", "Ranking circuits by win rate")),     # model-selected
        _Chart(_Spec("histogram", FALLBACK_INTENT)),              # fallback
        _Chart(_Spec("scatter", FALLBACK_INTENT)),                # fallback
    ]
    assert is_fallback_spec(charts[0].spec) is False
    assert is_fallback_spec(charts[1].spec) is True

    comp = chart_composition(charts)
    assert comp["chartCount"] == 3
    assert comp["fallbackChartCount"] == 2
    assert comp["modelChartCount"] == 1
    assert comp["usedFallback"] is True
    assert comp["allFallback"] is False
    assert comp["fallbackRatio"] == round(2 / 3, 3)
    assert comp["chartKinds"] == ["bar", "histogram", "scatter"]
    # all-fallback case (the hurricanes report)
    assert chart_composition(charts[1:])["allFallback"] is True


def test_drop_duplicate_fallback_charts():
    """The fallback re-derives charts from the same df, so it can re-make a chart the model
    already selected (same kind + source columns) — e.g. a 'cards over year' line when the
    model already charted cards over year. Those duplicates must be dropped (the World Cup
    cards report showed the cards and reds lines twice each)."""
    from fallback import drop_duplicates

    class S:
        def __init__(self, kind, cols):
            self.kind, self.source_columns = kind, cols

    model = [S("line", ["year", "cards"]), S("line", ["year", "reds"])]
    fb = [
        S("line", ["year", "cards"]),     # duplicate of model[0]
        S("line", ["reds", "year"]),      # duplicate of model[1] (column order ignored)
        S("histogram", ["matches"]),      # genuinely new
    ]
    kept = drop_duplicates(model, fb)
    assert [s.kind for s in kept] == ["histogram"], "should keep only the non-duplicate chart"


def test_fallback_timeseries_leads_with_line_not_histogram():
    """A time-series table — a year/ordinal column + metrics, no categorical — should lead
    with a LINE of the metric over time, not a histogram of the year. Regression for the
    World Cup 'blowouts by year' report that came out as a 'year — distribution' histogram."""
    df = pd.DataFrame({
        "Year": [1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022],
        "Blowout_pct": [5.8, 7.7, 6.2, 4.7, 4.7, 4.7, 6.2, 3.1, 4.7],
    })
    profile = profile_dataframe(df)
    specs = pick_fallback_charts(profile, df)
    assert specs, "fallback produced no charts"
    lead = specs[0]
    assert lead.kind == "line", f"time-series lead should be a line, got {lead.kind!r}"
    assert "Blowout_pct" in lead.source_columns, f"lead should chart the metric: {lead.source_columns}"
    assert lead.y and len(set(lead.y)) > 1, "lead must plot real metric values, not a count distribution"
