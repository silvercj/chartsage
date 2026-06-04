"""Synthetic edge-case dataset generators for the QA harness.

Each generator is a zero-arg function returning (name, DataFrame). `name` is a
stable slug used in result filenames and the report. Each targets one known
failure mode. SYNTHETIC is the registry the runner iterates.

Generators are deterministic (fixed seed) so reruns are comparable.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

_RNG = np.random.default_rng(1234)


def gen_year_as_int() -> tuple[str, pd.DataFrame]:
    """Bare integer years + a measure. The motivating bug: naive to_datetime
    turns 2019/2020/... into epoch-nanosecond timestamps (~1970), collapsing a
    multi-year line to one cluster. A correct pipeline yields a >=5-point line."""
    years = list(range(2015, 2024))
    rows = []
    for y in years:
        for _ in range(40):
            rows.append({"year": y, "revenue": float(_RNG.integers(1000, 9000)),
                         "region": _RNG.choice(["north", "south", "east", "west"])})
    return "year_as_int", pd.DataFrame(rows)


def gen_mixed_date_formats() -> tuple[str, pd.DataFrame]:
    """A date column whose cells use clashing formats incl. an Excel serial int."""
    raw = ["2020-01-15", "02/03/2020", "03-04-20", "44105", "Jan 2020",
           "2020-05-01", "06/07/2020", "07-08-20", "44200", "Feb 2020"] * 12
    return "mixed_date_formats", pd.DataFrame({
        "event_date": raw,
        "amount": [float(_RNG.integers(10, 500)) for _ in raw],
    })


def gen_nan_heavy() -> tuple[str, pd.DataFrame]:
    """Columns that are mostly null (one >95% null to trip the unusable anomaly)."""
    n = 200
    almost_all_null = [float(_RNG.integers(1, 100)) if i % 50 == 0 else np.nan for i in range(n)]
    half_null = [float(_RNG.integers(1, 100)) if i % 2 == 0 else np.nan for i in range(n)]
    cat = [_RNG.choice(["a", "b", "c"]) if i % 3 else None for i in range(n)]
    return "nan_heavy", pd.DataFrame({
        "mostly_null": almost_all_null, "half_null": half_null, "category": cat,
    })


def gen_high_cardinality_cat() -> tuple[str, pd.DataFrame]:
    """A categorical with 500+ distinct values (must roll into top-N + Other,
    not error out or chart every value)."""
    n = 3000
    cats = [f"sku_{_RNG.integers(0, 600)}" for _ in range(n)]
    return "high_cardinality_cat", pd.DataFrame({
        "sku": cats, "units": [float(_RNG.integers(1, 50)) for _ in range(n)],
    })


def gen_unicode_emoji() -> tuple[str, pd.DataFrame]:
    """Non-ASCII + emoji category labels (must survive JSON + chart x/labels)."""
    labels = ["café", "naïve", "Zürich", "東京", "🚀rocket", "Москва", "São Paulo", "δοκιμή"]
    n = 240
    return "unicode_emoji", pd.DataFrame({
        "place": [labels[i % len(labels)] for i in range(n)],
        "score": [float(_RNG.integers(1, 100)) for _ in range(n)],
    })


def gen_currency_strings() -> tuple[str, pd.DataFrame]:
    """Money stored as strings with symbols/grouping (£1,200 / $1.2k / 1.234,56) —
    must NOT be silently charted as a numeric measure."""
    vals = ["£1,200", "$1.2k", "1.234,56", "€980", "$3,400", "£500", "2.000,00", "$1.1k"]
    n = 200
    return "currency_strings", pd.DataFrame({
        "region": [_RNG.choice(["north", "south", "east", "west"]) for _ in range(n)],
        "revenue": [vals[i % len(vals)] for i in range(n)],
    })


def gen_single_row_tiny() -> tuple[str, pd.DataFrame]:
    """A 2-row frame: probes the >=1-row guard and degenerate aggregations."""
    return "single_row_tiny", pd.DataFrame({
        "category": ["a", "b"], "value": [10.0, 20.0],
    })


def gen_duplicate_columns() -> tuple[str, pd.DataFrame]:
    """Two columns named identically (post-lower-casing collision) — the loader
    lower-cases names; the harness must not crash on the duplicate."""
    df = pd.DataFrame(
        [[1, "x", 5.0], [2, "y", 6.0], [3, "z", 7.0]],
        columns=["Value", "label", "value"],
    )
    return "duplicate_columns", df


def gen_all_categorical() -> tuple[str, pd.DataFrame]:
    """No numeric columns at all — count-based charts only."""
    n = 300
    return "all_categorical", pd.DataFrame({
        "color": [_RNG.choice(["red", "green", "blue"]) for _ in range(n)],
        "size": [_RNG.choice(["s", "m", "l", "xl"]) for _ in range(n)],
        "shape": [_RNG.choice(["circle", "square", "tri"]) for _ in range(n)],
    })


def gen_all_numeric() -> tuple[str, pd.DataFrame]:
    """No categoricals/dates — distributions, scatter, correlation only."""
    n = 500
    a = _RNG.normal(50, 10, n)
    return "all_numeric", pd.DataFrame({
        "a": a, "b": a * 1.5 + _RNG.normal(0, 5, n), "c": _RNG.uniform(0, 100, n),
    })


def gen_wide_50col() -> tuple[str, pd.DataFrame]:
    """50+ columns — exercises column-handling at width."""
    n = 100
    data = {f"num_{i}": _RNG.normal(0, 1, n) for i in range(45)}
    for i in range(6):
        data[f"cat_{i}"] = [_RNG.choice(["p", "q", "r"]) for _ in range(n)]
    return "wide_50col", pd.DataFrame(data)


def gen_tall_100k() -> tuple[str, pd.DataFrame]:
    """100k+ rows — exercises sample_for_analysis (MAX_ANALYSIS_ROWS=50000)."""
    n = 120_000
    return "tall_100k", pd.DataFrame({
        "category": _RNG.choice(["a", "b", "c", "d", "e"], n),
        "value": _RNG.normal(100, 25, n),
    })


def gen_boolean_ish() -> tuple[str, pd.DataFrame]:
    """Yes/No, 0/1, True/False columns — low-cardinality, boolean-like."""
    n = 240
    return "boolean_ish", pd.DataFrame({
        "subscribed": [_RNG.choice(["Yes", "No"]) for _ in range(n)],
        "active": [int(_RNG.integers(0, 2)) for _ in range(n)],
        "flag": [bool(_RNG.integers(0, 2)) for _ in range(n)],
        "amount": [float(_RNG.integers(1, 100)) for _ in range(n)],
    })


def gen_mixed_type_column() -> tuple[str, pd.DataFrame]:
    """One column mixing ints and strings (object dtype) — type sniffing must
    not treat it as clean numeric."""
    n = 200
    col = [int(_RNG.integers(1, 100)) if i % 2 else f"n/a_{i}" for i in range(n)]
    return "mixed_type_column", pd.DataFrame({
        "region": [_RNG.choice(["north", "south"]) for _ in range(n)],
        "messy": col,
    })


def gen_id_like_bigints() -> tuple[str, pd.DataFrame]:
    """A unique big-integer id column (>=50 rows so the identifier heuristic
    fires) — must be classified identifier, never charted as a measure."""
    n = 300
    base = 1_000_000_000
    return "id_like_bigints", pd.DataFrame({
        "order_id": [base + i for i in range(n)],
        "region": [_RNG.choice(["north", "south", "east", "west"]) for _ in range(n)],
        "amount": [float(_RNG.integers(1, 500)) for _ in range(n)],
    })


def gen_negatives_outliers() -> tuple[str, pd.DataFrame]:
    """Negative values in a 'should-be-non-negative' column + an extreme outlier."""
    n = 300
    vals = _RNG.normal(50, 8, n)
    vals[0] = -200.0          # negative in a 'count'-named column -> anomaly
    vals[1] = 50_000.0        # extreme outlier -> anomaly
    return "negatives_outliers", pd.DataFrame({
        "region": [_RNG.choice(["north", "south", "east", "west"]) for _ in range(n)],
        "count": vals,
    })


SYNTHETIC: list[Callable[[], tuple[str, pd.DataFrame]]] = [
    gen_year_as_int,
    gen_mixed_date_formats,
    gen_nan_heavy,
    gen_high_cardinality_cat,
    gen_unicode_emoji,
    gen_currency_strings,
    gen_single_row_tiny,
    gen_duplicate_columns,
    gen_all_categorical,
    gen_all_numeric,
    gen_wide_50col,
    gen_tall_100k,
    gen_boolean_ish,
    gen_mixed_type_column,
    gen_id_like_bigints,
    gen_negatives_outliers,
]
