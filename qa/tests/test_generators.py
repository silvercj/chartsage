"""Each synthetic generator returns (name, DataFrame) with its intended edge-case shape."""
import pandas as pd
import pytest

from qa.generators import SYNTHETIC


def test_registry_has_at_least_15_generators():
    assert len(SYNTHETIC) >= 15


def test_every_generator_returns_name_and_dataframe():
    seen_names = set()
    for gen in SYNTHETIC:
        name, df = gen()
        assert isinstance(name, str) and name
        assert name not in seen_names, f"duplicate generator name {name!r}"
        seen_names.add(name)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] >= 1


def test_every_generator_is_chartable_shape_or_intentionally_tiny():
    # The pipeline requires >=2 cols and >=1 row; only the intentional 'single_row_tiny'
    # and 'duplicate_columns' edge cases may probe the guards. All others must be runnable.
    for gen in SYNTHETIC:
        name, df = gen()
        assert df.shape[0] >= 1, f"{name} produced 0 rows"


def test_year_as_int_is_integer_dtype():
    name, df = _by_name("year_as_int")
    assert "year" in df.columns
    assert pd.api.types.is_integer_dtype(df["year"])
    # Multiple distinct years so a working line chart has >=2 points.
    assert df["year"].nunique() >= 5


def test_tall_dataset_exceeds_sampling_threshold():
    name, df = _by_name("tall_100k")
    assert len(df) >= 100_000


def test_wide_dataset_has_50_plus_columns():
    name, df = _by_name("wide_50col")
    assert df.shape[1] >= 50


def test_high_cardinality_category_500_plus():
    name, df = _by_name("high_cardinality_cat")
    cat_col = [c for c in df.columns if df[c].dtype == object][0]
    assert df[cat_col].nunique() >= 500


def test_id_like_big_ints_are_unique_and_large():
    name, df = _by_name("id_like_bigints")
    id_col = [c for c in df.columns if c.endswith("_id")][0]
    assert df[id_col].nunique() == len(df)
    assert int(df[id_col].min()) > 10_000


def _by_name(target: str):
    for gen in SYNTHETIC:
        name, df = gen()
        if name == target:
            return name, df
    raise AssertionError(f"no generator named {target!r}")
