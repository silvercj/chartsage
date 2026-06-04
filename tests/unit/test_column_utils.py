"""normalize_columns: lower-case headers and de-duplicate names that collide
after lower-casing (the duplicate-column crash the QA harness surfaced)."""
import pandas as pd

from column_utils import normalize_columns


def test_lowercases_columns():
    df = pd.DataFrame({"Region": ["n"], "Revenue": [1.0]})
    out = normalize_columns(df)
    assert list(out.columns) == ["region", "revenue"]


def test_no_collision_unchanged():
    df = pd.DataFrame({"region": ["n"], "revenue": [1.0]})
    out = normalize_columns(df)
    assert list(out.columns) == ["region", "revenue"]


def test_dedupes_case_collision():
    # 'Value' and 'value' both lower-case to 'value' -> must be made distinct.
    df = pd.DataFrame([[1, "x", 5.0], [2, "y", 6.0]], columns=["Value", "label", "value"])
    out = normalize_columns(df)
    assert list(out.columns) == ["value", "label", "value.1"]
    # The crash's root cause: a duplicate label makes df[name] a DataFrame.
    # After normalization every label selects a Series.
    assert isinstance(out["value"], pd.Series)
    assert isinstance(out["value.1"], pd.Series)


def test_multiple_collisions():
    df = pd.DataFrame([[1, 2, 3]], columns=["Date", "DATE", "date"])
    out = normalize_columns(df)
    assert list(out.columns) == ["date", "date.1", "date.2"]


def test_suffix_avoids_existing_name():
    # A pre-existing 'value.1' must not collide with the mangled suffix.
    df = pd.DataFrame([[1, 2, 3]], columns=["value", "value.1", "Value"])
    out = normalize_columns(df)
    assert list(out.columns) == ["value", "value.1", "value.2"]
    assert len(set(out.columns)) == 3


def test_does_not_mutate_input():
    df = pd.DataFrame([[1, 2]], columns=["A", "a"])
    _ = normalize_columns(df)
    assert list(df.columns) == ["A", "a"]   # original untouched


def test_profile_dataframe_survives_collision():
    """The actual crash: profile_dataframe iterated df[name] and called .dtype,
    which fails when name is duplicated. Normalizing first must let it through."""
    from profile import profile_dataframe
    df = normalize_columns(
        pd.DataFrame([[1, "x", 5.0], [2, "y", 6.0], [3, "z", 7.0]],
                     columns=["Value", "label", "value"])
    )
    prof = profile_dataframe(df)   # must not raise AttributeError
    assert prof.row_count == 3
    assert len(prof.columns) == 3
