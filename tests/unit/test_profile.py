import pandas as pd
from profile import profile_dataframe


def test_detects_numeric_role(activities):
    profile = profile_dataframe(activities)
    duration = next(c for c in profile.columns if c.name == "duration_minutes")
    assert duration.role == "numeric"
    assert duration.min is not None
    assert duration.max is not None


def test_detects_categorical_role(activities):
    profile = profile_dataframe(activities)
    atype = next(c for c in profile.columns if c.name == "activity_type")
    assert atype.role == "categorical"
    assert atype.top_values is not None
    assert len(atype.top_values) > 0


def test_detects_date_role(activities):
    profile = profile_dataframe(activities)
    adate = next(c for c in profile.columns if c.name == "activity_date")
    assert adate.role == "date"
    assert adate.min_date is not None


def test_identifier_by_name_suffix():
    df = pd.DataFrame({"activity_id": [1, 2, 3, 4, 5], "x": [10, 20, 30, 40, 50]})
    profile = profile_dataframe(df)
    aid = next(c for c in profile.columns if c.name == "activity_id")
    assert aid.role == "identifier"


def test_identifier_by_cardinality():
    df = pd.DataFrame({"unique_col": list(range(100)), "low_card": [1, 2] * 50})
    profile = profile_dataframe(df)
    unique = next(c for c in profile.columns if c.name == "unique_col")
    assert unique.role == "identifier"


def test_low_cardinality_numeric_still_numeric():
    """Ratings 1-5 should be numeric, not unusable."""
    df = pd.DataFrame({"rating": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]})
    profile = profile_dataframe(df)
    rating = next(c for c in profile.columns if c.name == "rating")
    assert rating.role == "numeric"


def test_anomaly_negative_duration(negative_duration):
    profile = profile_dataframe(negative_duration)
    assert any("negative" in a.lower() for a in profile.anomalies)


def test_correlations_for_numeric_pairs():
    df = pd.DataFrame({
        "x": list(range(20)),
        "y": [i * 2 + 1 for i in range(20)],  # perfectly correlated
        "category": ["a"] * 20,
    })
    profile = profile_dataframe(df)
    assert len(profile.correlations) >= 1
    assert any(abs(r) > 0.9 for r in profile.correlations.values())


def test_degenerate_df_still_profiles(degenerate):
    profile = profile_dataframe(degenerate)
    assert profile.row_count == 8
    assert len(profile.columns) == 1


def test_to_text_includes_columns(activities):
    profile = profile_dataframe(activities)
    text = profile.to_text()
    assert "activity_type" in text
    assert "duration_minutes" in text
