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


def test_high_cardinality_repeating_categorical_is_usable():
    """A categorical string column with many distinct values that REPEAT (well below
    the row count) is chartable as a top-N (frequency bar / treemap) — it must be
    'categorical', not 'unusable'. Mirrors Netflix 'country' (~70 distinct over 8807
    rows): cardinality > 50 but values recur heavily."""
    import random
    random.seed(0)
    countries = [f"Country {i}" for i in range(70)]
    df = pd.DataFrame({"country": [random.choice(countries) for _ in range(2000)]})
    country = next(c for c in profile_dataframe(df).columns if c.name == "country")
    assert country.role == "categorical"
    assert country.top_values is not None and len(country.top_values) > 0


def test_near_unique_text_column_stays_unusable():
    """A free-text / near-unique column (cardinality ≈ row count, like a movie title)
    is NOT chartable — it must stay 'unusable' so it never becomes a frequency bar."""
    df = pd.DataFrame({"title": [f"Title {i}" for i in range(2000)]})
    title = next(c for c in profile_dataframe(df).columns if c.name == "title")
    assert title.role == "unusable"


def test_thousands_of_distinct_names_stay_unusable():
    """Even when values repeat (ratio < 0.5), a column with thousands of distinct
    values (a director / person name) reads as free text — the absolute ceiling keeps
    it 'unusable' rather than turning it into a 1000-node treemap. Guards conservatism."""
    import random
    random.seed(1)
    names = [f"Director {i}" for i in range(1500)]   # 1500 distinct, well over the ceiling
    df = pd.DataFrame({"director": [random.choice(names) for _ in range(6000)]})
    director = next(c for c in profile_dataframe(df).columns if c.name == "director")
    assert director.role == "unusable"


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


def test_multi_value_country_is_usable_categorical():
    import pandas as pd, random
    from profile import profile_dataframe
    random.seed(1)
    n = 400
    countries = ["United States", "India", "United Kingdom", "Japan", "Canada", "France", "Spain", "Mexico"]
    def multi(pool, lo, hi): return ", ".join(random.sample(pool, random.randint(lo, hi)))
    df = pd.DataFrame({
        "country": [multi(countries, 1, 3) for _ in range(n)],
        "cast": [", ".join(f"Actor{random.randint(0,3000)}" for _ in range(4)) for _ in range(n)],
        "description": [f"A unique, long sentence number {i} describing the title at length." for i in range(n)],
        "rating": [random.choice(["TV-MA", "PG-13", "R", "TV-14"]) for _ in range(n)],
    })
    prof = profile_dataframe(df)
    by = {c.name: c for c in prof.columns}
    assert by["country"].role == "categorical" and by["country"].multi_value is True
    assert by["country"].delimiter == ", "
    assert by["cast"].role == "unusable"
    assert by["description"].role == "unusable"
    assert by["rating"].role == "categorical" and by["rating"].multi_value is False
