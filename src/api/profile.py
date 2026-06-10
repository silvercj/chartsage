"""DataFrame profiling for AI chart selection.

profile_dataframe(df) returns a DataProfile that captures the structural facts
Claude needs to pick good charts: column roles, basic stats, correlations, anomalies.
The raw DataFrame is never sent to Claude — only this profile.
"""
import re
import numpy as np
import pandas as pd
from schemas import ColumnInfo, DataProfile
from multi_value import detect_multi_value, explode_multi_value


_IDENTIFIER_SUFFIXES = ("_id", "_code", "_uuid", "_key")
# Low-cardinality categoricals get their full top-5 list (small enough to enumerate).
_LOW_CARDINALITY_MAX = 50
# A high-cardinality object column is still chartable as a top-N (frequency bar /
# treemap roll the long tail into "Other") AS LONG AS its values actually repeat.
# Two guards keep this conservative:
#   - distinct count must be well below the row count (values recur, not free text);
#   - distinct count must stay under an absolute ceiling that comfortably covers
#     real category sets (countries, genres, US states) but excludes columns with
#     thousands of distinct values (names, directors) that read as free text.
# A near-unique column (a movie title, a description) trips both and stays "unusable".
_CATEGORICAL_MAX_DISTINCT_RATIO = 0.5
_CATEGORICAL_MAX_DISTINCT = 200
_NON_NEGATIVE_KEYWORDS = ("duration", "count", "quantity", "age", "price",
                          "amount", "revenue", "sales", "cost")
_DATE_KEYWORDS = ("date", "time", "created", "updated", "start", "end", "timestamp")

# A numeric column whose name IS one of these (e.g. "Year", "Decade") is a time/ordinal
# axis. Matched on the WHOLE name (letters only) — not as a substring — so a rate like
# "majors_per_year" or "goals_per_month", which merely *contains* a temporal word, is
# never mistaken for the axis (that bug made the hurricanes report flat-line).
_TEMPORAL_NAMES = {
    "year", "years", "yr", "date", "dates", "decade", "decades", "season", "seasons",
    "period", "periods", "quarter", "quarters", "month", "months", "week", "weeks",
}


def _has_temporal_name(name: str) -> bool:
    return "".join(ch for ch in name.lower() if ch.isalpha()) in _TEMPORAL_NAMES


def _is_temporal_ordinal(name: str, series: pd.Series) -> bool:
    """True for a numeric time/ordinal axis: a temporally-named column with (near-)one
    row per value — the x of a time series (e.g. Year in a one-row-per-year table)."""
    if not _has_temporal_name(name):
        return False
    s = pd.to_numeric(series, errors="coerce").dropna()
    return len(s) >= 4 and s.nunique() >= 0.9 * len(s)


def _is_identifier(name: str, dtype: str, cardinality: int, row_count: int) -> bool:
    lower = name.lower()
    if any(lower.endswith(s) for s in _IDENTIFIER_SUFFIXES):
        return True
    # Cardinality-based: only integer columns where every value is unique
    # Require a minimum of 50 rows to avoid misclassifying small analytic ranges (e.g. x=0..19)
    # A temporally-named column (Year, Decade) is exempt: a long yearly series is all-unique
    # ints too, but it's the time axis, not a row ID.
    if (np.issubdtype(np.dtype(dtype), np.integer)
            and cardinality == row_count
            and row_count >= 50
            and not _has_temporal_name(name)):
        return True
    return False


def is_identifier_column(name: str, series: pd.Series, row_count: int) -> bool:
    """Whether a column reads as a row ID (an *_id-style name, or all-unique integers).
    Public wrapper for executors that must keep IDs out of numeric computations (e.g.
    the correlation heatmap), mirroring the profile's role detection."""
    return _is_identifier(
        name, str(series.dtype), int(series.nunique(dropna=True)), row_count,
    )


def _is_date(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    nonnull = series.dropna()
    if len(nonnull) == 0:
        return False
    # Don't try to coerce purely numeric data to datetime
    if pd.api.types.is_numeric_dtype(nonnull):
        return False
    try:
        parsed = pd.to_datetime(nonnull, errors="coerce")
    except Exception:
        return False
    return (parsed.notna().sum() / len(nonnull)) >= 0.8


def _profile_column(name: str, series: pd.Series, row_count: int) -> ColumnInfo:
    dtype = str(series.dtype)
    cardinality = int(series.nunique(dropna=True))
    null_count = int(series.isna().sum())

    if _is_identifier(name, dtype, cardinality, row_count):
        return ColumnInfo(name=name, dtype=dtype, role="identifier",
                          cardinality=cardinality, null_count=null_count)

    if _is_date(series):
        parsed = pd.to_datetime(series, errors="coerce").dropna()
        return ColumnInfo(
            name=name, dtype=dtype, role="date",
            cardinality=cardinality, null_count=null_count,
            min_date=parsed.min().isoformat() if len(parsed) else None,
            max_date=parsed.max().isoformat() if len(parsed) else None,
        )

    if pd.api.types.is_numeric_dtype(series):
        nonnull = series.dropna()
        return ColumnInfo(
            name=name, dtype=dtype, role="numeric",
            cardinality=cardinality, null_count=null_count,
            min=float(nonnull.min()) if len(nonnull) else None,
            max=float(nonnull.max()) if len(nonnull) else None,
            mean=float(nonnull.mean()) if len(nonnull) else None,
            median=float(nonnull.median()) if len(nonnull) else None,
            std=float(nonnull.std()) if len(nonnull) > 1 else None,
            temporal_ordinal=_is_temporal_ordinal(name, series),
        )

    # Low-cardinality categorical: enumerate the full top-5.
    if cardinality <= _LOW_CARDINALITY_MAX:
        top = series.value_counts(dropna=True).head(5)
        return ColumnInfo(
            name=name, dtype=dtype, role="categorical",
            cardinality=cardinality, null_count=null_count,
            top_values=[(k, int(v)) for k, v in top.items()],
        )

    # Multi-value / multi-label column ("United States, India" or "Drama|Comedy"):
    # split into atoms and treat as a usable categorical (charted top-N of the atoms).
    # Run BEFORE the high-cardinality branch: a delimited column whose raw combinations
    # happen to fall under the cardinality ceiling must still be FLAGGED multi_value, so
    # the chart executors explode it into atoms rather than charting whole combinations.
    # detect_multi_value() is itself conservative (atom count/length/ratio guards), so
    # free text like a `cast` list of thousands of distinct actors stays unusable below.
    delim = detect_multi_value(series)
    if delim is not None:
        atoms = explode_multi_value(series, delim)
        top = atoms.value_counts().head(5)
        return ColumnInfo(
            name=name, dtype=dtype, role="categorical",
            cardinality=int(atoms.nunique()), null_count=null_count,
            top_values=[(k, int(v)) for k, v in top.items()],
            multi_value=True, delimiter=delim,
        )

    # High-cardinality but its values REPEAT (distinct count well below the row count):
    # still chartable as a top-N — frequency_bar/treemap roll the long tail into "Other".
    # This rescues catalog columns like country/genre/director from being dropped, while
    # near-unique free text (a title, ratio ≈ 1.0) falls through to "unusable" below.
    nonnull_count = int(series.notna().sum())
    distinct_ratio = (cardinality / nonnull_count) if nonnull_count else 1.0
    if cardinality <= _CATEGORICAL_MAX_DISTINCT and distinct_ratio <= _CATEGORICAL_MAX_DISTINCT_RATIO:
        top = series.value_counts(dropna=True).head(5)
        return ColumnInfo(
            name=name, dtype=dtype, role="categorical",
            cardinality=cardinality, null_count=null_count,
            top_values=[(k, int(v)) for k, v in top.items()],
        )

    return ColumnInfo(
        name=name, dtype=dtype, role="unusable",
        cardinality=cardinality, null_count=null_count,
        unusable_reason=f"object column with {cardinality} unique values, too high-cardinality to chart",
    )


def _compute_correlations(df: pd.DataFrame, columns: list[ColumnInfo]) -> dict[str, float]:
    numeric_cols = [c.name for c in columns if c.role == "numeric"]
    if len(numeric_cols) < 2:
        return {}
    corr_matrix = df[numeric_cols].corr(numeric_only=True)
    result: dict[str, float] = {}
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1:]:
            r = corr_matrix.loc[c1, c2]
            if pd.notna(r) and abs(r) >= 0.3:
                result[f"{c1}||{c2}"] = float(round(r, 3))
    return result


def _detect_anomalies(df: pd.DataFrame, columns: list[ColumnInfo]) -> list[str]:
    anomalies: list[str] = []
    now = pd.Timestamp.utcnow().tz_localize(None)

    for col in columns:
        lower = col.name.lower()

        if col.role == "numeric" and any(kw in lower for kw in _NON_NEGATIVE_KEYWORDS):
            if col.min is not None and col.min < 0:
                anomalies.append(
                    f"{col.name} contains negative values (min={col.min}); "
                    f"column name suggests it should be non-negative."
                )

        if col.role == "date" and col.max_date:
            try:
                max_ts = pd.Timestamp(col.max_date).tz_localize(None)
                if max_ts > now + pd.Timedelta(days=365):
                    anomalies.append(f"{col.name} contains future dates (max={col.max_date}).")
            except Exception:
                pass

        if col.null_count > 0.95 * df.shape[0] and df.shape[0] > 0:
            anomalies.append(f"{col.name} is >95% null ({col.null_count}/{df.shape[0]}); unusable.")

        if col.role == "numeric" and col.cardinality <= 2:
            anomalies.append(f"{col.name} has cardinality {col.cardinality}; behaves like a boolean.")

        if (col.role == "numeric" and col.std is not None and col.std > 0
                and col.max is not None and col.mean is not None
                and col.max > col.mean + 10 * col.std):
            anomalies.append(
                f"{col.name} has an extreme outlier (max={col.max}, mean={col.mean}, std={col.std}); "
                f"histograms may show empty bins."
            )

    return anomalies


def profile_dataframe(df: pd.DataFrame) -> DataProfile:
    """Build a DataProfile from a pandas DataFrame."""
    row_count = int(df.shape[0])
    columns = [_profile_column(name, df[name], row_count) for name in df.columns]
    correlations = _compute_correlations(df, columns)
    anomalies = _detect_anomalies(df, columns)
    return DataProfile(
        row_count=row_count,
        columns=columns,
        correlations=correlations,
        anomalies=anomalies,
    )
