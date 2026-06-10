"""Pure data-processing helpers shared across executors."""
import logging
import numpy as np
import pandas as pd


_MONTHS = ["january", "february", "march", "april", "may", "june",
           "july", "august", "september", "october", "november", "december"]
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _positions(names: list[str], extra: dict[str, int] | None = None) -> dict[str, int]:
    pos = {}
    for i, n in enumerate(names):
        pos[n] = i
        pos[n[:3]] = i        # common 3-letter abbreviations (jan, mon, …)
    pos.update(extra or {})
    return pos


_SEQUENCES = (
    _positions(_MONTHS, {"sept": 8}),
    _positions(_WEEKDAYS, {"tues": 1, "thurs": 3}),
    {f"q{i + 1}": i for i in range(4)},
)


def natural_category_order(values: list) -> list | None:
    """Reorder month / weekday / quarter names chronologically, or None when the values
    aren't such a sequence. Category axes default to value/count order, which renders
    months alphabetically (Apr, Aug, Dec…) or by revenue — noise for a time-like axis.
    Conservative: every value must belong to ONE known sequence, and at least 3 distinct
    values are required ('May'/'Mar' alone could be people's names)."""
    keys = [str(v).strip().lower() for v in values]
    if len(set(keys)) < 3:
        return None
    for pos in _SEQUENCES:
        if all(k in pos for k in keys):
            return [v for _, v in sorted(zip(keys, values), key=lambda kv: pos[kv[0]])]
    return None


def compute_group_count(df: pd.DataFrame, group_col: str) -> tuple[list, list]:
    """Return (categories, counts) sorted by count descending."""
    if group_col not in df.columns:
        return [], []
    counts = df[group_col].value_counts()
    return counts.index.tolist(), counts.values.tolist()


def compute_histogram_bins_and_freqs(
    values: pd.Series, max_bins: int = 20, min_bins: int = 5
) -> tuple[list[str], list[int]]:
    """Trimmed-IQR histogram binning robust to outliers.

    Drops values outside [Q1 - 3·IQR, Q3 + 3·IQR] before binning so a single
    extreme outlier doesn't pull the bin range and leave most bins empty.
    Falls back to full range if trimming removes >20% of data.
    """
    nonnull = pd.to_numeric(values, errors="coerce").dropna()
    if len(nonnull) == 0:
        return [], []

    if nonnull.nunique() < 2:
        return [], []

    q1, q3 = nonnull.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr > 0:
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        trimmed = nonnull[(nonnull >= lo) & (nonnull <= hi)]
        if len(trimmed) < 0.8 * len(nonnull):
            trimmed = nonnull  # too many dropped, use full range
    else:
        trimmed = nonnull

    n = len(trimmed)
    target_bins = max(min_bins, min(max_bins, int(np.ceil(2 * n ** (1 / 3)))))

    counts, edges = np.histogram(trimmed, bins=target_bins)
    labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(edges) - 1)]
    return labels, counts.tolist()
