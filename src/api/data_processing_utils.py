"""Pure data-processing helpers shared across executors."""
import logging
import numpy as np
import pandas as pd


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
