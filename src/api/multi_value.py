"""Detect and explode delimited multi-value / multi-label columns (e.g. a
Netflix `country`="United States, India" or `listed_in`="Dramas, Comedies").
Shared by the profiler (to mark the column usable) and the chart executors
(to split → count individual atoms). Conservative on purpose: never treat a
free-text column as multi-value."""
import pandas as pd

_MV_DELIMITERS = ["; ", " | ", "|", ", ", ","]
_MV_MAX_ATOMS = 50
_MV_MAX_ATOM_LEN = 30
_MV_MAX_ATOM_LEN_HARD = 60
_MV_MIN_MULTI_FRAC = 0.6
_MV_MAX_ATOM_RATIO = 0.5
_MV_MIN_ROWS = 10


def detect_multi_value(series: pd.Series) -> str | None:
    s = series.dropna().astype(str)
    if len(s) < _MV_MIN_ROWS:
        return None
    for delim in _MV_DELIMITERS:
        parts = s.str.split(delim, regex=False)
        multi_frac = (parts.str.len() >= 2).mean()
        if multi_frac < _MV_MIN_MULTI_FRAC:
            continue
        atoms = parts.explode().str.strip()
        atoms = atoms[atoms != ""]
        total = len(atoms)
        if total == 0:
            continue
        distinct = atoms.nunique()
        if distinct == 0 or distinct > _MV_MAX_ATOMS:
            continue
        lengths = atoms.str.len()
        if lengths.mean() > _MV_MAX_ATOM_LEN or lengths.max() > _MV_MAX_ATOM_LEN_HARD:
            continue
        if (distinct / total) > _MV_MAX_ATOM_RATIO:
            continue
        return delim
    return None


def explode_multi_value(series: pd.Series, delimiter: str) -> pd.Series:
    atoms = series.dropna().astype(str).str.split(delimiter, regex=False).explode().str.strip()
    return atoms[atoms != ""]
