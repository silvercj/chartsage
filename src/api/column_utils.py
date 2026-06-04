"""Column-name normalization — shared by the API endpoint (main.py) and the QA
harness (qa/pipeline.py), kept lean (pandas-only) so tools can reuse it.

Lower-cases headers (the analysis is case-insensitive) and then de-duplicates any
names that COLLIDE after lower-casing. Without this, a file with both 'Value' and
'value' headers (common in Excel exports, or 'Date'/'date') produces two
identically-named columns, and df['value'] returns a *DataFrame* (not a Series) —
which crashes the profiler (`series.dtype`) and every executor. Collisions are
mangled pandas-style: 'value', 'value.1', 'value.2'.
"""
import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with lower-cased, collision-free column names."""
    seen: dict[str, int] = {}
    new_cols: list[str] = []
    for col in df.columns:
        name = str(col).lower()
        if name in seen:
            seen[name] += 1
            candidate = f"{name}.{seen[name]}"
            # Guard against the suffixed name itself already existing (e.g. a real
            # 'value.1' column alongside two 'value' columns).
            while candidate in seen:
                seen[name] += 1
                candidate = f"{name}.{seen[name]}"
            seen[candidate] = 0
            name = candidate
        else:
            seen[name] = 0
        new_cols.append(name)
    out = df.copy()
    out.columns = new_cols
    return out
