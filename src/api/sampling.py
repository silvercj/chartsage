"""Row-sampling for analysis — the single source of truth shared by the API
endpoint (main.py) and the QA harness (qa/pipeline.py).

Kept in its own lean, pandas-only module (no FastAPI/stripe/web imports) so tools
can reuse the *real* production sampling without dragging in the whole web app.
"""
import os

import pandas as pd

# Above this row count we analyze a deterministic random sample (the analysis is
# column-driven, so a representative sample is statistically faithful while keeping
# memory/latency bounded). Env-overridable.
MAX_ANALYSIS_ROWS = int(os.environ.get("MAX_ANALYSIS_ROWS", "50000"))


def sample_for_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, int]:
    """Bound rows for analysis. Returns (frame, was_sampled, original_row_count).
    Deterministic (fixed seed) so generate-more / deepen — which re-read the stored
    CSV — analyze the same rows."""
    total = len(df)
    if total > MAX_ANALYSIS_ROWS:
        return df.sample(n=MAX_ANALYSIS_ROWS, random_state=0).reset_index(drop=True), True, total
    return df, False, total
