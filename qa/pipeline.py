"""Fidelity helper: run a DataFrame through the EXACT production generation
sequence used by POST /generate-report, returning the stored report dict (or a
captured exception). This is the harness's anti-drift guard — it imports the
real endpoint helpers and the real ReportGenerator, so any production change in
sampling, models, or report assembly flows through automatically.

Mirrors src/api/main.py generate_report() lines ~337-392 and
report_generator.ReportGenerator.build_report().
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv

# Make the production package importable when run outside pytest (from repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_API_DIR = _REPO_ROOT / "src" / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

# Same intent as main.py:38 (pull ANTHROPIC_API_KEY etc. from .env). We point at
# the repo-root .env explicitly: bare load_dotenv()'s caller-frame walk is
# unreliable when this module is imported transitively, so an explicit path keeps
# the key load deterministic regardless of how the harness is launched.
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()  # fallback: also honor any .env discovered from CWD / process env

from claude_client import ClaudeClient                       # noqa: E402
from llm_config import MODEL_NARRATIVE, MODEL_SELECTION      # noqa: E402
# Sampling lives in its own lean module (NOT main.py): importing main would drag in
# the whole FastAPI/stripe/supabase web stack — slow and unnecessary for the harness.
from sampling import MAX_ANALYSIS_ROWS, sample_for_analysis  # noqa: E402  (the real prod sampling)
from profile import profile_dataframe                        # noqa: E402
from report_generator import ReportGenerator                 # noqa: E402


@dataclass
class RunResult:
    """Outcome of one dataset run: the stored report dict OR an error string,
    plus timing and the post-sampling row/col counts actually analyzed."""
    name: str
    report: Optional[dict]
    error: Optional[str]
    elapsed_ms: int
    rows_analyzed: int
    cols_analyzed: int
    was_sampled: bool
    original_rows: int
    profile_text: str = ""   # the to_text() of the profile the report was built from (for the judge)


def _build_claude() -> ClaudeClient:
    """Construct the REAL Anthropic client exactly like main.get_claude_client()."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (expected in .env)")
    return ClaudeClient(api_key=api_key)


def run_report(df: pd.DataFrame, custom_prompt: str | None = None, name: str = "") -> RunResult:
    """Reproduce the /generate-report body and return a RunResult.

    Sequence (main.py): lower-case columns -> sample_for_analysis -> 2-col/1-row
    guards -> profile_dataframe -> ReportGenerator(...) -> build_report(deep=False)
    -> prepend sampling note -> return report.model_dump().
    """
    started = time.perf_counter()
    original_rows = int(df.shape[0])
    try:
        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]          # main.py:337
        df, was_sampled, total_rows = sample_for_analysis(df)       # main.py:339

        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns to chart.")   # main.py:343-344
        if df.shape[0] < 1:
            raise ValueError("File has no data rows.")                        # main.py:345-346

        profile = profile_dataframe(df)                            # main.py:358
        profile_text = profile.to_text()   # exact profile the report is built from (for the judge)
        claude = _build_claude()
        gen = ReportGenerator(
            profile=profile, df=df, claude=claude,
            model_selection=MODEL_SELECTION, model_narrative=MODEL_NARRATIVE,
            custom_prompt=custom_prompt,
        )                                                          # main.py:359-363
        report = gen.build_report(deep=False)                      # main.py:364

        if was_sampled:                                            # main.py:390-392
            note = (f"Analyzed a representative random sample of "
                    f"{MAX_ANALYSIS_ROWS:,} of {total_rows:,} rows.")
            report.data_quality = [note] + list(report.data_quality or [])

        report_dict: dict[str, Any] = report.model_dump()         # main.py:412 (stored shape)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return RunResult(
            name=name, report=report_dict, error=None, elapsed_ms=elapsed_ms,
            rows_analyzed=int(df.shape[0]), cols_analyzed=int(df.shape[1]),
            was_sampled=was_sampled, original_rows=total_rows,
            profile_text=profile_text,
        )
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return RunResult(
            name=name, report=None,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            elapsed_ms=elapsed_ms, rows_analyzed=0, cols_analyzed=0,
            was_sampled=False, original_rows=original_rows,
        )
