"""QA harness runner: feed synthetic generators + dropped-in CSVs through the
real production pipeline, validate (deterministic + judge), and write results.

Usage (from repo root):
    venv/bin/python qa/run_eval.py [--only synthetic|real|<name>] [--no-judge] [--limit N]

Outputs land in qa/results/<timestamp>/:
    <dataset>.json   per-dataset: run meta, report, deterministic issues, judge verdict
    report.md        scannable PASS/WARN/FAIL roll-up
    summary.json     machine-readable roll-up
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_QA_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _QA_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))   # so `import qa.*` works when run as a script

from qa.generators import SYNTHETIC          # noqa: E402
from qa.pipeline import RunResult, run_report  # noqa: E402


def _discover_real() -> list[tuple[str, pd.DataFrame]]:
    """Auto-discover qa/datasets/*.csv (name = filename stem)."""
    out: list[tuple[str, pd.DataFrame]] = []
    for csv in sorted((_QA_DIR / "datasets").glob("*.csv")):
        try:
            df = pd.read_csv(csv)
        except UnicodeDecodeError:
            df = pd.read_csv(csv, encoding="latin1")
        out.append((csv.stem, df))
    return out


def _collect(only: str | None) -> list[tuple[str, pd.DataFrame]]:
    synthetic = [gen() for gen in SYNTHETIC]
    real = _discover_real()
    if only == "synthetic":
        return synthetic
    if only == "real":
        return real
    if only:  # a specific dataset name
        both = synthetic + real
        picked = [(n, d) for (n, d) in both if n == only]
        if not picked:
            raise SystemExit(f"No dataset named {only!r} (synthetic or in qa/datasets/).")
        return picked
    return synthetic + real


def _to_jsonable(result: RunResult, det_issues, verdict) -> dict:
    return {
        "name": result.name,
        "error": result.error,
        "elapsed_ms": result.elapsed_ms,
        "rows_analyzed": result.rows_analyzed,
        "cols_analyzed": result.cols_analyzed,
        "was_sampled": result.was_sampled,
        "original_rows": result.original_rows,
        "report": result.report,
        "deterministic_issues": [dataclasses.asdict(i) for i in det_issues],
        "judge": dataclasses.asdict(verdict) if verdict is not None else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ChartSage QA / Eval Harness")
    parser.add_argument("--only", default=None,
                        help="synthetic | real | <dataset-name>")
    parser.add_argument("--no-judge", action="store_true",
                        help="deterministic checks only (free, no Haiku judge calls)")
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N datasets")
    args = parser.parse_args(argv)

    # Lazy imports so a JSON-only smoke works before these modules exist.
    from qa.validators import validate            # noqa: E402
    judge_report = None
    if not args.no_judge:
        from qa.judge import judge_report         # noqa: E402
    from qa.report_writer import write_report     # noqa: E402

    datasets = _collect(args.only)
    if args.limit is not None:
        datasets = datasets[: args.limit]

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = _QA_DIR / "results" / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    per_dataset: list[dict] = []
    for name, df in datasets:
        print(f"[qa] running {name} ({df.shape[0]} rows x {df.shape[1]} cols) ...", flush=True)
        result = run_report(df, name=name)
        # Validate against the EXACT frame the charts were built from (post-sample/
        # lowercase). For big sampled datasets, recomputing counts/sums from the raw df
        # would falsely mismatch the chart values (which came from the 50k sample).
        check_df = result.analyzed_df if result.analyzed_df is not None else df
        det_issues = validate(check_df, result.report)   # validate() handles report=None
        verdict = None
        if judge_report is not None and result.report is not None and result.profile_text:
            try:
                verdict = judge_report(result.profile_text, result.report)
            except Exception as e:  # judge must never sink the whole run
                print(f"[qa] judge failed for {name}: {e}", flush=True)
        payload = _to_jsonable(result, det_issues, verdict)
        (run_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))
        per_dataset.append(payload)

    write_report(run_dir, per_dataset)
    print(f"[qa] done -> {run_dir}/report.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
