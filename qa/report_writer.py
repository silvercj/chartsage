"""Render a QA run into report.md (scannable) + summary.json (machine-readable)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _det_issues(payload: dict) -> list[dict]:
    return payload.get("deterministic_issues") or []


def _judge(payload: dict) -> dict | None:
    return payload.get("judge")


def classify(payload: dict) -> str:
    """PASS / WARN / FAIL per the spec's rule."""
    issues = _det_issues(payload)
    if any(i.get("severity") == "fail" for i in issues):
        return "FAIL"
    judge = _judge(payload)
    if judge:
        charts = judge.get("charts") or []
        if any((not c.get("makes_sense", True)) or c.get("severity") == "fail" for c in charts):
            return "FAIL"
    if any(i.get("severity") == "warn" for i in issues):
        return "WARN"
    if judge:
        if any(c.get("severity") == "warn" for c in (judge.get("charts") or [])):
            return "WARN"
        if judge.get("narrative_matches") is False:
            return "WARN"
    return "WARN" if (payload.get("report") is None and not issues) else "PASS"


def _counts(payload: dict) -> tuple[int, int]:
    issues = _det_issues(payload)
    fails = sum(1 for i in issues if i.get("severity") == "fail")
    warns = sum(1 for i in issues if i.get("severity") == "warn")
    judge = _judge(payload)
    if judge:
        for c in (judge.get("charts") or []):
            if (not c.get("makes_sense", True)) or c.get("severity") == "fail":
                fails += 1
            elif c.get("severity") == "warn":
                warns += 1
    return fails, warns


def write_report(run_dir: str | Path, per_dataset_results: list[dict]) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    statuses = {p["name"]: classify(p) for p in per_dataset_results}
    totals = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for s in statuses.values():
        totals[s] += 1

    # ---- report.md ----
    lines = [f"# ChartSage QA run — {run_dir.name}", "",
             f"PASS {totals['PASS']} · WARN {totals['WARN']} · FAIL {totals['FAIL']} "
             f"(of {len(per_dataset_results)} datasets)", "",
             "| dataset | status | fails | warns | charts | ms |",
             "|---|---|---|---|---|---|"]
    for p in per_dataset_results:
        fails, warns = _counts(p)
        n_charts = len((p.get("report") or {}).get("charts") or []) if p.get("report") else 0
        lines.append(f"| {p['name']} | {statuses[p['name']]} | {fails} | {warns} | "
                     f"{n_charts} | {p.get('elapsed_ms', 0)} |")
    lines.append("")

    # Per-dataset detail (FAIL/WARN first).
    order = sorted(per_dataset_results,
                   key=lambda p: {"FAIL": 0, "WARN": 1, "PASS": 2}[statuses[p["name"]]])
    for p in order:
        name = p["name"]
        lines.append(f"## {name} — {statuses[name]}")
        if p.get("error"):
            first = p["error"].splitlines()[0] if p["error"] else ""
            lines.append(f"- generation error: `{first}`")
        for i in _det_issues(p):
            tag = f" (chart {i['chart_id']})" if i.get("chart_id") else ""
            lines.append(f"- [{i['severity']}] {i['code']}: {i['message']}{tag}")
        judge = _judge(p)
        if judge:
            if judge.get("narrative_matches") is False:
                lines.append("- [judge] narrative does not match the charts")
            for c in (judge.get("charts") or []):
                if (not c.get("makes_sense", True)) or c.get("severity") in ("warn", "fail"):
                    lines.append(f"- [judge:{c.get('severity')}] chart {c.get('chart_id')}: "
                                 f"{c.get('issue')}")
        lines.append("")
    (run_dir / "report.md").write_text("\n".join(lines))

    # ---- summary.json ----
    summary: dict[str, Any] = {
        "run": run_dir.name,
        "totals": totals,
        "datasets": {
            p["name"]: {
                "status": statuses[p["name"]],
                "fails": _counts(p)[0],
                "warns": _counts(p)[1],
                "elapsed_ms": p.get("elapsed_ms", 0),
                "was_sampled": p.get("was_sampled", False),
            }
            for p in per_dataset_results
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
