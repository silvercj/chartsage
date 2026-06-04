"""Deterministic validators: pure functions over (df, report_dict) returning a
flat list of Issue. No network, no LLM. These are the harness's hard gate.

`validate(df, report)` runs every check and concatenates their issues. A report
of None means generation crashed (the pipeline captured an exception).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from report_generator import MIN_CHARTS_FOR_NO_FALLBACK  # the real selection floor (==3)


@dataclass
class Issue:
    severity: str            # 'fail' | 'warn'
    code: str
    message: str
    chart_id: Optional[str] = None


# ---- small helpers ---------------------------------------------------------

def _charts(report: dict) -> list[dict]:
    return list(report.get("charts") or [])


def _spec(chart: dict) -> dict:
    return chart.get("spec") or {}


def _numeric_ys(spec: dict) -> list[float]:
    """Flatten a chart's y values to floats (drop non-finite/non-numeric).
    Handles flat y, series-with-'y', series-with-'data', and treemap nodes."""
    out: list[float] = []

    def _push(v: Any):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return
        if math.isfinite(f):
            out.append(f)

    if spec.get("y"):
        for v in spec["y"]:
            _push(v)
    for s in (spec.get("series") or []):
        for key in ("y", "data"):
            for v in (s.get(key) or []):
                _push(v)
        if "value" in s:
            _push(s.get("value"))
    for n in (spec.get("nodes") or []):
        _push(n.get("value"))
        for child in (n.get("children") or []):
            _push(child.get("value"))
    return out


def _approx(a: float, b: float, *, rel: float = 0.02, abs_: float = 1e-6) -> bool:
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


# ---- checks ----------------------------------------------------------------

def check_generation_error(df: pd.DataFrame, report: Optional[dict]) -> list[Issue]:
    if report is None:
        return [Issue("fail", "generation_error",
                      "Report generation raised an exception (no report produced).")]
    return []


def check_degenerate_charts(df: pd.DataFrame, report: dict) -> list[Issue]:
    issues: list[Issue] = []
    for chart in _charts(report):
        spec = _spec(chart)
        cid = chart.get("chart_id")
        kind = spec.get("kind")
        has_xy = bool(spec.get("x")) and bool(spec.get("y"))
        has_series = bool(spec.get("series"))
        has_nodes = bool(spec.get("nodes"))
        if not (has_xy or has_series or has_nodes):
            issues.append(Issue("fail", "degenerate_chart",
                                f"[{kind}] '{spec.get('title')}' has empty x/y, series and nodes.",
                                cid))
            continue
        # Line charts need >=2 distinct x points (the year-as-int bug class).
        if kind == "line":
            xs = spec.get("x") or []
            if xs and len({str(v) for v in xs}) < 2:
                issues.append(Issue("fail", "degenerate_chart",
                                    f"[line] '{spec.get('title')}' has <2 distinct x points "
                                    f"({len(xs)} points, {len(set(map(str, xs)))} distinct).", cid))
            # series-form line: every series collapsed to one distinct period
            for s in (spec.get("series") or []):
                sx = s.get("x") or []
                if sx and len({str(v) for v in sx}) < 2:
                    issues.append(Issue("fail", "degenerate_chart",
                                        f"[line] '{spec.get('title')}' series "
                                        f"'{s.get('name')}' has <2 distinct x points.", cid))
        # Numeric y degeneracy: all-NaN / all-zero / all-identical.
        ys = _numeric_ys(spec)
        raw_count = len(spec.get("y") or []) + sum(
            len(s.get("y") or s.get("data") or []) for s in (spec.get("series") or []))
        if raw_count > 0 and len(ys) == 0:
            issues.append(Issue("fail", "degenerate_chart",
                                f"[{kind}] '{spec.get('title')}' has all-NaN y values.", cid))
        elif len(ys) >= 2:
            if all(v == 0 for v in ys):
                issues.append(Issue("fail", "degenerate_chart",
                                    f"[{kind}] '{spec.get('title')}' has all-zero y values.", cid))
            elif len(set(ys)) == 1:
                issues.append(Issue("warn", "degenerate_chart",
                                    f"[{kind}] '{spec.get('title')}' has all-identical y "
                                    f"values ({ys[0]}).", cid))
    return issues


def check_chart_data_consistency(df: pd.DataFrame, report: dict) -> list[Issue]:
    """Spot-check that bar/line values are derivable from the source df.

    Only the unambiguous cases are checked (everything else is skipped, not failed):
      - frequency bar: y_display_type == 'count' and exactly one source column ->
        compare reported (x->y) to df[col].value_counts().
      - sum aggregation/line: y_label starts with 'Sum of <col>' and there is a
        clear group/date column -> compare reported (x->y) to a groupby-sum.
    A mismatch on a checkable case is a 'fail'.
    """
    issues: list[Issue] = []
    cols = set(df.columns)
    for chart in _charts(report):
        spec = _spec(chart)
        cid = chart.get("chart_id")
        kind = spec.get("kind")
        x = spec.get("x")
        y = spec.get("y")
        srcs = [c for c in (spec.get("source_columns") or []) if c in cols]
        if kind not in ("bar", "line") or not x or not y or len(x) != len(y):
            continue
        reported = {}
        for xv, yv in zip(x, y):
            try:
                reported[str(xv)] = float(yv)
            except (TypeError, ValueError):
                reported = {}
                break
        if not reported:
            continue

        # Case 1: frequency count bar over a single categorical column.
        if (kind == "bar" and spec.get("y_display_type") == "count" and len(srcs) == 1):
            col = srcs[0]
            vc = df[col].dropna().astype(str).value_counts()
            mism = [k for k, v in reported.items()
                    if k in vc.index and not _approx(v, float(vc[k]))]
            if mism:
                issues.append(Issue(
                    "fail", "chart_data_mismatch",
                    f"[bar] '{spec.get('title')}' count(s) for {mism[:3]} "
                    f"don't match df['{col}'].value_counts().", cid))
            continue

        # Case 2: sum aggregation (bar or line) of <value_col> by a group/date col.
        m = re.match(r"sum of (.+)", str(spec.get("y_label", "")).strip(), re.IGNORECASE)
        if m:
            value_col = m.group(1).strip().lower()
            group_candidates = [c for c in srcs if c != value_col]
            if value_col in cols and len(group_candidates) == 1:
                gcol = group_candidates[0]
                work = df[[gcol, value_col]].copy()
                work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
                work = work.dropna()
                grouped = work.groupby(work[gcol].astype(str))[value_col].sum()
                mism = [k for k, v in reported.items()
                        if k in grouped.index and not _approx(v, float(grouped[k]))]
                if mism:
                    issues.append(Issue(
                        "fail", "chart_data_mismatch",
                        f"[{kind}] '{spec.get('title')}' sum(s) for {mism[:3]} "
                        f"don't match a groupby-sum of '{value_col}' by '{gcol}'.", cid))
    return issues


_YEAR_LABEL_RE = re.compile(r"\b(year|years|count|total)\b", re.IGNORECASE)


def check_kpi_sanity(df: pd.DataFrame, report: dict) -> list[Issue]:
    issues: list[Issue] = []
    for m in (report.get("key_metrics") or []):
        label = str(m.get("label", ""))
        raw = m.get("value")
        try:
            val = float(raw)
        except (TypeError, ValueError):
            issues.append(Issue("fail", "kpi_non_finite",
                                f"KPI '{label}' has a non-numeric value ({raw!r})."))
            continue
        if not math.isfinite(val):
            issues.append(Issue("fail", "kpi_non_finite",
                                f"KPI '{label}' is non-finite ({val})."))
            continue
        # Year-look-alike: a year/count/total metric whose value IS a calendar year.
        if _YEAR_LABEL_RE.search(label) and float(val).is_integer() and 1900 <= val <= 2100:
            issues.append(Issue("warn", "kpi_year_lookalike",
                                f"KPI '{label}' = {int(val)} looks like a calendar year, "
                                f"not a {label.lower()} (e.g. the '2,022 years covered' slip)."))
    return issues


def check_narrative(df: pd.DataFrame, report: dict) -> list[Issue]:
    summary = str(report.get("summary") or "").strip()
    if not summary:
        return [Issue("fail", "narrative_missing", "Report summary is empty.")]
    return []


def check_chart_count_floor(df: pd.DataFrame, report: dict) -> list[Issue]:
    # WARN (not fail): production's fallback usually guarantees the floor, but tiny
    # or degenerate inputs (single-row, all-duplicate) legitimately yield fewer
    # charts. A low count is worth a human's eyes, not a hard regression failure.
    n = len(_charts(report))
    if n < MIN_CHARTS_FOR_NO_FALLBACK:
        return [Issue("warn", "chart_count_below_floor",
                      f"Only {n} chart(s); the selection floor is "
                      f"{MIN_CHARTS_FOR_NO_FALLBACK} (fallback normally guarantees it).")]
    return []


def check_axis_labels(df: pd.DataFrame, report: dict) -> list[Issue]:
    """x_label/y_label should be present. Some kinds (pie/treemap/box/heatmap)
    legitimately leave an axis blank, so a missing label is a 'warn', not a 'fail'."""
    issues: list[Issue] = []
    for chart in _charts(report):
        spec = _spec(chart)
        cid = chart.get("chart_id")
        if not str(spec.get("x_label") or "").strip():
            issues.append(Issue("warn", "missing_axis_label",
                                f"[{spec.get('kind')}] '{spec.get('title')}' has no x_label.", cid))
        if not str(spec.get("y_label") or "").strip():
            issues.append(Issue("warn", "missing_axis_label",
                                f"[{spec.get('kind')}] '{spec.get('title')}' has no y_label.", cid))
    return issues


def validate(df: pd.DataFrame, report: Optional[dict]) -> list[Issue]:
    """Run every deterministic check. If generation failed, return only that."""
    err = check_generation_error(df, report)
    if err:
        return err
    issues: list[Issue] = []
    issues += check_degenerate_charts(df, report)
    issues += check_chart_data_consistency(df, report)
    issues += check_kpi_sanity(df, report)
    issues += check_narrative(df, report)
    issues += check_chart_count_floor(df, report)
    issues += check_axis_labels(df, report)
    return issues
