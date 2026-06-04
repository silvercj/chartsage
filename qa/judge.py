"""LLM judge: ask Haiku whether each chart makes sense for the data and whether
the narrative matches the charts. Uses the REAL Anthropic client (key from .env)
and a single forced tool call, mirroring report_generator.generate_narrative.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_API_DIR = Path(__file__).resolve().parent.parent / "src" / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

# override=True: the harness shell injects an empty ANTHROPIC_API_KEY; without
# override the empty value would shadow the real key in .env.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
load_dotenv(override=True)

# Drop the harness-injected Anthropic SDK vars that would hijack the client (proxy
# ANTHROPIC_BASE_URL, blank ANTHROPIC_AUTH_TOKEN -> "Bearer " with no token).
for _v in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "ANTHROPIC_CUSTOM_HEADERS"):
    os.environ.pop(_v, None)

from claude_client import ClaudeClient            # noqa: E402
from llm_config import MODEL_NARRATIVE            # noqa: E402  (cheap Haiku alias)


JUDGE_SYSTEM = (
    "You are a data-visualization QA reviewer. You are given a dataset PROFILE "
    "and a generated REPORT (chart specs + narrative). For EACH chart, decide if "
    "it makes sense for this data: is it degenerate (a line with one point, an "
    "all-zero/identical series), misleading (an identifier charted as a measure, a "
    "year treated as a quantity), or redundant? Then decide whether the narrative "
    "summary is supported by the charts (no claims a chart doesn't back). Be strict "
    "but fair; only mark makes_sense=false when there's a real problem. Always call "
    "submit_judgement exactly once."
)

JUDGE_TOOL: dict = {
    "name": "submit_judgement",
    "description": "Submit the QA judgement for this report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "charts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chart_id": {"type": "string"},
                        "makes_sense": {"type": "boolean"},
                        "issue": {"type": ["string", "null"],
                                  "description": "Short problem description, or null."},
                        "severity": {"type": "string",
                                     "enum": ["none", "warn", "fail"]},
                    },
                    "required": ["chart_id", "makes_sense", "severity"],
                },
            },
            "narrative_matches": {"type": "boolean"},
        },
        "required": ["charts", "narrative_matches"],
    },
}


@dataclass
class ChartVerdict:
    chart_id: str
    makes_sense: bool
    severity: str            # 'none' | 'warn' | 'fail'
    issue: Optional[str] = None


@dataclass
class JudgeVerdict:
    charts: list[ChartVerdict] = field(default_factory=list)
    narrative_matches: bool = True

    @property
    def any_chart_fails(self) -> bool:
        return any((not c.makes_sense) or c.severity == "fail" for c in self.charts)


def _build_claude() -> ClaudeClient:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (expected in .env)")
    return ClaudeClient(api_key=api_key)


def _compact_report(report: dict) -> str:
    """A token-lean view of the report for the judge: one line per chart + summary."""
    lines = [f"SUMMARY: {report.get('summary', '')}", "", "CHARTS:"]
    for chart in (report.get("charts") or []):
        spec = chart.get("spec") or {}
        x = spec.get("x") or []
        y = spec.get("y") or []
        sample = ""
        if x and y:
            sample = f" x[:5]={list(x)[:5]} y[:5]={list(y)[:5]}"
        elif spec.get("series"):
            sample = f" series={len(spec['series'])} e.g. {str(spec['series'][0])[:120]}"
        elif spec.get("nodes"):
            sample = f" nodes={len(spec['nodes'])}"
        lines.append(
            f"- id={chart.get('chart_id')} [{spec.get('kind')}] {spec.get('title')} "
            f"| x_label={spec.get('x_label')} y_label={spec.get('y_label')} "
            f"y_display={spec.get('y_display_type')}{sample}")
    return "\n".join(lines)


def judge_report(profile_text: str, report: dict) -> JudgeVerdict:
    claude = _build_claude()
    user = (f"DATASET PROFILE:\n{profile_text}\n\n"
            f"GENERATED REPORT:\n{_compact_report(report)}\n\n"
            f"Judge every chart and the narrative. Call submit_judgement once.")
    response = claude.messages_create(
        model=MODEL_NARRATIVE,
        max_tokens=1500,
        system=JUDGE_SYSTEM,
        tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "submit_judgement"},
        messages=[{"role": "user", "content": user}],
        cache_static=True,
    )
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_judgement":
            data = block.input
            charts = [
                ChartVerdict(
                    chart_id=str(c.get("chart_id", "")),
                    makes_sense=bool(c.get("makes_sense", True)),
                    severity=str(c.get("severity", "none")),
                    issue=c.get("issue"),
                )
                for c in (data.get("charts") or [])
            ]
            return JudgeVerdict(
                charts=charts,
                narrative_matches=bool(data.get("narrative_matches", True)),
            )
    # No tool call -> treat as inconclusive-pass (deterministic gate still applies).
    return JudgeVerdict(charts=[], narrative_matches=True)
