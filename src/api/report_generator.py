"""Two-pass report generation orchestrator.

Pass #1: chart selection via parallel tool calls. Tool errors get one retry round
         with structured tool_result messages, then fall back to heuristics if
         we still have fewer than 3 charts.
Pass #2: narrative generation via single forced submit_narrative call.
"""
import logging
import os
from pathlib import Path
from typing import Any
import pandas as pd
from schemas import ChartSpec, Report, ChartWithCaption, ReportNarrative, ToolError, DataProfile
from chart_tools import CHART_TOOLS, NARRATIVE_TOOL
from chart_executor import TOOL_EXECUTORS
from fallback import pick_fallback_charts


MAX_CHARTS = 10
MIN_CHARTS_FOR_NO_FALLBACK = 3

_PROMPT_DIR = Path(__file__).parent / "prompts"
SELECTION_SYSTEM = (_PROMPT_DIR / "selection_system.txt").read_text()
NARRATIVE_SYSTEM = (_PROMPT_DIR / "narrative_system.txt").read_text()


def _serialize_content(blocks: list[Any]) -> list[dict]:
    """Convert response.content blocks back into request-shape dicts."""
    out: list[dict] = []
    for b in blocks:
        if getattr(b, "type", None) == "tool_use":
            out.append({
                "type": "tool_use",
                "id": b.id,
                "name": b.name,
                "input": b.input,
            })
        elif getattr(b, "type", None) == "text":
            out.append({"type": "text", "text": b.text})
    return out


class ReportGenerator:
    def __init__(
        self,
        profile: DataProfile,
        df: pd.DataFrame,
        claude: Any,
        model_selection: str,
        model_narrative: str,
    ):
        self.profile = profile
        self.df = df
        self.claude = claude
        self.model_selection = model_selection
        self.model_narrative = model_narrative

    def generate_charts(self) -> list[ChartSpec]:
        """Pass #1: tool-use selection + 1 retry round + fallback."""
        specs, errors, response_content = self._call_selection_initial()

        if errors:
            specs2, _ = self._call_selection_retry(response_content, errors)
            specs.extend(specs2)

        if len(specs) < MIN_CHARTS_FOR_NO_FALLBACK:
            specs.extend(pick_fallback_charts(
                self.profile, self.df, max_charts=MAX_CHARTS - len(specs),
            ))

        return specs[:MAX_CHARTS]

    def _call_selection_initial(self) -> tuple[list[ChartSpec], list[dict], list[Any]]:
        response = self.claude.messages_create(
            model=self.model_selection,
            max_tokens=4096,
            system=SELECTION_SYSTEM,
            tools=CHART_TOOLS,
            messages=[{"role": "user", "content": self.profile.to_text()}],
            cache_static=True,
        )
        specs, errors = self._execute_tool_calls(response.content)
        return specs, errors, response.content

    def _call_selection_retry(
        self, prior_content: list[Any], errors: list[dict],
    ) -> tuple[list[ChartSpec], list[dict]]:
        # Anthropic requires a tool_result for EVERY tool_use in the prior assistant turn,
        # not just the failed ones. Build the full set: errors get their reason; successes
        # get a brief OK so the message is well-formed.
        error_by_id = {e["id"]: e["reason"] for e in errors}
        tool_results = []
        for block in prior_content:
            if getattr(block, "type", None) != "tool_use":
                continue
            if block.id in error_by_id:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": error_by_id[block.id],
                    "is_error": True,
                })
            else:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Chart accepted.",
                })

        messages = [
            {"role": "user", "content": self.profile.to_text()},
            {"role": "assistant", "content": _serialize_content(prior_content)},
            {"role": "user", "content": tool_results},
        ]

        response = self.claude.messages_create(
            model=self.model_selection,
            max_tokens=4096,
            system=SELECTION_SYSTEM,
            tools=CHART_TOOLS,
            messages=messages,
            cache_static=True,
        )
        return self._execute_tool_calls(response.content)

    def _execute_tool_calls(self, content_blocks: list[Any]) -> tuple[list[ChartSpec], list[dict]]:
        specs: list[ChartSpec] = []
        errors: list[dict] = []
        for block in content_blocks:
            if getattr(block, "type", None) != "tool_use":
                continue
            executor = TOOL_EXECUTORS.get(block.name)
            if executor is None:
                errors.append({"id": block.id, "reason": f"unknown tool '{block.name}'"})
                continue
            result = executor(self.df, block.input)
            if isinstance(result, ToolError):
                errors.append({"id": block.id, "reason": result.reason})
                logging.warning("[GEN] tool '%s' error: %s", block.name, result.reason)
            else:
                specs.append(result)
                if len(specs) >= MAX_CHARTS:
                    break
        return specs, errors

    def generate_narrative(self, charts: list[ChartSpec]) -> ReportNarrative:
        """Pass #2: forced submit_narrative tool call."""
        user_message = self._format_charts_for_narrative(charts)
        response = self.claude.messages_create(
            model=self.model_narrative,
            max_tokens=2048,
            system=NARRATIVE_SYSTEM,
            tools=[NARRATIVE_TOOL],
            tool_choice={"type": "tool", "name": "submit_narrative"},
            messages=[{"role": "user", "content": user_message}],
            cache_static=True,
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "submit_narrative":
                data = block.input
                return ReportNarrative(
                    summary=data.get("summary", ""),
                    captions=list(data.get("captions", [])),
                    data_quality=list(data.get("data_quality", [])),
                )
        return self._narrative_template_fallback(charts)

    def _format_charts_for_narrative(self, charts: list[ChartSpec]) -> str:
        lines = ["Profile:", self.profile.to_text(), "", "Charts to caption (in order):"]
        for i, c in enumerate(charts, 1):
            data_sample = self._summarize_chart_data(c)
            lines.append(f"{i}. [{c.kind}] {c.title}")
            lines.append(f"   Intent: {c.intent}")
            lines.append(f"   Data: {data_sample}")
        return "\n".join(lines)

    @staticmethod
    def _summarize_chart_data(spec: ChartSpec) -> str:
        if spec.series:
            return f"{len(spec.series)} series, e.g. {spec.series[0]}"[:200]
        if spec.x and spec.y:
            n = len(spec.x)
            sample_x = spec.x[:5]
            sample_y = spec.y[:5]
            return f"{n} points; sample x={sample_x} y={sample_y}"[:200]
        return f"{spec.data_point_count} data points"

    def _narrative_template_fallback(self, charts: list[ChartSpec]) -> ReportNarrative:
        return ReportNarrative(
            summary=f"Automated analysis of your data. The report contains {len(charts)} charts highlighting "
                    f"key patterns across the {self.profile.row_count} rows.",
            captions=[c.intent for c in charts],
            data_quality=list(self.profile.anomalies),
        )

    def generate_more(self, existing: list[ChartWithCaption]) -> tuple[list[ChartWithCaption], list["ChartLayoutEntry"]]:
        """Run a focused pass-1 to produce 5 additional charts different from `existing`.

        Returns (new charts with fresh chart_ids, new layout entries to append).
        Existing charts and layout are not modified.
        """
        from uuid import uuid4
        from schemas import ChartLayoutEntry

        # Build a focused user message that warns Claude off the existing chart angles
        existing_summary = "\n".join(
            f"- [{c.spec.kind}] {c.spec.title} — {c.spec.intent}"
            for c in existing
        )
        focused_message = (
            f"{self.profile.to_text()}\n\n"
            f"You have already produced these charts:\n{existing_summary}\n\n"
            f"Pick 5 different angles that are NOT repeats of the above. "
            f"Vary kinds; aim for chart types or column combinations not yet covered."
        )

        response = self.claude.messages_create(
            model=self.model_selection,
            max_tokens=4096,
            system=SELECTION_SYSTEM,
            tools=CHART_TOOLS,
            messages=[{"role": "user", "content": focused_message}],
            cache_static=True,
        )
        specs, errors = self._execute_tool_calls(response.content)

        if errors:
            specs2, _ = self._call_selection_retry(response.content, errors)
            specs.extend(specs2)

        # Cap at 5 new charts
        specs = specs[:5]

        # Narrative captions for the new charts only
        narrative = self.generate_narrative(specs) if specs else None
        captions = narrative.captions if narrative else [s.intent for s in specs]
        if len(captions) < len(specs):
            captions = captions + [s.intent for s in specs[len(captions):]]

        new_charts = [
            ChartWithCaption(chart_id=uuid4().hex, spec=spec, caption=cap)
            for spec, cap in zip(specs, captions)
        ]

        # New layout entries always start in sidebar
        return new_charts, [
            ChartLayoutEntry(chart_id=c.chart_id, position="sidebar", order=0)  # order set by caller
            for c in new_charts
        ]

    def build_report(self) -> Report:
        from datetime import datetime
        from uuid import uuid4
        from schemas import ChartLayoutEntry

        charts = self.generate_charts()
        narrative = self.generate_narrative(charts)

        captions = narrative.captions
        if len(captions) < len(charts):
            captions = captions + [c.intent for c in charts[len(captions):]]

        # Assign stable chart_ids
        charts_with_caption = [
            ChartWithCaption(chart_id=uuid4().hex, spec=spec, caption=cap)
            for spec, cap in zip(charts, captions)
        ]

        # Default layout: first 5 -> main, next 5 -> sidebar
        layout: list[ChartLayoutEntry] = []
        for i, cwc in enumerate(charts_with_caption[:5]):
            layout.append(ChartLayoutEntry(chart_id=cwc.chart_id, position="main", order=i))
        for i, cwc in enumerate(charts_with_caption[5:]):
            layout.append(ChartLayoutEntry(chart_id=cwc.chart_id, position="sidebar", order=i))

        return Report(
            generated_at=datetime.utcnow().isoformat(),
            summary=narrative.summary or self._narrative_template_fallback(charts).summary,
            data_quality=narrative.data_quality,
            charts=charts_with_caption,
            layout=layout,
            metadata={
                "model_selection": self.model_selection,
                "model_narrative": self.model_narrative,
                "row_count": self.profile.row_count,
                "column_count": len(self.profile.columns),
            },
        )
