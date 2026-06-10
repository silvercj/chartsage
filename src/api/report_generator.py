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
from chart_executor import TOOL_EXECUTORS, execute_key_metrics, normalize_percentage_spec
from fallback import pick_fallback_charts, drop_duplicates


MAX_CHARTS = 10
MIN_CHARTS_FOR_NO_FALLBACK = 3
# When pass-1 selection finishes (clean, no execution errors) with fewer than this
# many charts, fire one extra "reach for more" round before falling back to
# heuristics. Catalog/text-heavy datasets (e.g. Netflix) used to collapse to ~3.
MIN_CHARTS_TARGET = 6

# Deep analysis (iterative deepening): hard caps that bound cost + latency. The
# loop also stops early when a round proposes 0 new charts (the AI's "done" signal).
MAX_DEEP_ROUNDS = 3
MAX_DEEP_CHARTS = 20

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
        custom_prompt: str | None = None,
    ):
        self.profile = profile
        self.df = df
        self.claude = claude
        self.model_selection = model_selection
        self.model_narrative = model_narrative
        # Optional free-text steer. Trimmed and capped; empty -> None so we never
        # inject an empty focus block. Goes into the USER message only (the system
        # prompt is cached) and is treated as guidance, not a rule override.
        self.custom_prompt = (custom_prompt or "").strip()[:280] or None
        self._key_metrics: list = []
        self._token_totals = {
            "input_tokens_total": 0,
            "output_tokens_total": 0,
            "cache_read_input_tokens_total": 0,
        }

    def _call_claude(self, **kwargs):
        response = self.claude.messages_create(**kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._token_totals["input_tokens_total"] += getattr(usage, "input_tokens", 0) or 0
            self._token_totals["output_tokens_total"] += getattr(usage, "output_tokens", 0) or 0
            self._token_totals["cache_read_input_tokens_total"] += getattr(usage, "cache_read_input_tokens", 0) or 0
        return response

    def _focus_block(self) -> str:
        """User-message suffix carrying the optional focus steer (empty when unset)."""
        if not self.custom_prompt:
            return ""
        return f"\n\nUser's focus (guidance — still follow all the rules above): {self.custom_prompt}"

    def generate_charts(self) -> list[ChartSpec]:
        """Pass #1: tool-use selection + 1 retry round + fallback."""
        specs, errors, response_content = self._call_selection_initial()

        if errors:
            specs2, _ = self._call_selection_retry(response_content, errors)
            specs.extend(specs2)

        # Clean under-selection: the model picked too few charts but hit no errors,
        # so neither the error-retry nor the fallback floor (3) would fire. Push for
        # more — but only when there's headroom; the fallback stays the last resort.
        if not errors and len(specs) < MIN_CHARTS_TARGET and len(specs) < MAX_CHARTS:
            specs.extend(self._call_selection_more(specs))

        if len(specs) < MIN_CHARTS_FOR_NO_FALLBACK:
            # Fallback specs are built deterministically (not via _execute_tool_calls), so
            # their percentage scale is normalized here. Drop any that re-make a chart the
            # model already selected — the fallback re-derives from the same dataframe and
            # would otherwise duplicate model charts (each line showed up twice in the cards
            # report when the model under-picked).
            fb = [
                normalize_percentage_spec(s)
                for s in pick_fallback_charts(
                    self.profile, self.df, max_charts=MAX_CHARTS - len(specs),
                )
            ]
            specs.extend(drop_duplicates(specs, fb))

        return specs[:MAX_CHARTS]

    def _call_selection_initial(self) -> tuple[list[ChartSpec], list[dict], list[Any]]:
        response = self._call_claude(
            model=self.model_selection,
            max_tokens=4096,
            system=SELECTION_SYSTEM,
            tools=CHART_TOOLS,
            messages=[{"role": "user", "content": self.profile.to_text() + self._focus_block()}],
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
            {"role": "user", "content": self.profile.to_text() + self._focus_block()},
            {"role": "assistant", "content": _serialize_content(prior_content)},
            {"role": "user", "content": tool_results},
        ]

        response = self._call_claude(
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
            if block.name == "key_metrics":
                # A numeric temporal-ordinal axis (Year stored as ints) is a time axis for
                # the KPI guard's purposes — max(year) is as meaningless as max(date).
                roles = {
                    c.name: ("date" if c.temporal_ordinal else c.role)
                    for c in self.profile.columns
                }
                res = execute_key_metrics(self.df, block.input, roles=roles)
                if isinstance(res, ToolError):
                    errors.append({"id": block.id, "reason": res.reason})
                else:
                    self._key_metrics = res   # stored, NOT counted as a chart
                continue
            executor = TOOL_EXECUTORS.get(block.name)
            if executor is None:
                errors.append({"id": block.id, "reason": f"unknown tool '{block.name}'"})
                continue
            try:
                result = executor(self.df, block.input)
            except Exception as exc:  # noqa: BLE001 - one bad spec must never 500 the whole report
                logging.warning(
                    "[GEN] tool '%s' raised %s: %s", block.name, type(exc).__name__, exc
                )
                errors.append({"id": block.id, "reason": f"{block.name} failed: {exc}"})
                continue
            if isinstance(result, ToolError):
                errors.append({"id": block.id, "reason": result.reason})
                logging.warning("[GEN] tool '%s' error: %s", block.name, result.reason)
            else:
                specs.append(normalize_percentage_spec(result))
                if len(specs) >= MAX_CHARTS:
                    break
        return specs, errors

    def generate_narrative(self, charts: list[ChartSpec]) -> ReportNarrative:
        """Pass #2: forced submit_narrative tool call."""
        user_message = self._format_charts_for_narrative(charts)
        response = self._call_claude(
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
        return "\n".join(lines) + self._focus_block()

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
            f"{self._focus_block()}"
        )

        response = self._call_claude(
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

    def add_chart(self, mode: str, chart_type: str | None, prompt: str | None) -> ChartWithCaption | None:
        """Focused single-chart selection for the request-a-chart action.

        mode='type'     forces the given tool (chart_type), model picks the columns.
        mode='describe' lets the model choose the tool that best answers the prompt.
        Honors the report's persisted custom_prompt via _focus_block(). Returns a
        ChartWithCaption, or None when no spec is produced (caller turns that into a
        422 with no debit).
        """
        from uuid import uuid4

        if mode == "type" and chart_type:
            instruction = f"Create one {chart_type} for this data — choose the most revealing columns."
            tool_choice = {"type": "tool", "name": chart_type}
        else:
            instruction = f"Create one chart that best answers: {prompt}"
            tool_choice = {"type": "any"}

        user = f"{self.profile.to_text()}{self._focus_block()}\n\n{instruction}"
        response = self._call_claude(
            model=self.model_selection,
            max_tokens=1024,
            system=SELECTION_SYSTEM,
            tools=CHART_TOOLS,
            tool_choice=tool_choice,
            messages=[{"role": "user", "content": user}],
            cache_static=True,
        )
        specs, _ = self._execute_tool_calls(response.content)
        if not specs:
            return None
        spec = specs[0]
        return ChartWithCaption(chart_id=uuid4().hex, spec=spec, caption=spec.intent)

    @staticmethod
    def _angle_key(spec: ChartSpec) -> tuple:
        """Dedupe key for a chart: its kind + the set of columns it draws from.
        Two charts with the same kind over the same columns are 'the same angle'."""
        return (spec.kind, tuple(sorted(spec.source_columns)))

    def _call_selection_more(self, existing_specs: list[ChartSpec]) -> list[ChartSpec]:
        """One extra selection round when pass-1 under-selected (no errors). Asks for
        ADDITIONAL, DIFFERENT charts toward MAX_CHARTS, explicitly green-lighting more
        bar/top-N charts. Returns the NEW specs, deduped against existing by angle key
        and capped to the remaining room. Modeled on a single deepen() round."""
        summary = "\n".join(f"- [{c.kind}] {c.title} — {c.intent}" for c in existing_specs)
        msg = (
            f"{self.profile.to_text()}\n\n"
            f"Charts already selected:\n{summary or '(none yet)'}\n\n"
            f"You selected only {len(existing_specs)} charts — aim for about {MAX_CHARTS}. "
            f"Propose ADDITIONAL, DIFFERENT charts that reveal something not already shown: "
            f"top-N breakdowns of high-cardinality categories (frequency bar or treemap), "
            f"trends over time from any date/year column, cross-tabs (grouped bars), and "
            f"distributions you haven't used. Additional bar/top-N charts are fine here — a "
            f"fuller report matters more than avoiding bars. Do not repeat charts already selected."
            f"{self._focus_block()}"
        )
        response = self._call_claude(
            model=self.model_selection,
            max_tokens=4096,
            system=SELECTION_SYSTEM,
            tools=CHART_TOOLS,
            messages=[{"role": "user", "content": msg}],
            cache_static=True,
        )
        specs, _ = self._execute_tool_calls(response.content)

        seen = {self._angle_key(s) for s in existing_specs}
        room = MAX_CHARTS - len(existing_specs)
        new_specs: list[ChartSpec] = []
        for s in specs:
            key = self._angle_key(s)
            if key in seen:
                continue
            seen.add(key)
            new_specs.append(s)
            if len(new_specs) >= room:
                break
        return new_specs

    def deepen(self, seed_specs: list[ChartSpec]) -> list[ChartSpec]:
        """Iterative-deepening loop: add follow-up charts the AI thinks reveal more,
        until a round adds nothing (its 'done' signal) or the caps hit. Returns the
        NEW specs (beyond seed).

        Hard-capped by MAX_DEEP_ROUNDS / MAX_DEEP_CHARTS. Uses auto tool_choice so
        the model can return zero tool calls to signal the analysis is complete.
        """
        have = list(seed_specs)
        added: list[ChartSpec] = []
        for _ in range(MAX_DEEP_ROUNDS):
            if len(have) >= MAX_DEEP_CHARTS:
                break
            summary = "\n".join(f"- [{c.kind}] {c.title} — {c.intent}" for c in have)
            msg = (
                f"{self.profile.to_text()}{self._focus_block()}\n\n"
                f"Charts already in the report:\n{summary or '(none yet)'}\n\n"
                f"Propose additional charts that reveal something NOT already shown — different "
                f"columns, relationships, breakdowns, or segments. If the analysis is already "
                f"complete, return no tool calls."
            )
            response = self._call_claude(
                model=self.model_selection,
                max_tokens=4096,
                system=SELECTION_SYSTEM,
                tools=CHART_TOOLS,
                messages=[{"role": "user", "content": msg}],
                cache_static=True,
            )
            specs, _ = self._execute_tool_calls(response.content)
            if not specs:
                break                       # AI's "done" signal
            room = MAX_DEEP_CHARTS - len(have)
            specs = specs[:room]
            have.extend(specs)
            added.extend(specs)
        return added

    def build_report(self, deep: bool = False) -> Report:
        from datetime import datetime
        from uuid import uuid4
        from schemas import ChartLayoutEntry

        charts = self.generate_charts()
        if deep:
            # Iterative deepening enriches the set beyond the initial MAX_CHARTS cap.
            charts = charts + self.deepen(charts)
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
            key_metrics=self._key_metrics,
            charts=charts_with_caption,
            layout=layout,
            metadata={
                "model_selection": self.model_selection,
                "model_narrative": self.model_narrative,
                "row_count": self.profile.row_count,
                "column_count": len(self.profile.columns),
                "custom_prompt": self.custom_prompt,
                "deep": deep,
                **self._token_totals,
            },
        )
