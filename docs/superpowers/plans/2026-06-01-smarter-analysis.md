# ChartSage Smarter / Steerable Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> **Subagents must NOT run `git checkout`, `git switch`, `git reset`, or `git stash`.** Stay on branch `smarter-analysis`; commit only the files each task lists.

**Goal:** A custom-prompt steer, an on-demand single-chart action (20 cr), and an opt-in iterative-deepening "Deep analysis" mode (250 cr).

**Architecture:** All three extend the existing `ReportGenerator` 2-pass pipeline. `custom_prompt` is injected into the **user** message of the selection + narrative calls (keeps the cached system prompt intact) and persisted on `Report.metadata`. Request-a-chart and Deep analysis are new endpoints that mirror `generate_more`'s gate→work→append→debit-after-success pattern; the deepening loop reuses the existing tool-call machinery, hard-capped. Credits gate as 402 `OUT_OF_CREDITS`.

**Tech Stack:** FastAPI, Anthropic tool-use, Next.js. No new deps.

**Spec:** `docs/superpowers/specs/2026-06-01-smarter-analysis-design.md`
**Branch:** `smarter-analysis` (carries the spec).

**Key templates (read these):** `generate_more` endpoint (`main.py:450-565`) — the gate/work/append/debit pattern; `ReportGenerator` (`report_generator.py`) — `generate_charts`, `generate_more`, `_execute_tool_calls`, `generate_narrative`, `_summarize_chart_data`, `_call_claude`; `credits.py` (`_int_env` constants); `CHART_TOOLS`/`TOOL_EXECUTORS`.

---

# PHASE 1 — Custom prompt

### Task 1: Thread `custom_prompt` through generation + persist + upload field

**Files:** Modify `src/api/report_generator.py`, `src/api/main.py`, `src/app/app/page.tsx`. Test: `tests/integration/test_custom_prompt.py`.

- [ ] **Step 1: Failing test** `tests/integration/test_custom_prompt.py` — use the existing FakeClaude pattern (see `tests/integration/test_pipeline_happy.py`) to capture the messages sent to the model and assert the focus text appears.
```python
from report_generator import ReportGenerator
from profile import profile_dataframe
from tests.helpers.fake_claude import FakeClaude, tool_use

def test_custom_prompt_in_selection_message(sales):
    fc = FakeClaude([{"tool_calls": [tool_use("frequency_bar_chart", {"column":"region","title":"T","intent":"i"})]},
                     {"tool_calls": [tool_use("submit_narrative", {"summary":"S.","captions":["c"],"data_quality":[]})]}])
    import types
    gen = ReportGenerator(profile=profile_dataframe(sales), df=sales, claude=types.SimpleNamespace(messages_create=fc),
                          model_selection="m", model_narrative="m", custom_prompt="focus on margins by region")
    gen.build_report()
    sent = " ".join(str(m) for call in fc.calls for m in call["messages"])  # FakeClaude records calls
    assert "focus on margins by region" in sent
```
(If `FakeClaude` doesn't already record `.calls`, capture the messages however the existing fakes expose them — check `tests/helpers/fake_claude.py`.) Run → FAIL.

- [ ] **Step 2: Implement in `ReportGenerator`**
  - `__init__(..., custom_prompt: str | None = None)` → `self.custom_prompt = (custom_prompt or "").strip()[:280] or None`.
  - Add a helper:
```python
def _focus_block(self) -> str:
    if not self.custom_prompt:
        return ""
    return f"\n\nUser's focus (guidance — still follow all the rules above): {self.custom_prompt}"
```
  - Append `self._focus_block()` to the **user content** in `_call_selection_initial`, `_call_selection_retry` (the `self.profile.to_text()` user message), `generate_more` (the `focused_message`), and to the narrative user message in `_format_charts_for_narrative`. (Do NOT modify the system prompts — they're cached.)
  - In `build_report`, add `"custom_prompt": self.custom_prompt` to the `metadata={...}` dict.

- [ ] **Step 3: Run test → PASS.**

- [ ] **Step 4: Wire the endpoint** — in `main.py` `generate_report`, read an optional form field `custom_prompt: str | None = Form(None)` and pass `custom_prompt=custom_prompt` into the `ReportGenerator(...)` constructor. (For `generate_more`, read `row["report_json"].get("metadata",{}).get("custom_prompt")` and pass it through so it honors the same focus.)

- [ ] **Step 5: Upload field** — in `src/app/app/page.tsx`, add an optional `<textarea>` ("Anything specific to focus on? (optional)", ~280 char cap, dark semantic styling) above the Generate button; on generate, `fd.append('custom_prompt', focus)` when non-empty. (Everything else in the upload flow unchanged.)

- [ ] **Step 6: Verify + commit**
```bash
./venv/bin/python -m pytest tests/integration/test_custom_prompt.py -q && npx tsc --noEmit
git add src/api/report_generator.py src/api/main.py "src/app/app/page.tsx" tests/integration/test_custom_prompt.py
git commit -m "feat(analysis): custom prompt steers selection + narrative, persisted to metadata"
```

---

# PHASE 2 — Request-a-chart (20 credits)

### Task 2: `ADD_CHART_COST` + `ReportGenerator.add_chart` + `/add-chart` endpoint (TDD)

**Files:** Modify `src/api/credits.py`, `src/app/lib/credits.ts`, `src/api/report_generator.py`, `src/api/main.py`. Test: `tests/integration/test_add_chart.py`.

- [ ] **Step 1: Constants** — `credits.py`: `ADD_CHART_COST = _int_env("ADD_CHART_COST", 20)`. `lib/credits.ts`: `export const ADD_CHART_COST = 20;`.

- [ ] **Step 2: Failing test** `tests/integration/test_add_chart.py` (FakeClaude returning one tool call; assert the returned chart + that an unmatched/empty response yields None):
```python
def test_add_chart_pick_type_returns_one(sales):
    fc = FakeClaude([{"tool_calls": [tool_use("scatter_chart", {"x_col":"units","y_col":"revenue","title":"U vs R","intent":"i"})]}])
    import types
    gen = ReportGenerator(profile=profile_dataframe(sales), df=sales, claude=types.SimpleNamespace(messages_create=fc), model_selection="m", model_narrative="m")
    cwc = gen.add_chart(mode="type", chart_type="scatter_chart", prompt=None)
    assert cwc is not None and cwc.spec.kind == "scatter"

def test_add_chart_no_chart_returns_none(sales):
    fc = FakeClaude([{"tool_calls": []}])
    import types
    gen = ReportGenerator(profile=profile_dataframe(sales), df=sales, claude=types.SimpleNamespace(messages_create=fc), model_selection="m", model_narrative="m")
    assert gen.add_chart(mode="describe", chart_type=None, prompt="show me x") is None
```
Run → FAIL.

- [ ] **Step 3: Implement `ReportGenerator.add_chart`**
```python
def add_chart(self, mode: str, chart_type: str | None, prompt: str | None):
    """Focused single-chart selection. mode='type' forces the given tool; mode='describe' lets the model choose. Returns a ChartWithCaption or None."""
    from uuid import uuid4
    from schemas import ChartWithCaption
    if mode == "type" and chart_type:
        instruction = f"Create one {chart_type} for this data — choose the most revealing columns."
        tool_choice = {"type": "tool", "name": chart_type}
    else:
        instruction = f"Create one chart that best answers: {prompt}"
        tool_choice = {"type": "any"}
    user = f"{self.profile.to_text()}{self._focus_block()}\n\n{instruction}"
    response = self._call_claude(
        model=self.model_selection, max_tokens=1024, system=SELECTION_SYSTEM,
        tools=CHART_TOOLS, tool_choice=tool_choice,
        messages=[{"role": "user", "content": user}], cache_static=True,
    )
    specs, _ = self._execute_tool_calls(response.content)
    if not specs:
        return None
    spec = specs[0]
    return ChartWithCaption(chart_id=uuid4().hex, spec=spec, caption=spec.intent)
```
(`chart_type` is a tool name from `CHART_TOOLS`, e.g. `scatter_chart`, `treemap_chart`. The endpoint validates it's a real tool.)

- [ ] **Step 4: Run test → PASS.**

- [ ] **Step 5: `/add-chart` endpoint** in `main.py` — mirror `generate_more` (auth required → 402 UPGRADE_REQUIRED for anon; `_ensure_profile_tracked`; `balance < ADD_CHART_COST` → 402 `OUT_OF_CREDITS`; load report + CSV via `download_by_key`; build `ReportGenerator` with the report's persisted `custom_prompt`; call `gen.add_chart(...)`; if None → 422 (no debit); else append to `report_dict["charts"]` + a sidebar `layout` entry (order = sidebar_max+1, like generate_more); `update_report_json`; `spend_credits(user, ADD_CHART_COST, "add_chart", session_id)` → `credits_spent {amount, balance, reason:"add_chart"}`; return the updated report dict). Body via Pydantic: `{ mode: Literal["type","describe"], chart_type: str|None, prompt: str|None }`; validate `chart_type` is in the tool set for `mode="type"`.

- [ ] **Step 6: Verify + commit**
```bash
./venv/bin/python -m pytest tests/integration/test_add_chart.py -q
git add src/api/credits.py "src/app/lib/credits.ts" src/api/report_generator.py src/api/main.py tests/integration/test_add_chart.py
git commit -m "feat(analysis): request-a-chart — add_chart (type/describe) + /add-chart endpoint (20cr)"
```

---

### Task 3: Add-a-chart modal (frontend)

**Files:** Modify `src/app/report/[id]/Toolbar.tsx` (+ a small `AddChartModal` component, in the same dir).

- [ ] **Step 1: Build `AddChartModal.tsx`** — a dark modal (match `OutOfCreditsModal` styling) with two tabs:
  - **Pick a type:** a grid/list of the chart tools with friendly labels (Frequency bar, Aggregation bar, Histogram, Scatter, Trend line, Pie, Box plot, Heatmap, Treemap, Grouped bar, Dual-axis) → `POST /report/${id}/add-chart {mode:'type', chart_type:'<tool_name>'}`.
  - **Describe it:** a textarea ("e.g. revenue by channel over time") → `{mode:'describe', prompt}`.
  - On 402 `OUT_OF_CREDITS` → show OutOfCreditsModal; on 200 → `onReportUpdated(updated)` + `refetch()` + close; on 422 → inline "Couldn't build that chart — try another type or description." Label the action **"Add chart · 20"** using `ADD_CHART_COST`.
- [ ] **Step 2: Wire into Toolbar** — add a **"+ Add a chart"** button opening the modal; reuse `useCredits().refetch` + the existing `onReportUpdated`.
- [ ] **Step 3: Verify + commit**
```bash
npx tsc --noEmit
git add "src/app/report/[id]/Toolbar.tsx" "src/app/report/[id]/AddChartModal.tsx"
git commit -m "feat(analysis): Add-a-chart modal (pick type / describe)"
```

---

# PHASE 3 — Deep analysis (250 credits, iterative deepening)

### Task 4: `DEEP_ANALYSIS_COST` + `deepen` loop + deep generation + `/deepen` endpoint (TDD)

**Files:** Modify `src/api/credits.py`, `src/app/lib/credits.ts`, `src/api/report_generator.py`, `src/api/main.py`. Test: `tests/integration/test_deep_analysis.py`.

- [ ] **Step 1: Constants** — `credits.py`: `DEEP_ANALYSIS_COST = _int_env("DEEP_ANALYSIS_COST", 250)`; module consts `MAX_DEEP_ROUNDS = 3`, `MAX_DEEP_CHARTS = 20` (in `report_generator.py`). `lib/credits.ts`: `export const DEEP_ANALYSIS_COST = 250;`.

- [ ] **Step 2: Failing test** `tests/integration/test_deep_analysis.py` — FakeClaude scripted so round 1 adds a chart, round 2 adds none (stop):
```python
def test_deepen_adds_then_stops(sales):
    fc = FakeClaude([
        {"tool_calls": [tool_use("histogram_chart", {"column":"revenue","title":"Rev dist","intent":"i"})]},  # round 1: 1 new
        {"tool_calls": []},                                                                                     # round 2: none -> stop
    ])
    import types
    gen = ReportGenerator(profile=profile_dataframe(sales), df=sales, claude=types.SimpleNamespace(messages_create=fc), model_selection="m", model_narrative="m")
    seed = []  # pretend no charts yet for the test
    extra = gen.deepen(seed)
    assert len(extra) == 1            # added one, then stopped on the empty round
```
Run → FAIL.

- [ ] **Step 3: Implement `ReportGenerator.deepen`**
```python
def deepen(self, seed_specs: list[ChartSpec]) -> list[ChartSpec]:
    """Iterative-deepening loop: add follow-up charts the AI thinks reveal more, until a round
    adds nothing (its 'done' signal) or the caps hit. Returns the NEW specs (beyond seed)."""
    have = list(seed_specs)
    added: list[ChartSpec] = []
    for _ in range(MAX_DEEP_ROUNDS):
        if len(have) >= MAX_DEEP_CHARTS:
            break
        summary = "\n".join(f"- [{c.kind}] {c.title} — {c.intent}" for c in have)
        msg = (f"{self.profile.to_text()}{self._focus_block()}\n\n"
               f"Charts already in the report:\n{summary or '(none yet)'}\n\n"
               f"Propose additional charts that reveal something NOT already shown — different "
               f"columns, relationships, breakdowns, or segments. If the analysis is already "
               f"complete, return no tool calls.")
        response = self._call_claude(
            model=self.model_selection, max_tokens=4096, system=SELECTION_SYSTEM,
            tools=CHART_TOOLS, messages=[{"role": "user", "content": msg}], cache_static=True)
        specs, _ = self._execute_tool_calls(response.content)
        if not specs:
            break                       # AI's "done" signal
        room = MAX_DEEP_CHARTS - len(have)
        specs = specs[:room]
        have.extend(specs); added.extend(specs)
    return added
```
  - In `build_report`, accept `deep: bool = False`; after `charts = self.generate_charts()`, if `deep`: `charts = charts + self.deepen(charts)`. Then narrative over the full set. (No re-cap to 10 when deep.)

- [ ] **Step 4: Run test → PASS.**

- [ ] **Step 5: Endpoints** in `main.py`:
  - **Deep generation:** `generate_report` reads `deep: bool = Form(False)`. When `deep`, the action requires auth (402 `UPGRADE_REQUIRED` for anon — deep is a paid feature) and gates/debits at `DEEP_ANALYSIS_COST` with reason `deep_analysis` (else the existing `REPORT_COST`/`report`). Pass `deep=True` into `gen.build_report(deep=True)`.
  - **Deepen existing:** `POST /report/{id}/deepen` — mirror `generate_more`: auth required; `balance < DEEP_ANALYSIS_COST` → 402; load report + CSV; build `ReportGenerator` (with the persisted `custom_prompt`); `extra = gen.deepen(existing_report.charts)`; if empty → return the report unchanged (no debit); else append the new charts (sidebar layout), **re-narrate** over the full set (`gen.generate_narrative(all_specs)` → update summary/captions), `update_report_json`, `spend_credits(..., DEEP_ANALYSIS_COST, "deep_analysis", session_id)` → `credits_spent {reason:"deep_analysis"}`.

- [ ] **Step 6: Verify + commit**
```bash
./venv/bin/python -m pytest tests/integration/test_deep_analysis.py -q
git add src/api/credits.py "src/app/lib/credits.ts" src/api/report_generator.py src/api/main.py tests/integration/test_deep_analysis.py
git commit -m "feat(analysis): deep analysis — deepen loop + deep generation + /deepen (250cr)"
```

---

### Task 5: Deep analysis frontend (toggle + cost label + Deepen button)

**Files:** Modify `src/app/app/page.tsx`, `src/app/report/[id]/Toolbar.tsx`.

- [ ] **Step 1: Upload toggle** — in `page.tsx`, add a **"Deep analysis"** toggle (checkbox/switch) near the Generate button, labelled with the cost (`Deep analysis · ${DEEP_ANALYSIS_COST}`). When on, the generate button label shows `Generate report · ${DEEP_ANALYSIS_COST}` (else `· ${REPORT_COST}`), and `fd.append('deep', 'true')`. Keep the existing 402→OutOfCreditsModal handling (deep for an anon returns 402 `UPGRADE_REQUIRED` → the upsell, or `OUT_OF_CREDITS` if signed-in & short). Progress copy can note "Deepening…" when deep.
- [ ] **Step 2: Deepen button** — in `Toolbar.tsx`, add **"Deepen this report · 250"** (hidden if the report is already deep — check a `metadata.deep` flag set by deep generation, optional) → `POST /report/${id}/deepen` → on 200 `onReportUpdated` + `refetch`; 402 → OutOfCreditsModal; show a spinner/"Deepening…" while it runs (it's slow). Use `DEEP_ANALYSIS_COST`.
- [ ] **Step 3: Verify + commit**
```bash
npx tsc --noEmit
git add "src/app/app/page.tsx" "src/app/report/[id]/Toolbar.tsx"
git commit -m "feat(analysis): deep-analysis toggle (upload) + Deepen button (report)"
```

---

### Task 6: Build + verify + finish

- [ ] **Step 1:** `./venv/bin/python -m pytest -q` (green incl. the 3 new test files) + `rm -rf .next && npm run build` (exit 0).
- [ ] **Step 2: Live QA** (post-deploy needs the backend): generate with a custom prompt (the report reflects the focus); add a chart both ways (20 cr debited, appears); run Deep analysis from the toggle (more charts, 250 cr) and "Deepen this report" on an existing one; verify out-of-credits paths hit the modal and failed add-chart (422) doesn't debit.
- [ ] **Step 3: Finish** — use **superpowers:finishing-a-development-branch** to merge `smarter-analysis` → `main`. Backend changes (new endpoints, generator methods, constants) need a **Cloud Run deploy**; frontend auto-deploys via Vercel. Production deploy requires explicit user authorization.

---

## Self-Review

**Spec coverage:** custom prompt (steer + persist + free) → Task 1; request-a-chart (two modes, 20cr, debit-on-success, 422-no-debit) → Tasks 2–3; deep analysis (iterative deepening, caps, 250cr, toggle + deepen) → Tasks 4–5; new constants + gating + `credits_spent` reasons → Tasks 2/4; phasing custom→request→deep → Phases 1/2/3; verification → Task 6. Non-goals (no conversational Q&A, no new chart types, prices 100/40/300 unchanged, capped loop) respected. No gaps.

**Placeholder scan:** Complete code for the contract-critical pieces — `_focus_block`, `add_chart`, `deepen` (with caps + stop), the TDD tests pinning each, and the endpoints specified by mirroring the shown `generate_more` (gate → ensure-profile → 402 → work → append → `spend_credits` after `update_report_json` → `credits_spent`). Frontend modal/toggle are recipes against existing components (`OutOfCreditsModal`, the Toolbar download/refetch pattern). No TBD/"handle edge cases".

**Type/name consistency:** `custom_prompt` set in `ReportGenerator.__init__`, used by `_focus_block` (selection/narrative/generate_more/add_chart/deepen) and persisted in `build_report` metadata, read back by the endpoints. `add_chart(mode, chart_type, prompt) -> ChartWithCaption|None` and `deepen(seed_specs) -> list[ChartSpec]` match their call sites. Constants `ADD_CHART_COST=20`/`DEEP_ANALYSIS_COST=250` mirrored backend (`credits.py`) ↔ frontend (`lib/credits.ts`); `spend_credits(..., "add_chart"|"deep_analysis", report_id)` + `credits_spent` reasons match the credits-page LABELS additions. Endpoints reuse `_ensure_profile_tracked`, `download_by_key`, `update_report_json`, `spend_credits` — all existing.

**Risk note:** Deep generation/deepen require auth (paid feature; anon free report stays normal-only). The deepen loop is hard-capped (`MAX_DEEP_ROUNDS=3`, `MAX_DEEP_CHARTS=20`) and stops on a 0-new round, bounding cost/latency. `custom_prompt` goes into the user message (not the cached system prompt) so prompt caching is preserved, and is treated as guidance (no rule override).
