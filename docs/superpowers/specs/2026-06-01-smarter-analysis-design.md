# ChartSage Smarter / Steerable Analysis — Design Spec

**Status:** Approved (design), pending spec review
**Date:** 2026-06-01
**Type:** Backend + frontend feature (the analysis engine)

**Goal:** Give users more control over the analysis and let the AI do deeper work — via a custom-prompt steer, on-demand single charts, and an opt-in iterative-deepening "Deep analysis" mode.

**Context:** Today generation is a 2-pass pipeline in `ReportGenerator` (`report_generator.py`): pass 1 = chart selection via parallel tool calls (`CHART_TOOLS` → `TOOL_EXECUTORS`, with a retry round + heuristic fallback, capped at `MAX_CHARTS=10`); pass 2 = `submit_narrative`. `generate_more` already does a focused "5 different angles" pass. Credits (`credits.py` backend + `lib/credits.ts` frontend): report **100**, generate-more **40**, signup grant **300**. Reports are gated 402 `OUT_OF_CREDITS`; `credits_spent {amount, balance, reason}` events fire on debit.

**Approved decisions:** harness = **iterative deepening**; Deep analysis = **opt-in, 250 credits**; request-a-chart = **20 credits**, two modes (pick-a-type → AI fills columns, or describe-it → AI picks tool+columns).

---

## Facet 1 — Custom prompt ("steer") — *free, part of normal generation*

- **UX:** an optional **"Anything specific to focus on?"** textarea on the upload page (`/app`), and available on Deep analysis + request-a-chart. ~280 char cap.
- **Use:** threaded into the **selection** system prompt and the **narrative** prompt as a clearly-delimited *user focus* block (e.g. `User's focus (guidance, not a rule): "<text>"`). It steers but never overrides the hard rules (identifier/unusable columns, the bar soft-cap, etc.). Treat as untrusted text — no rule-override, no tool-injection.
- **Persistence:** stored on `Report.metadata.custom_prompt` so `generate_more`, request-a-chart, and Deep analysis honor the same focus.
- **Backend:** `ReportGenerator.__init__` takes an optional `custom_prompt: str | None`; `generate_report` reads it from a form field and passes it through; the selection/narrative message builders prepend the focus block when present.

---

## Facet 2 — Request-a-chart ("direct control") — *20 credits*

On a report, an **"+ Add a chart"** control (in/near the Toolbar) with two tabs:
- **Pick a type:** choose a chart kind (the existing kinds: bar / line / scatter / pie / box / heatmap / treemap / grouped-bar / dual-axis) → a **focused Claude call** is forced to call that single chart's tool, choosing the best columns from the profile → executor → append.
- **Describe it:** a text box ("show me margin by region") → a focused Claude call picks the appropriate tool + params → executor → append.

- **Endpoint:** `POST /report/{id}/add-chart` body `{ mode: "type"|"describe", chart_type?: str, prompt?: str }`. Identity required. Flow: ensure-profile + pre-check `balance < ADD_CHART_COST` → 402 `OUT_OF_CREDITS`; run the focused selection (1 chart) using the existing `CHART_TOOLS`/`TOOL_EXECUTORS` (for `mode=type`, force `tool_choice` to that tool; for `describe`, let the model choose); on a valid chart, append to the report (`ChartWithCaption` + a sidebar layout entry, like `generate_more`), generate a caption, save, then **debit 20** (`spend_credits(..., reason="add_chart", ref=report_id)`) — debit AFTER the save succeeds, matching generate-more's order. If the model produces no usable chart, return 422 and **do not debit**.
- Returns the updated report (frontend appends/refreshes), and `refetch()` updates the balance.

---

## Facet 3 — Deep analysis ("smarter", iterative deepening) — *opt-in, 250 credits*

- **UX:** a **"Deep analysis (250 cr)"** toggle on the upload page (alongside the custom-prompt field), and a **"Deepen this report"** action on an existing report. The progress UI shows the deepening rounds.
- **Mechanics (the loop):** after a normal pass-1 selection, run an iterative-deepening loop in `ReportGenerator.deepen(...)`:
  1. Build a focused message: the profile + the custom prompt + **summaries of the charts so far** (kind/title/intent + a short data sample via the existing `_summarize_chart_data`) + the instruction *"Propose additional charts that reveal something NOT already shown — different columns, relationships, breakdowns, or segments. If the analysis is already complete, return no tool calls."*
  2. Call the selection model with `CHART_TOOLS`; execute the returned tool calls → append the new (non-duplicate) specs.
  3. Repeat up to **`MAX_DEEP_ROUNDS = 3`**, stopping early when a round adds **0** new charts (the AI's "done" signal) or the chart count hits a hard cap (**`MAX_DEEP_CHARTS = 20`**).
  4. Run the **narrative pass over the full enriched set**.
- Reuses the existing `_execute_tool_calls` + tools + executors + dedupe-by-angle logic (same machinery as `generate_more`, looped + grounded + with a stop condition).
- **Entry points:** (a) generation-time toggle → `generate_report` runs `deep` (debit **250** instead of 100, gated like a normal report); (b) `POST /report/{id}/deepen` on an existing report → runs the loop + re-narrative, debits **250**. Both: ensure-profile + 402 pre-check; debit after save; `credits_spent {reason: "deep_analysis"}`.
- **Cost/latency:** several extra Claude calls → slower + more tokens; that's why it's opt-in + priced at 250.

---

## Credits & config

- **New constants** (backend `credits.py` + frontend `lib/credits.ts`, env-overridable like the others): `DEEP_ANALYSIS_COST = 250`, `ADD_CHART_COST = 20`. Unchanged: `REPORT_COST=100`, `GENERATE_MORE_COST=40`, `SIGNUP_GRANT=300`.
- **Gating:** the new paid actions pre-check balance → 402 `OUT_OF_CREDITS` (→ the existing OutOfCreditsModal), and debit only after the work succeeds (the established order). `credits_spent` events gain reasons `deep_analysis` and `add_chart` (with `{amount, balance}`).
- The credits-page `LABELS` map gains `deep_analysis` → "Deep analysis", `add_chart` → "Added chart".

---

## Backend changes (`report_generator.py`, `main.py`)

- `ReportGenerator`: `__init__(..., custom_prompt=None)`; selection + narrative message builders inject the focus block; add `deepen(existing_specs)` (the loop) and a `deep` flag path in `build_report`; a focused `add_chart(mode, chart_type, prompt)` single-chart method.
- `main.py`: `generate_report` reads `custom_prompt` + `deep` form fields (deep → 250 debit); new `POST /report/{id}/add-chart` (20) and `POST /report/{id}/deepen` (250). All reuse the credit-gate + debit-after-success pattern.

## Frontend

- **Upload page (`/app`):** the custom-prompt textarea + the "Deep analysis (250 cr)" toggle; the generate button cost label reflects 100 vs 250.
- **Report Toolbar:** "+ Add a chart" (the two-tab modal) and "Deepen this report" (when not already deep), both wired to the endpoints with 402→OutOfCreditsModal + `useCredits().refetch`.
- **Credits page:** the new transaction labels.

---

## Phasing (plan order — each ships on its own)
1. **Custom prompt** (smallest): thread `custom_prompt` through generation + the upload field + persist.
2. **Request-a-chart**: `ADD_CHART_COST`, the `add-chart` endpoint + the focused single-chart path, the two-tab modal.
3. **Deep analysis** (biggest): `DEEP_ANALYSIS_COST`, the deepening loop, the `deep` generation flag + `deepen` endpoint, the toggle + progress UI.

## Scope & non-goals
**In scope:** the three facets above with credit gating + events; new constants; the deepening loop; the add-chart focused path.

**Non-goals:**
- **Conversational data-Q&A chat** (the alternative harness shape we did *not* pick) — out of scope.
- Native editable charts, new chart *types* (the chart vocabulary is fixed here), or export changes.
- Changing the free-first-report or existing prices (100/40/300 unchanged).
- Unbounded agent loops — Deep analysis is hard-capped (`MAX_DEEP_ROUNDS`, `MAX_DEEP_CHARTS`).
- Prompt-driven rule overrides — the custom prompt is guidance only.

## Verification
- **Backend (TDD):** custom prompt appears in the selection/narrative messages when set + is persisted to `Report.metadata`; `add_chart` (both modes) returns one valid chart appended + debits 20 only on success (422 + no debit on failure); the deepening loop adds charts across rounds, stops on a 0-new round, respects `MAX_DEEP_ROUNDS`/`MAX_DEEP_CHARTS`, re-narrates, and debits 250; 402 pre-checks fire when balance is short. Full `pytest` green.
- **Frontend:** `tsc` + `next build` clean; the custom-prompt field + Deep toggle + cost label; the Add-a-chart modal (both tabs); the Deepen action; 402→modal.
- **Live (post-deploy):** generate with a custom prompt (it steers); add a chart (both modes, 20 cr); run Deep analysis (more charts, 250 cr); out-of-credits paths.
