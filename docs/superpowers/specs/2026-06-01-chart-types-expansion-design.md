# ChartSage Chart-Types Expansion — Design Spec

**Status:** Approved (design), pending spec review
**Date:** 2026-06-01
**Type:** Backend + frontend feature (the AI's chart vocabulary)

**Goal:** Expand the report beyond the current 7 chart kinds so generated reports are richer and more useful — adding a headline KPI band, bar/trend variants, and part-to-whole charts. (A separate "chart-diversity tune" already rebalanced selection away from bars; this adds *new* options to pick from.)

**Approved scope (this project):** KPI stat tiles · stacked & grouped bar · dual-axis combo · area / stacked-area · treemap (+ optional sunburst). **Geo/maps deferred** to its own brainstorm.

**Context:** Charts flow as: Claude calls chart tools (schemas in `chart_tools.py`) → executors compute `ChartSpec`s from the dataframe (`chart_executor.py`) → `report_generator.py` assembles the `Report` → frontend renders each `spec.kind` via `ChartContent.tsx` + a per-kind component, themed by `charts/chartTheme.ts`. Current kinds: `bar, histogram, scatter, line, pie, box, heatmap`. The selection prompt lives in `prompts/selection_system.txt`.

---

## 1. KPI stat tiles — the headline band

The biggest "feels like a real report" upgrade. A row of headline numbers above the charts.

- **New tool `key_metrics`** (called at most once): the AI proposes 3–5 metrics, each `{ label, column, agg, format }` where `agg ∈ {sum, mean, median, min, max, count, nunique}` and `format ∈ {number, currency, percent}`. The AI picks *what* to measure and how to label it.
- **The backend computes the value** from the dataframe (`execute_key_metrics`) — the AI never supplies the number, so tiles can't be hallucinated. Invalid metrics (bad column/agg) are dropped.
- **Schema:** add `KeyMetric { label: str, value: float, format: str }` and `Report.key_metrics: list[KeyMetric] = []`. The `key_metrics` tool result populates `Report.key_metrics` (it is **not** a chart and is **not** counted toward the 10 charts).
- **Frontend:** a new `KpiTiles` component renders the band (big `font-mono` numerals + `text-ink-3` labels), formatted via the existing `lib/format.ts` `getFormatter`. Shown at the top of the report (`report/[id]/page.tsx`) and in the **PDF print route**.
- **v1 keeps tiles simple** — a computed value + label. **Deltas ("▲ 18% vs Q2") are a deliberate fast-follow**, not v1 (they need period-over-period logic + a time column); noted as future.

## 2. Bar & trend variants

- **Grouped / stacked bar** — new tool `grouped_bar_chart(category_col, breakdown_col, value_col, agg, mode)` with `mode ∈ {grouped, stacked}`. New `ChartKind = "grouped_bar"`. Executor builds multi-series data (one series per `breakdown_col` value, sharing the `category_col` x-axis). Frontend `GroupedBarChart` renders ECharts multiple bar series, `series.stack` set when `mode='stacked'`. Guard: cap breakdown cardinality (≈6) and category cardinality (reuse existing `MAX_CATEGORIES`).
- **Dual-axis combo** — new tool `dual_axis_chart(x_col, bar_value_col, line_value_col, bar_agg, line_agg)`. New `ChartKind = "dual_axis"`. Two metrics on two y-axes (bar + line). Executor produces two series tagged with axis/type; spec carries `y_label` + `y_label_secondary`. Frontend `DualAxisChart` renders two `yAxis` + a bar series and a line series.
- **Area / stacked-area** — **no new kind**; extend `line_chart`'s tool schema + executor with an optional `area: bool`. `LineChart.tsx` reads `spec.area` and adds the teal `areaStyle` gradient (already available via `chartTheme.tealAreaGradient()`); multi-series + area → stacked area (`series.stack`).

## 3. Part-to-whole

- **Treemap** — new tool `treemap_chart(category_col, value_col, agg, subcategory_col?)`. New `ChartKind = "treemap"`. The go-to when a pie would overflow past ~8 slices, or for a 2-level hierarchy. Executor builds hierarchical nodes `[{ name, value, children? }]`. Frontend `Treemap` renders ECharts `treemap` with the editorial palette + `font-mono` labels.
- **Sunburst (optional stretch)** — same hierarchical node data, radial. New `ChartKind = "sunburst"` + `Sunburst` component reusing the treemap executor's node shape. **Ship only if cheap once treemap exists; drop without guilt otherwise.**

## 4. Schema additions (`schemas.py`)

- `ChartKind` gains `"grouped_bar"`, `"dual_axis"`, `"treemap"` (+ `"sunburst"` if built).
- `ChartSpec` gains a few **optional** fields to carry the new shapes without disturbing existing charts: `stacked: bool = False`, `y_label_secondary: str | None = None`, and a hierarchical `nodes: list | None = None` (for treemap/sunburst). Existing `series` (list of `{name, data, …}`) carries grouped-bar / dual-axis multi-series; per-series `type`/`yAxisIndex` hints live in the series dicts.
- `KeyMetric` + `Report.key_metrics` as in §1.

## 5. AI selection guidance (`prompts/selection_system.txt`)

Add to the existing (already-tuned) rules:
- **Always** open with `key_metrics` — the 3–5 numbers a reader most wants (a total, an average, a key rate, a notable count).
- A category split by a sub-category → `grouped_bar_chart` (grouped to compare, stacked to show composition).
- Two metrics on different scales over the same x (e.g. revenue + conversion %) → `dual_axis_chart`.
- A trend where cumulative volume matters → `line_chart` with `area: true`.
- A category's composition with many slices or a 2-level hierarchy → `treemap_chart` (prefer over pie when >8 categories).
Keep the soft cap (~3 bars of 10) — the new types make variety easier to hit.

## 6. Report structure, theme, credits

- **Structure:** `key_metrics` band on top → the 10 charts (now from the richer palette). `report/[id]/page.tsx` + the print route render the band.
- **Theme:** every new chart imports `charts/chartTheme.ts` (palette, stripped axes, `monoFamily()` labels, tooltip) — they match existing charts. The example report on the marketing landing is unaffected (it uses a fixed fixture).
- **Credits / generate-more / layout:** unchanged. New kinds participate in the existing 10-chart selection, the dnd-kit layout, hide-to-sidebar, and PDF export exactly like current charts. KPI tiles are layout-fixed (top band), not draggable.

## 7. Delivery phases (the plan will follow this order; each ships on its own)

1. **KPI tiles** — schema (`KeyMetric`/`Report.key_metrics`), `key_metrics` tool + executor, prompt line, `KpiTiles` frontend + PDF, tests.
2. **Bar/trend variants** — `grouped_bar_chart`, `dual_axis_chart`, `line area` flag; kinds + executors + components + prompt + tests.
3. **Part-to-whole** — `treemap_chart` (+ optional `sunburst`); kind + executor + component + prompt + tests.

## 8. Scope & non-goals

**In scope:** the tools/kinds/executors/components/schema/prompt above, with tests; KPI band rendering (app + PDF).

**Non-goals:**
- **Geo / maps** — its own brainstorm (needs profiler geo-detection + map assets).
- KPI **deltas** / period-over-period — fast-follow.
- funnel, waterfall, bubble, radar, calendar heatmap, Pareto — not now.
- No changes to credits, auth, the upload flow, or the report layout/dnd system beyond rendering new kinds + the KPI band.

## 9. Verification

- Backend: TDD per executor — each new executor has unit tests (valid spec out, guards reject bad input), plus `key_metrics` value-computation correctness (the computed number matches a hand-calc on a fixture). Full `pytest` green.
- Selection: a pipeline test that the new tools are registered + callable; the fallback/diversity tests still pass.
- Frontend: `tsc --noEmit` + `next build` clean; each new component renders its `spec` (verify via the report page / a fixture).
- The KPI band renders on the report page **and** in the PDF; new charts inherit the light theme.
- No regression: existing 7 chart kinds, generate-more, layout dnd, PDF export, credits all unchanged.
