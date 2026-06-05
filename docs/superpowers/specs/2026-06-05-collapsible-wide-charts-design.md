# Collapsible wide bar charts — design

*2026-06-05. Brainstormed + approved. Follows the chart-rendering fixes (percentage normalization, horizontal high-cardinality bars).*

## Problem

The report main area is a 2-column CSS grid (`grid-cols-1 lg:grid-cols-2`) where every chart renders at a fixed height (320px). A bar chart with many categories (e.g. 30 F1 circuits) either crams as vertical bars (colliding value-labels, rotated/unreadable x-axis names) or, rendered horizontal, grows tall — and a tall card stretches its grid row-mate (uniform-row default), leaving a dead gap. The grid assumes uniform card heights; a many-category ranking wants to be big.

## Decision

Let big bar charts **grow the layout**, and let each one be **collapsed** back to compact — per card, persisted.

- **Wide chart** = single-series `bar` with > 12 categories.
- **Default: expanded** — full-width (`col-span-2`), horizontal ranked leaderboard, all categories, height bounded ≤ ~760px.
- **Collapse/expand toggle** in the card chrome (wide charts only). **Collapsed** = compact top-12, horizontal, normal 320px card, `col-span-1` (rejoins the 2-up grid).
- The collapsed/expanded choice **persists** with the report (owner-curated), via the existing layout-persistence path.

## Components

**Backend**
- `schemas.ChartLayoutEntry` gains `collapsed: bool = False`. The existing `PATCH /report/{id}/layout` already does `entry.model_dump()` → `db.update_layout`, so the field persists with no new endpoint. Owner-gated (`require_owner=True`).

**Frontend**
- `isWideChart(spec)` — `spec.kind === 'bar' && (spec.x?.length ?? 0) > 12`.
- `useReportLayout` / `ChartLayoutEntry` type: add `collapsed`, and `toggleCollapse(chartId)` that flips it and debounce-PATCHes (same mechanism as `reorder`/`move`).
- `ChartCard`: render the toggle (wide only); apply `lg:col-span-2` when `wide && !collapsed`; pass `collapsed` to the chart.
- `BarChart`: horizontal; expanded → all categories, bounded height; collapsed → top-12, 320px. Keeps the whole-`%`-tick fix.

**Renders that inherit it** (all render from the layout): the interactive report, `/print` (PDF), `/embed`, and the `/og` hero image. A collapsed hero → compact top-12 → fits the 1200×630 OG card.

## Scope / non-goals

- Single-series `bar` only. `grouped_bar` / `dual_axis` unchanged.
- Non-owner viewers see the owner's curated state; their toggles don't persist (PATCH is owner-only) — acceptable.
- Grid gap: a full-width card landing mid-row can leave a half-row gap; acceptable (rankings are usually first). No `grid-auto-flow: dense` (it fights drag-reorder).

## Verification

- Backend: unit test that `ChartLayoutEntry` accepts/defaults `collapsed` and it survives `model_dump()` (the PATCH persistence path).
- Frontend: `tsc`; on the live deploy — drag-reorder still works with a full-width card; toggle persists across reload; a collapsed hero yields a clean landscape OG.
