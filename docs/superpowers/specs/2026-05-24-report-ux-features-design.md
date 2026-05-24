# Report UX Features — Design

**Date:** 2026-05-24
**Author:** Chris Silver (with Claude)
**Status:** Approved for implementation planning

## Context

The v2 report is currently a static, scrollable layout: collapsible-by-OS narrative at the top, then a fixed grid of 10 charts. This spec adds four pieces of interactivity that turn it from a one-shot artifact into something users can shape and share:

1. **Collapsible insight summary** — clear the deck when the user just wants to look at the charts
2. **Drag-and-drop chart reordering** — let users put the chart that matters most at the top
3. **Sidebar of "extra" charts** — out of 10 generated charts, 5 sit in the main report and 5 in a side rail; users can swap between
4. **"Generate 5 more" button** — request more charts on demand (paywall hook for the future)
5. **PDF export** — server-rendered, identical for every user, distributable to stakeholders

These plug into the existing pipeline without changing the chart generation logic. The 10-chart Claude call remains the foundation; what's new is everything that happens to the report after the user opens it.

## Goals

- Users can promote, demote, and reorder charts; their layout persists on reload and shows in the exported PDF
- The "Generate more" path produces 5 additional charts that are different from the existing set (Claude is told what to avoid)
- PDF export produces a clean, identical-for-everyone document with crisp vector charts
- Layout edits feel instant: optimistic UI updates, with a debounced persistence write
- No regressions on existing tests (109 passing)

## Non-goals (v1.1)

- Multi-user collaborative editing on the same report
- Undo / redo of layout edits
- Custom PDF templates or branding swap
- "Restore default layout" button
- Paywall enforcement on "Generate 5 more" (just leave the hook)
- Frontend automated tests (continue to manual-test)

## Data model

The existing `Report` JSON gains a `layout` array. Each chart gets a stable `chart_id`. Charts themselves stay immutable; layout is the mutable interaction state.

```python
class ChartLayoutEntry(BaseModel):
    chart_id: str                              # UUID, stable per chart
    position: Literal["main", "sidebar"]
    order: int                                 # 0-indexed within its position

class ChartWithCaption(BaseModel):
    chart_id: str                              # NEW
    spec: ChartSpec
    caption: str

class Report(BaseModel):
    generated_at: str
    summary: str
    data_quality: list[str]
    charts: list[ChartWithCaption]
    layout: list[ChartLayoutEntry]             # NEW
    metadata: dict[str, Any]
```

**Default layout** (built in `build_report`):
- Charts in the order Claude's tool calls came back
- First 5 → `position="main"`, `order=0..4`
- Next 5 → `position="sidebar"`, `order=0..4`

**Why separate layout array instead of mutating `charts`:** charts are an immutable artifact of one Claude call; layout is the user's interaction state. Separation makes the "Generate more" flow clean (new charts get appended without touching existing positions) and keeps the spec readable.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Browser                                                       │
│                                                                │
│  ┌──────────────────────────────┐  ┌────────────────────────┐ │
│  │ Toolbar                       │  │                        │ │
│  │  Export PDF · Generate 5 more │  │                        │ │
│  ├──────────────────────────────┤  │                        │ │
│  │ ReportSummary (collapsible)   │  │                        │ │
│  ├──────────────────────────────┤  │   Sidebar               │ │
│  │ DataQualityCallout            │  │   ┌──────────────┐    │ │
│  ├──────────────────────────────┤  │   │ SidebarCard 1│    │ │
│  │ MainGrid                      │  │   │   → main     │    │ │
│  │  ┌──────┐ ┌──────┐            │  │   ├──────────────┤    │ │
│  │  │ Card │ │ Card │            │  │   │ SidebarCard 2│    │ │
│  │  │ 01 ⋮│ │ 02 ⋮│ ×          │  │   ├──────────────┤    │ │
│  │  └──────┘ └──────┘            │  │   │ SidebarCard 3│    │ │
│  │  drag handle ⋮  hide ×        │  │   └──────────────┘    │ │
│  └──────────────────────────────┘  └────────────────────────┘ │
│                                                                │
│  useReportLayout hook owns state, dispatches optimistic        │
│  updates + debounced PATCH                                     │
└──────────────────────────────────────────────────────────────┘
              ↓ PATCH /report/{id}/layout
              ↓ POST  /report/{id}/generate-more
              ↓ GET   /report/{id}/export.pdf
┌──────────────────────────────────────────────────────────────┐
│  FastAPI                                                       │
│   ┌──────────────────────┐  ┌──────────────────────────────┐ │
│   │ PATCH layout         │  │ POST generate-more           │ │
│   │  - validate chart_ids │  │  - Claude pass-1 with        │ │
│   │  - write to Redis    │  │    "don't repeat these"      │ │
│   └──────────────────────┘  │  - append charts + layout    │ │
│                              └──────────────────────────────┘ │
│   ┌──────────────────────────────────────────────────────┐   │
│   │ GET export.pdf                                         │   │
│   │  - launch Playwright Chromium (lazy, reused)          │   │
│   │  - navigate to /report/{id}/print                     │   │
│   │  - wait for data-charts-ready flag                    │   │
│   │  - page.pdf({ format: A4, printBackground: true })   │   │
│   └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

## Components

### Frontend (`src/app/report/[id]/`)

**New:**
- `Sidebar.tsx` — collapsible right rail listing `SidebarCard`s, sorted by layout `order`
- `SidebarCard.tsx` — compact card (title, chart-kind label, "Move to main" button)
- `Toolbar.tsx` — sticky top bar with **Export PDF** and **Generate 5 more** buttons
- `useReportLayout.ts` — hook owning layout state; exposes `reorder(chartId, position, newOrder)`, `move(chartId, newPosition)`, debounced backend sync
- `print/page.tsx` — print-only route at `/report/[id]/print`, renders MainGrid only, sets `data-charts-ready="true"` on the body once all ECharts components have rendered

**Modified:**
- `page.tsx` — wraps content in `<DndContext>` from `@dnd-kit/core`; two-column layout (main ~75% / sidebar ~25%); sidebar can collapse to a thin spine
- `ReportSummary.tsx` — chevron toggle on the header; body height transitions smoothly. Uses the `<details>` element for accessibility (keyboard- and screen-reader-friendly)
- `ChartCard.tsx` — drag handle replaces the leading number (number moves to right-side badge area); adds a small `×` hide button in the upper-right; index numbering is now computed by position in the main grid

### Backend (`src/api/`)

**New endpoints (in `main.py`):**

- `PATCH /report/{session_id}/layout`
  - Body: `list[ChartLayoutEntry]`
  - Validates that every `chart_id` referenced exists in the stored report's `charts`
  - Writes updated `Report` back to Redis, refreshes TTL to 24h
  - Returns `204 No Content` on success

- `POST /report/{session_id}/generate-more`
  - Loads `Report` from Redis
  - Builds a Claude pass-1 message: profile + "Existing charts: [titles + intents]. Pick 5 different angles."
  - Runs the existing tool-use flow (parallel calls + 1 retry + executors)
  - Appends new `ChartWithCaption` entries to `charts[]` with new `chart_id`s
  - Appends new `ChartLayoutEntry` rows with `position="sidebar"`, `order` continuing from the existing max
  - Writes updated `Report` to Redis
  - Returns the updated `Report`

- `GET /report/{session_id}/export.pdf`
  - Delegates to `pdf_export.render_report_pdf(session_id)`
  - Returns `application/pdf` stream with `Content-Disposition: attachment; filename="chartsage-{short_id}.pdf"`

**New module:** `pdf_export.py`

- Owns a single Playwright `BrowserContext` lifecycle. Started lazily on first export. Held open for reuse (subsequent exports are fast).
- `async def render_report_pdf(session_id: str) -> bytes`:
  - Opens a new `Page`
  - Navigates to `http://localhost:3000/report/{session_id}/print`
  - Waits for the page to set `body[data-charts-ready="true"]` (10s timeout, then best-effort)
  - Calls `page.pdf({ format: 'A4', printBackground: True, margin: { top: '50px', bottom: '50px', left: '40px', right: '40px' } })`
  - Closes the page (keeps context alive for the next export)
  - Returns bytes

**Modified module:** `report_generator.py`

- `build_report()`: assigns `chart_id = uuid4().hex` to each chart; builds the default layout (first 5 → main, next 5 → sidebar, both starting at `order=0`)
- New method `generate_more(report: Report) -> Report`: runs the targeted Claude pass; mutates and returns the report

**Modified module:** `schemas.py`

- Add `chart_id: str` to `ChartWithCaption`
- Add `ChartLayoutEntry` model
- Add `layout: list[ChartLayoutEntry]` to `Report`

### Dependencies added

- `@dnd-kit/core` + `@dnd-kit/sortable` (~35KB total)
- `playwright` Python package
- `playwright install chromium` (one-time, ~150MB binary cache)

## Data flow

### Initial render

```
Browser → GET /report/{id}
  → FastAPI returns Report (with layout)
  → page.tsx partitions charts by layout.position
  → MainGrid renders main charts sorted by order
  → Sidebar renders sidebar charts sorted by order
```

### Drag to reorder within main

```
User drags card 03 → drops between 01 and 02
  → dnd-kit onDragEnd fires { active.id, over.id }
  → useReportLayout.reorder(activeId, mainIndex=1)
  → Optimistic state update (instant re-render, compact orders)
  → 500ms debounced PATCH /report/{id}/layout with the full new layout array
  → Backend validates and writes to Redis
  → If PATCH fails: revert UI to last-confirmed state, toast "Couldn't save your layout"
```

### Cross-context drag (main → sidebar) OR hide button click

The hide button is just a shortcut for "drag this to the sidebar". Both go through the same reducer:

```
useReportLayout.move(chartId, "sidebar")
  → entry's position flips
  → entry's order set to max(sidebar.order)+1
  → other main entries' orders compact down to remove the gap
  → optimistic + debounced PATCH
```

Symmetric for the "Move to main" button.

### Generate 5 more

```
User clicks "Generate 5 more"
  → Button shows spinner; disabled
  → POST /report/{id}/generate-more
    → Backend loads Report
    → Builds pass-1 message: data profile + "Existing charts: <titles>. Pick 5 different angles."
    → Runs tool-use flow + 1 retry + executors
    → Appends new ChartWithCaption with new chart_ids
    → Appends ChartLayoutEntry with position="sidebar", order continuing from existing sidebar max
    → Writes Report back to Redis with refreshed 24h TTL
    → Returns updated Report
  → Frontend replaces state with response
  → New cards animate into the sidebar
  → Spinner clears, button re-enables
```

**Cost:** roughly one Haiku selection pass per click (~$0.005). Paywall hook eventually slots in as middleware that 402s when the user isn't entitled.

### Export PDF

```
User clicks "Export PDF"
  → window.open(`${API}/report/${id}/export.pdf`, "_blank")
  → Backend lazily starts Playwright + Chromium (first call only, ~3-5s)
  → Subsequent calls reuse the open BrowserContext (~1-2s)
  → Playwright opens a new Page at /report/{id}/print
  → Print route renders MainGrid only (no sidebar, no toolbar):
     - Summary on page 1 in full
     - Charts paginated 1 or 2 per page depending on chart count
     - Page header: report title + generated date
     - Page footer: page number + "ChartSage"
     - All charts fit page width
  → Print page sets body[data-charts-ready="true"] after all <ReactECharts> components mount
  → Playwright waits for that flag (10s timeout, best-effort fallback)
  → page.pdf({ format: 'A4', printBackground: true })
  → Stream bytes back as application/pdf
  → Filename: chartsage-{session_id[:8]}.pdf
  → Browser downloads
```

## Error handling

| Where | What can go wrong | Response |
|---|---|---|
| **PATCH layout** | Invalid `chart_id`, malformed entry | 400 with reason; frontend reverts optimistic state; toast |
| **PATCH layout** | Session expired (404) | Toast: "This report has expired. Generate a new one." with link back to `/` |
| **PATCH layout** | Redis write error | 500; toast: "Couldn't save layout. Your changes are local only." Stop trying to PATCH for this session |
| **Generate more: Claude busy** | 529 / RetryableBusy | 503 to client; toolbar shows inline "Claude is busy, try again in 30s"; button re-enables |
| **Generate more: all tool calls error after retry** | Rare | Return the existing report unchanged; toast: "Couldn't generate new angles for this dataset." |
| **PDF export: Playwright fails to launch** | Cold-start race, missing binary | 500; user sees inline toolbar error with "Try again" |
| **PDF export: print route timeout (>10s)** | ECharts didn't signal ready | Playwright proceeds anyway (best-effort); log a warning; quality may degrade but PDF still returns |
| **Drag drop outside any container** | dnd-kit's `over` is null | No-op |
| **Drag drop on self** | Same start and end position | No-op |
| **Concurrent generate-more + PATCH** | Race on the same Redis key | Last-write-wins (acceptable — losing one layout edit is minor) |

## Testing

**Backend unit tests (new):**

`tests/unit/test_report_generator_layout.py`:
- `build_report` assigns unique `chart_id`s
- Default layout: first 5 → main, next 5 → sidebar, orders 0-4 in each
- `generate_more`: appends new charts with `position="sidebar"`, preserves existing layout, fresh chart_ids, orders continue from existing sidebar max

`tests/integration/test_api_layout.py`:
- `PATCH /report/{id}/layout` with valid body returns 204 and updates Redis state
- PATCH with unknown `chart_id` returns 400
- PATCH on expired session returns 404
- `POST /report/{id}/generate-more` (mocked Claude) appends 5 charts to sidebar

`tests/unit/test_pdf_export.py` (opt-in, gated by `RUN_PDF_TESTS=true`):
- Generates a fixture report in Redis
- Calls `render_report_pdf(session_id)`
- Asserts bytes start with `%PDF-` and are >10KB

**Manual smoke test plan:**

- [ ] Drag chart 03 between 01 and 02 → 02 becomes 03, 03 becomes 02; reload preserves order
- [ ] Click `×` on chart 04 → animates to sidebar; sidebar count = 6
- [ ] Click "Move to main" on a sidebar card → animates back to main, takes the next main slot
- [ ] Reload → custom layout persists
- [ ] Click "Generate 5 more" → spinner, then 5 new cards appear in sidebar
- [ ] Collapse insight summary → body hides, chevron rotates; expand → body returns
- [ ] Export PDF on a 10-chart report → PDF opens, contains only the 5 main charts + summary + page numbers
- [ ] Export after hiding charts → only the currently-visible main charts appear

## File structure

### New files

- `src/api/pdf_export.py`
- `src/app/report/[id]/Sidebar.tsx`
- `src/app/report/[id]/SidebarCard.tsx`
- `src/app/report/[id]/Toolbar.tsx`
- `src/app/report/[id]/useReportLayout.ts`
- `src/app/report/[id]/print/page.tsx`
- `tests/unit/test_report_generator_layout.py`
- `tests/integration/test_api_layout.py`
- `tests/unit/test_pdf_export.py`

### Modified files

- `src/api/schemas.py` — add `ChartLayoutEntry`, `chart_id` on `ChartWithCaption`, `layout` on `Report`
- `src/api/report_generator.py` — assign `chart_id`, build default layout, add `generate_more`
- `src/api/main.py` — add 3 new endpoints
- `src/app/report/[id]/page.tsx` — DnD context + two-column layout
- `src/app/report/[id]/ReportSummary.tsx` — collapse toggle
- `src/app/report/[id]/ChartCard.tsx` — drag handle + hide button
- `package.json` — add `@dnd-kit/core`, `@dnd-kit/sortable`
- `requirements.txt` — add `playwright`
- `Makefile` — add `test-pdf` target

## Open questions

None — all decisions resolved in the sections above.
