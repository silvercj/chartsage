# ChartSage Rebuild — Design

**Date:** 2026-05-23
**Author:** Chris Silver (with Claude)
**Status:** Approved for implementation planning

## Context

ChartSage takes a CSV/Excel file, asks Claude to pick interesting charts, and renders them. The existing system has three problems that compound:

1. **The most common chart pattern is broken.** Frequency bar charts (the prompt instructs Claude to use these for half the dashboard) silently produce a histogram-of-counts instead of the actual frequency distribution. The bug is in [src/api/bar_chart_processor.py:167-181](../../../src/api/bar_chart_processor.py). The previous module ([src/api/chart_processing.py](../../../src/api/chart_processing.py)) implemented this correctly but was replaced during a refactor.
2. **The protocol between Claude and the backend is magic strings.** Claude emits derived field names like `group_count_by_activity_type`; the backend parses these strings with regex/prefix matching to figure out what to compute. The naming convention, the parser, and the compute logic drift independently. The 318-line prompt contradicts itself in places.
3. **The README advertises features that don't exist.** No auth, no Stripe, no PDF export, no PPT export, no shareable links are implemented despite being in `.env.example` and `README.md`. Empty `dashboard/` and `workers/` directories add to the confusion.

This rebuild fixes the engine and replaces the protocol, scoped to make end-to-end CSV → report work on any reasonable dataset. Auth, payments, and exports stay out of scope for v1.

## Goals

- A CSV upload produces a useful report (executive summary + 5-7 charts + data-quality callout) in under 10 seconds on Haiku 4.5.
- Claude cannot emit invalid chart specs. The Anthropic tool-use schema enforces shape; the backend validates semantics and feeds errors back for one retry round.
- Every chart we render is provably correct on a known input (golden unit tests).
- Switching model (Haiku → Sonnet → Opus) is a one-line env-var change with no code edits.
- Failed charts surface in the report's data-quality section, never silently disappear.

## Non-goals (v1)

- Authentication, payments, user accounts
- PDF/PowerPoint export
- Persistent reports beyond Redis 24h TTL
- Multi-user shareable links (beyond `?session=<id>` on localhost)
- Follow-up questions, regenerate button, iterative refinement
- Custom chart styling controls in the UI
- Frontend automated tests (manual eyeball for v1)

## Architecture

```
┌─────────────────┐      POST /generate-report          ┌──────────────────────────────┐
│  Next.js app    │  ──────────────────────────────→    │  FastAPI                     │
│  (page.tsx)     │  ←─── { session_id } ────           │                              │
│                 │                                      │  ┌────────────────────────┐  │
│  /report?id=… ──┼──── GET /report/{session_id} ───→   │  │ 1. profile_data(df)    │  │
│  (report page)  │  ←── { report } ─────                │  │    (pandas summary)    │  │
└─────────────────┘                                      │  ├────────────────────────┤  │
                                                          │  │ 2. Claude #1: pick     │  │
                                                          │  │    charts via parallel │  │
                                                          │  │    tool use (8 tools)  │  │
                                                          │  ├────────────────────────┤  │
                                                          │  │ 3. execute_tool_call() │  │
                                                          │  │    per tool → spec     │  │
                                                          │  ├────────────────────────┤  │
                                                          │  │ 4. Claude #2: write    │  │
                                                          │  │    summary + captions  │  │
                                                          │  ├────────────────────────┤  │
                                                          │  │ 5. assemble + store    │  │
                                                          │  │    in Redis (24h TTL)  │  │
                                                          │  └────────────────────────┘  │
                                                          └──────────────────────────────┘
```

**Tech stack stays the same:** Next.js 14, FastAPI, pandas, ECharts, Redis, Anthropic SDK. The change is internal: replace the magic-string protocol with tool use, replace the chart grid with a narrative report, drop the never-built auth/stripe/PDF scaffolding from docs and `.env.example`.

**What goes away:**

- `src/api/bar_chart_processor.py`
- `src/api/chart_processing.py`
- `src/api/insight_prompt.txt`
- `src/api/bar_chart_prompt.txt`
- `src/api/log_viewer.py` (unused outside one-off debugging)
- `src/api/__pycache__/pdf_generator.cpython-311.pyc` (orphan)
- `src/api/temp/*.pdf` (orphans)
- The `_last_uploaded_df` global in `main.py`
- The `print = lambda ...` rebind in `main.py`
- The `/upload` endpoint (preview happens client-side via PapaParse)
- The `/visualizations` and `/download/*` endpoints
- The orphan `src/app/dashboard/` and `src/app/workers/` directories
- The 28 historical log files in `src/api/logs/`
- Stripe and NextAuth env vars in `.env.example`
- Stripe and NextAuth claims in `README.md`

**What stays:**

- Next.js upload UX (with polish)
- Redis for session handoff (24h TTL)
- ECharts for chart rendering
- FastAPI base, anthropic SDK
- The per-run logging system (genuinely useful for debugging Claude responses)

## Components

### Backend

**`src/api/profile.py` — data understanding**

Single entry point: `profile_dataframe(df) → DataProfile`.

```python
class ColumnInfo(BaseModel):
    name: str
    dtype: str
    role: Literal["categorical", "numeric", "date", "identifier", "unusable"]
    cardinality: int
    null_count: int
    # role-specific:
    top_values: list[tuple[Any, int]] | None = None   # categorical
    min: float | None = None                          # numeric
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min_date: str | None = None                       # date (ISO)
    max_date: str | None = None
    unusable_reason: str | None = None

class DataProfile(BaseModel):
    row_count: int
    columns: list[ColumnInfo]
    correlations: dict[tuple[str, str], float]   # numeric pairs only
    anomalies: list[str]                          # human-readable warnings

    def to_text(self) -> str: ...                 # serialized for Claude
```

Role detection rules (evaluated in order, first match wins):
- `identifier` if name ends in `_id`/`_code`/`_uuid`/`_key` **or** (dtype is int/float **and** cardinality > 0.5 × row_count)
- `date` if `pd.to_datetime(col, errors='coerce')` parses ≥80% of non-null values
- `numeric` if dtype is int/float (rating-style 1-5 columns count — don't filter on cardinality)
- `categorical` if dtype is object **and** cardinality ≤ 50
- `unusable` otherwise (with reason populated — e.g., "object column with 1247 unique values, too high-cardinality to chart")

Anomaly detection:
- Negative values in columns whose name contains `duration|count|quantity|age|price|amount|revenue|sales`
- Date values in the future (>now + 1 year)
- Columns with >95% nulls
- Numeric columns with cardinality ≤ 2 (likely boolean or constant)
- Numeric columns with `max > mean + 10×std` (extreme outliers that will distort histograms)

**`src/api/chart_tools.py` — the typed protocol**

Eight Anthropic tool definitions. The schemas are the contract; deviating is impossible (Anthropic API rejects malformed calls).

| Tool name | Required params | Purpose |
|---|---|---|
| `frequency_bar_chart` | `column`, `title`, `intent` | Count by category |
| `aggregation_bar_chart` | `value_col`, `group_col`, `agg`, `title`, `intent` | Sum/mean/median/min/max by group |
| `histogram_chart` | `column`, `title`, `intent` | Numeric distribution; binning robust to outliers |
| `scatter_chart` | `x_col`, `y_col`, `color_by?`, `title`, `intent` | Correlation between two numerics |
| `line_chart` | `date_col`, `value_col`, `agg`, `granularity`, `group_by?`, `title`, `intent` | Time trends |
| `pie_chart` | `category_col`, `value_col?`, `title`, `intent` | Composition (cap 8 slices + "Other") |
| `box_plot` | `value_col`, `group_col?`, `title`, `intent` | Distribution shape, optionally per group |
| `heatmap_chart` | `mode` (`correlation`\|`pivot`), `params...`, `title`, `intent` | Correlation matrix or 2-D pivot |

`agg` is an enum whose allowed values depend on the tool:

- `aggregation_bar_chart.agg`: `sum|mean|median|min|max` (no `count` — that's `frequency_bar_chart`'s job)
- `line_chart.agg`: `count|sum|mean|median|min|max`
- `pie_chart.agg` (implicit when `value_col` provided): `sum|mean|count`
- `heatmap_chart.agg` (when `mode=pivot`): `sum|mean|count`

`granularity` is `day|week|month|quarter|year`. `intent` is Claude's own one-sentence rationale — fed back in pass #2 to write the user-facing caption. The selection prompt asks for **5-7 charts**; we accept 3-10 (under 3 triggers fallback; over 10 we keep the first 10).

**`src/api/chart_executor.py` — the safe compute layer**

One executor per tool. Each is a pure function: `(df: DataFrame, params: dict) → ChartSpec | ToolError`.

```python
class ChartSpec(BaseModel):
    kind: Literal["bar", "histogram", "scatter", "line", "pie", "box", "heatmap"]
    title: str
    intent: str
    # Data (shapes differ by kind):
    x: list[Any] | None = None
    y: list[Any] | None = None
    series: list[dict] | None = None      # for multi-series (line w/ group_by, heatmap)
    # Axis/display:
    x_label: str
    y_label: str
    x_display_type: Literal["category", "number", "date", "text"]
    y_display_type: Literal["count", "currency", "percentage", "number"]
    # Provenance:
    source_columns: list[str]
    data_point_count: int

class ToolError(BaseModel):
    reason: str   # specific, actionable, includes available alternatives
```

Each executor must:
1. Validate every referenced column exists in `df`.
2. Validate every column's role/dtype matches expectations (e.g., `value_col` must be numeric).
3. Validate cardinality constraints (≤30 categories for bar/pie; ≤8 for pie before "Other" rollup).
4. Drop NaN rows where mathematically required (correlations, aggregations on the value col).
5. Return `ChartSpec` with x/y same length and no NaN/Inf in the data arrays.
6. On any failure, return `ToolError` whose `reason` names the offending column and offers concrete alternatives (e.g., `"'revenue' is not a column. Available numeric: ['duration_minutes', 'activity_id']"`).

Histogram-specific: bin selection uses trimmed-IQR (drop rows beyond Q1 - 3·IQR or Q3 + 3·IQR before binning) to prevent the outlier-driven empty-bin problem. Bin count is `min(20, max(5, ceil(2 · n^(1/3))))` on the trimmed data.

**`src/api/report_generator.py` — orchestrator**

```python
class ReportGenerator:
    def __init__(self, profile: DataProfile, df: pd.DataFrame): ...

    def generate_charts(self) -> list[ChartSpec]:
        """Pass #1: tool-use selection + 1 retry round + fallback."""

    def generate_narrative(self, charts: list[ChartSpec]) -> ReportNarrative:
        """Pass #2: summary + captions + data-quality notes."""

    def build_report(self) -> Report:
        charts = self.generate_charts()
        narrative = self.generate_narrative(charts)
        return Report(...)
```

Fallback path (when pass #1 yields <3 successful charts after retry) lives in `src/api/fallback.py`:
- Frequency chart for the top 2 categorical columns with cardinality ≤30
- Histogram for the top 2 numeric columns
- Scatter for the numeric pair with the highest |correlation| ≥ 0.3 (if any)

**`src/api/claude_client.py` — thin wrapper**

```python
def messages_create(messages, tools=None, model=None, system=None, cache_static=True):
    """Wrap anthropic SDK. Add cache_control to system + tools on cache_static=True.
       Retry transient 5xx with exp backoff (1s, 2s, 4s). Surface 529 as RetryableBusy."""
```

**`src/api/llm_config.py` — model selection**

```python
MODEL_ALIASES = {
    "haiku-4-5":  "claude-haiku-4-5-20251001",
    "sonnet-4-6": "claude-sonnet-4-6",
    "opus-4-7":   "claude-opus-4-7",
}

def resolve(alias_or_id: str) -> str:
    return MODEL_ALIASES.get(alias_or_id, alias_or_id)

# Resolution order: pass-specific env > generic env > default
MODEL_SELECTION = resolve(getenv("CLAUDE_MODEL_SELECTION") or getenv("CLAUDE_MODEL") or "haiku-4-5")
MODEL_NARRATIVE = resolve(getenv("CLAUDE_MODEL_NARRATIVE") or getenv("CLAUDE_MODEL") or "haiku-4-5")
```

**`src/api/main.py` — FastAPI**

Two endpoints:

```python
POST /generate-report
  Body: multipart file
  Response: { "session_id": str }

GET /report/{session_id}
  Response: Report JSON
```

All other endpoints removed. The `/logs` endpoints stay for backend debugging (read-only, dev-only — gate with `ENABLE_LOG_ENDPOINTS=true`).

### Frontend

**`src/app/page.tsx`** — upload page, kept mostly as-is. Polish: drop the dead `handleViewVisualizations`, drop unused `busyCountdown` state, fix the post-to-session-dashboard branch (no longer needed since `/generate-report` returns session_id directly), restore CORS-correct fetch.

**`src/app/report/[id]/page.tsx`** — fetches the report, renders three blocks:

1. `<ReportSummary>` — title, generated-on date, executive summary paragraphs.
2. `<ChartGrid>` — 2-column grid of `<ChartCard>`.
3. `<DataQualityCallout>` — only renders if `anomalies` is non-empty; yellow banner.

(Path uses `[id]` route param instead of `?session=` query — cleaner URLs and matches Next.js conventions.)

**`src/app/report/[id]/ChartCard.tsx`** — switches on `spec.kind`, renders one of:
`<BarChart>`, `<HistogramChart>`, `<ScatterChart>`, `<LineChart>`, `<PieChart>`, `<BoxPlot>`, `<Heatmap>`.

Each is a thin ECharts wrapper. Shared `lib/format.ts` handles number formatting (the current `formatNumber` + `shortCurrency` + `getFormatter` move there).

The current `VisualizationCard.tsx` is deleted; its bar-rendering logic moves into `<BarChart>`. The old `visualizations/` route is deleted.

## Data flow

### Step 1 — Profile (synchronous, ~50ms)

`profile = profile_dataframe(df)` produces a `DataProfile` with column roles, basic stats, correlations, and anomalies. This profile is the only data context Claude sees in pass #1. The raw dataframe is never sent to Claude.

### Step 2 — Claude call #1: parallel tool use for chart selection

```python
response = claude_client.messages_create(
    model=MODEL_SELECTION,
    max_tokens=4096,
    system=SELECTION_SYSTEM_PROMPT,         # static, cached
    tools=CHART_TOOLS,                       # static, cached
    messages=[{"role": "user", "content": profile.to_text()}],
)
```

Claude returns 5-7 parallel `tool_use` blocks. The selection prompt is ~30 lines: "Here's a profile. Pick 5-7 charts that tell the most useful story. If `anomalies` mentions outliers, prefer median over mean. If a column is `role=identifier`, never use it as a metric. Reference any anomalies in your `intent`."

### Step 3 — Execute tool calls

```python
specs, errors = [], []
for tc in response.content:
    if tc.type != "tool_use": continue
    result = TOOL_EXECUTORS[tc.name](df, tc.input)
    (specs if isinstance(result, ChartSpec) else errors).append((tc.id, result))
```

### Step 4 — Retry on tool errors (max 1 round)

If `errors` is non-empty, build a `tool_result` user message with `is_error=True` and specific reasons, then call Claude again with the conversation history. Execute new `tool_use` blocks. After retry, drop still-failing tools and keep the rest.

### Step 5 — Fallback if needed

If `len(specs) < 3` after retry, run `fallback.pick_charts(profile, df)` to fill the gap. Mark fallback charts with `intent="fallback"`.

### Step 6 — Claude call #2: narrative

```python
narrative_response = claude_client.messages_create(
    model=MODEL_NARRATIVE,
    max_tokens=2048,
    system=NARRATIVE_SYSTEM_PROMPT,
    tools=[SUBMIT_NARRATIVE_TOOL],          # single-tool forced output (cleaner than parsing markdown)
    tool_choice={"type": "tool", "name": "submit_narrative"},
    messages=[{"role": "user", "content": format_charts_for_narrative(profile, specs)}],
)
narrative = narrative_response.content[0].input   # validated dict
```

The `submit_narrative` tool's schema:
```json
{
  "name": "submit_narrative",
  "input_schema": {
    "type": "object",
    "required": ["summary", "captions", "data_quality"],
    "properties": {
      "summary": {"type": "string", "description": "2-3 paragraph executive summary."},
      "captions": {"type": "array", "items": {"type": "string"}, "description": "One 1-2 sentence caption per chart, in order."},
      "data_quality": {"type": "array", "items": {"type": "string"}, "description": "Notes for the user about data issues. Empty array if none."}
    }
  }
}
```

This avoids markdown-parsing fragility.

### Step 7 — Assemble and store

```python
report = Report(
    generated_at=datetime.utcnow().isoformat(),
    summary=narrative["summary"],
    data_quality=narrative["data_quality"],
    charts=[ChartWithCaption(spec=s, caption=c) for s, c in zip(specs, narrative["captions"])],
    metadata={"model_selection": MODEL_SELECTION, "model_narrative": MODEL_NARRATIVE, ...},
)
session_id = uuid4().hex
redis.set(f"report:{session_id}", report.model_dump_json(), ex=86400)
return {"session_id": session_id}
```

### Step 8 — Frontend renders

`/report/{session_id}` GETs the report, renders the three blocks.

### Cost & latency budget

- Pass 1: ~2K in + ~700 out ≈ $0.005 on Haiku 4.5
- Pass 2: ~3K in + ~600 out ≈ $0.006 on Haiku 4.5
- **Total: ~$0.011 per report, ~6-8 seconds end-to-end.**
- Sonnet 4.6 equivalent: ~$0.035/report. Opus 4.7: ~$0.040/report.

## Error handling

### Failure surface map

| Where | What can go wrong | Response |
|---|---|---|
| **File upload** | Wrong type, >10MB, corrupt, empty, 0-1 columns | 422 with specific reason. Frontend shows the reason inline. |
| **Profiling** | All-null column, mixed-dtype column, date column with no parseable dates | Mark column as `role="unusable"` with reason. Don't fail the request. |
| **Claude API: network/5xx** | Transient | Exp backoff: 1s, 2s, 4s. Max 3 attempts. Then 503 to client. |
| **Claude API: 429/529** | Rate-limited or overloaded | Same backoff; surface "model is busy, please retry in 30s". |
| **Claude API: 4xx (auth, bad request)** | Misconfig | Fail-fast, log loudly, 500 with generic message. Real error in log only. |
| **Pass #1 tool errors** | Claude picked invalid columns / dtypes | `ToolError` → `tool_result` with `is_error=True` → 1 retry. Drop still-failing tools. |
| **Pass #1: zero charts after retry** | Claude completely failed or data is too thin | Fall back to heuristic chart picker (`fallback.py`). |
| **Pass #2: forced-tool-use fails** | Extremely rare with `tool_choice="tool"` | Template fallback: summary = "Automated analysis of <filename>", captions = chart `intent` strings, data_quality = profile anomalies verbatim. |
| **Frontend: session not found (>24h)** | Report expired | "This report has expired. Generate a new one." |
| **Frontend: empty report** | Backend somehow returned 0 charts | "We couldn't pull useful insights from this data" + data quality notes. |

### What we explicitly don't do

- **No silent chart drops.** Every failed tool call is logged with reason. If final chart count is below 3, the user sees a "we had trouble with this data" note in the report.
- **No automatic data cleaning.** Anomalies surface in the report; the data is presented as-is. (Auto-cleaning is dangerous — a negative revenue might be a legit refund.)
- **No retry loops on pass #2.** Forced tool use makes malformed output a non-issue; if it somehow fails, template-fallback rather than burning Claude time.
- **No global mutable state.** Every request is independent. The `_last_uploaded_df` pattern is gone.

### Logging

Per-run log file (existing pattern), plus a structured trailer:

```
=== RUN SUMMARY ===
run_id: 7a2c8f4e
file: raw_activities.csv
rows: 2007  cols: 5
model_selection: claude-haiku-4-5-20251001
model_narrative: claude-haiku-4-5-20251001
pass1_attempts: 1  pass1_tools_called: 7  pass1_tools_succeeded: 7
pass2_attempts: 1
input_tokens_total: 4823  (cached: 1421)  output_tokens_total: 1612
estimated_cost_usd: 0.0118
elapsed_ms: 6234
result: success
```

Log rotation: keep last 50 run files, delete older. Implemented in `setup_run_logging()`.

## Testing

### Tier 1 — Unit tests for tool executors (~50 tests, ~1s)

For each of the 8 executors:

- **Golden happy-path test** with hand-computed expected `x` and `y` values:
  ```python
  def test_frequency_bar_chart_happy_path():
      df = pd.DataFrame({"activity_type": ["a", "b", "a", "a", "b", "c"]})
      spec = execute_frequency_bar_chart(df, {"column": "activity_type", "title": "...", "intent": "..."})
      assert spec.x == ["a", "b", "c"]
      assert spec.y == [3, 2, 1]
  ```
  This is the regression net that would have caught the original frequency-chart bug.

- **Error-path tests**: missing column, wrong dtype, all-null, cardinality blown, every documented `ToolError` reason. Assert the reason string mentions the offending column and offers alternatives.

- **Edge cases per chart type**: scatter with NaN rows, histogram with constant values, line chart with one date, pie with one category, heatmap with no numeric pairs.

### Tier 2 — Pipeline integration tests with mocked Claude (~15 tests, ~3s)

`tests/helpers/fake_claude.py` provides a callable mock that returns canned `content` blocks based on the call number.

Scenarios:

1. Happy path: all tool calls succeed, narrative parses cleanly.
2. One tool call errors, retry succeeds. Verifies the `tool_result` round-trip.
3. All tool calls error twice → falls back to heuristic picker. Verify fallback produces ≥3 charts.
4. Claude returns no `tool_use` blocks at all → fallback path.
5. Anthropic 529 overloaded → 503 to client with the right message.
6. Anthropic transient 5xx → retries then succeeds.
7. File upload validation: too big, wrong type, corrupt CSV, empty.

### Tier 3 — End-to-end smoke tests against real Claude (~5 tests, ~60s, nightly + on-demand)

Five small CSVs in `tests/e2e/fixtures/`:

- `activities.csv` — existing test data; has anomalies (negative durations, ID-looking patient_id)
- `sales.csv` — clean revenue/region/date
- `signups.csv` — daily time-series count
- `survey.csv` — categorical responses
- `degenerate.csv` — 1 column, mostly null

For each, run the real pipeline and assert structural properties:

- Returns 200, has a `session_id`
- Report has ≥3 charts (≥1 for `degenerate.csv`)
- Each chart spec is valid (x/y same length, no NaN/Inf)
- For `activities.csv`: summary mentions "negative" or "outlier" (data-quality awareness check)
- Total tokens under budget (~6000 in, ~2000 out)

E2E tests run on `make test-e2e` and nightly CI, gated by `ANTHROPIC_API_KEY` and a `RUN_E2E=true` env var.

### Test directory layout

```
tests/
├── conftest.py                  # shared DataFrame fixtures
├── unit/
│   ├── test_profile.py
│   ├── test_executors_frequency.py
│   ├── test_executors_aggregation.py
│   ├── test_executors_histogram.py
│   ├── test_executors_scatter.py
│   ├── test_executors_line.py
│   ├── test_executors_pie.py
│   ├── test_executors_box.py
│   └── test_executors_heatmap.py
├── integration/
│   ├── test_pipeline_happy.py
│   ├── test_pipeline_retry.py
│   ├── test_pipeline_fallback.py
│   └── test_api_errors.py
├── e2e/
│   ├── test_real_claude_smoke.py    # opt-in via RUN_E2E=true
│   └── fixtures/
│       ├── activities.csv
│       ├── sales.csv
│       ├── signups.csv
│       ├── survey.csv
│       └── degenerate.csv
└── helpers/
    ├── fake_claude.py
    └── builders.py
```

### CI wiring

- `make test` → unit + integration (~4s, every commit)
- `make test-e2e` → real Claude smoke tests (~70s, nightly + on `[e2e]` in commit message)
- `make smoke` → boot FastAPI, upload one fixture, verify a report renders (in CI)

### What we deliberately don't test

- **No frontend tests in v1.** ECharts is solid, rendering logic is thin. Manual eyeball after each meaningful change.
- **No load tests.**
- **No "did Claude pick the right charts?" test.** Quality judgment stays manual; tests verify structural validity only.

## Open questions

None — all questions surfaced during brainstorming are resolved in the sections above.

## Appendix: Files touched

### New files

- `src/api/profile.py`
- `src/api/chart_tools.py`
- `src/api/chart_executor.py`
- `src/api/report_generator.py`
- `src/api/fallback.py`
- `src/api/claude_client.py`
- `src/api/llm_config.py`
- `src/api/schemas.py` (Pydantic models: `ChartSpec`, `ToolError`, `Report`, etc.)
- `src/app/report/[id]/page.tsx`
- `src/app/report/[id]/ChartCard.tsx`
- `src/app/report/[id]/ReportSummary.tsx`
- `src/app/report/[id]/DataQualityCallout.tsx`
- `src/app/report/[id]/charts/BarChart.tsx`
- `src/app/report/[id]/charts/HistogramChart.tsx`
- `src/app/report/[id]/charts/ScatterChart.tsx`
- `src/app/report/[id]/charts/LineChart.tsx`
- `src/app/report/[id]/charts/PieChart.tsx`
- `src/app/report/[id]/charts/BoxPlot.tsx`
- `src/app/report/[id]/charts/Heatmap.tsx`
- `src/app/lib/format.ts`
- `tests/` (full tree above)
- `Makefile`
- `prompts/selection_system.txt`
- `prompts/narrative_system.txt`

### Modified files

- `src/api/main.py` — strip to just `/generate-report` + `/report/{id}` (+ optional log endpoints)
- `src/api/data_processing_utils.py` — `compute_group_count` stays, `compute_histogram_bins_and_freqs` rewritten with trimmed-IQR binning
- `src/api/derived_fields.py` — kept only if any helper inside is genuinely used by the new executors; otherwise deleted entirely
- `src/app/page.tsx` — polish (drop dead state, fix navigation to `/report/[id]`)
- `README.md` — remove Auth/Stripe/PDF/PPT claims; document actual setup
- `.env.example` — remove Stripe/NextAuth vars; add `CLAUDE_MODEL`, `CLAUDE_MODEL_SELECTION`, `CLAUDE_MODEL_NARRATIVE`, `ENABLE_LOG_ENDPOINTS`
- `requirements.txt` — add pytest, add pydantic v2 if not already pinned correctly
- `ChartSage.md` — align with reality

### Deleted files / directories

- `src/api/bar_chart_processor.py`
- `src/api/chart_processing.py`
- `src/api/insight_prompt.txt`
- `src/api/bar_chart_prompt.txt`
- `src/api/log_viewer.py`
- `src/api/__pycache__/` (regenerated)
- `src/api/temp/` (orphan PDFs)
- `src/api/logs/*.log` (28 stale files; new ones generated as needed)
- `src/api/field_type_utils.py` (unused by the new pipeline)
- `src/app/visualizations/` (entire directory)
- `src/app/dashboard/` (empty)
- `src/app/workers/` (empty)
- `BAR_CHART_SYSTEM.md` (superseded by this doc)
- `REFACTORING_SUMMARY.md` (historical)
- `src/api/CHART_GENERATION_FIXES.md` (historical)
- `src/api/LOGGING_SYSTEM.md` (folded into this doc)
- `cursor_rule_prompt_json_formatting.mdc` (vestigial)
