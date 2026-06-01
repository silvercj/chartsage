# ChartSage Chart-Types Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> **Subagents must NOT run `git checkout`, `git switch`, `git reset`, or `git stash`.** Stay on branch `chart-expansion`; commit only the files each task lists.

**Goal:** Add a headline KPI band + four new chart options (grouped/stacked bar, dual-axis combo, area, treemap) to the AI's vocabulary, so reports are richer.

**Architecture:** New Anthropic tools (`chart_tools.py`) → executors (`chart_executor.py`, registered in `TOOL_EXECUTORS`) → `report_generator.py` assembles the `Report`. `key_metrics` is special-cased (it's not a chart — it populates `Report.key_metrics`). New chart kinds render via `ChartContent.tsx` + a per-kind component, themed by `charts/chartTheme.ts`. Backend is TDD; frontend is implement → `tsc` → commit. Geo is out of scope (separate brainstorm).

**Tech Stack:** FastAPI/pandas/Pydantic, Anthropic tool-use, Next.js, ECharts.

**Spec:** `docs/superpowers/specs/2026-06-01-chart-types-expansion-design.md`
**Branch:** `chart-expansion` (carries the spec).

---

## File Structure

**Backend (`src/api/`)**
- `schemas.py` — `KeyMetric` model + `Report.key_metrics`; `ChartKind` gains `grouped_bar`/`dual_axis`/`treemap` (+`sunburst`); `ChartSpec` gains optional `stacked`, `y_label_secondary`, `nodes`.
- `chart_tools.py` — new tool defs: `key_metrics`, `grouped_bar_chart`, `dual_axis_chart`, `treemap_chart` (+`sunburst_chart`); add `area` param to `line_chart`.
- `chart_executor.py` — new executors + `TOOL_EXECUTORS` registrations; `area` handling in `execute_line_chart`.
- `report_generator.py` — route `key_metrics` to `self._key_metrics` → `Report.key_metrics`.
- `prompts/selection_system.txt` — selection guidance for the new tools.

**Frontend (`src/app/`)**
- `report/[id]/useReportLayout.ts` — `KeyMetric` type + optional `Report.key_metrics`.
- `components/` or `report/[id]/` — `KpiTiles` component.
- `report/[id]/charts/` — `GroupedBarChart`, `DualAxisChart`, `Treemap` (+`Sunburst`); `LineChart` area flag; `ChartContent.tsx` switch additions.
- `report/[id]/page.tsx` + `report/[id]/print/page.tsx` — render the KPI band.

**Tests (`tests/`)** — `tests/unit/test_chart_executor*.py` (new executor tests), `tests/unit/test_key_metrics.py`, a `report_generator` test for KPI routing.

---

# PHASE 1 — KPI stat tiles

### Task 1: KPI schema (backend + frontend type)

**Files:** Modify `src/api/schemas.py`; `src/app/report/[id]/useReportLayout.ts`. Test: `tests/unit/test_key_metrics.py` (created here, expanded in Task 2).

- [ ] **Step 1: Add the model + report field** in `schemas.py`

After `class ToolError` (or near `Report`), add:
```python
KpiFormat = Literal["number", "currency", "percent"]


class KeyMetric(BaseModel):
    label: str
    value: float
    format: KpiFormat = "number"
```
And add to `class Report` (after `data_quality`):
```python
    key_metrics: list[KeyMetric] = Field(default_factory=list)
```
(Default empty → existing stored reports stay valid, no migration.)

- [ ] **Step 2: Failing test** `tests/unit/test_key_metrics.py`
```python
from schemas import KeyMetric, Report

def test_keymetric_defaults_number():
    m = KeyMetric(label="Revenue", value=1240.5)
    assert m.format == "number"

def test_report_key_metrics_defaults_empty():
    r = Report(generated_at="t", summary="s", data_quality=[], charts=[])
    assert r.key_metrics == []
```
Run: `./venv/bin/python -m pytest tests/unit/test_key_metrics.py -q` → PASS.

- [ ] **Step 3: Frontend type** in `useReportLayout.ts` — add above `interface Report`:
```ts
export interface KeyMetric {
  label: string;
  value: number;
  format: 'number' | 'currency' | 'percent';
}
```
and add to `interface Report` (optional for back-compat with old reports + the marketing fixture):
```ts
  key_metrics?: KeyMetric[];
```

- [ ] **Step 4: Verify + commit**
```bash
./venv/bin/python -m pytest tests/unit/test_key_metrics.py -q && npx tsc --noEmit
git add src/api/schemas.py src/app/report/[id]/useReportLayout.ts tests/unit/test_key_metrics.py
git commit -m "feat(charts): KeyMetric schema + Report.key_metrics"
```

---

### Task 2: `key_metrics` tool + executor + report routing (TDD)

**Files:** Modify `src/api/chart_tools.py`, `src/api/chart_executor.py`, `src/api/report_generator.py`. Test: `tests/unit/test_key_metrics.py`, `tests/integration/test_pipeline_happy.py` (or a focused generator test).

The AI proposes metrics; the **executor computes the values from the dataframe** (never trusts AI numbers).

- [ ] **Step 1: Failing executor test** in `tests/unit/test_key_metrics.py`
```python
import pandas as pd
from chart_executor import execute_key_metrics
from schemas import KeyMetric, ToolError

def test_key_metrics_computes_values():
    df = pd.DataFrame({"revenue": [100, 200, 300], "region": ["W", "E", "W"]})
    res = execute_key_metrics(df, {"metrics": [
        {"label": "Total revenue", "column": "revenue", "agg": "sum", "format": "currency"},
        {"label": "Regions", "column": "region", "agg": "nunique", "format": "number"},
    ]})
    assert isinstance(res, list)
    by_label = {m.label: m for m in res}
    assert by_label["Total revenue"].value == 600.0
    assert by_label["Total revenue"].format == "currency"
    assert by_label["Regions"].value == 2.0

def test_key_metrics_drops_invalid_and_errors_when_empty():
    df = pd.DataFrame({"revenue": [1, 2]})
    # bad column dropped; if none valid -> ToolError
    res = execute_key_metrics(df, {"metrics": [{"label": "X", "column": "nope", "agg": "sum"}]})
    assert isinstance(res, ToolError)
```
Run → FAIL (no `execute_key_metrics`).

- [ ] **Step 2: Implement `execute_key_metrics`** in `chart_executor.py` (place near the other executors; import `KeyMetric` from schemas)
```python
def execute_key_metrics(df: pd.DataFrame, params: dict) -> list[KeyMetric] | ToolError:
    out: list[KeyMetric] = []
    for m in (params.get("metrics") or [])[:5]:
        col, agg = m.get("column"), m.get("agg")
        label, fmt = m.get("label") or col, m.get("format", "number")
        if col not in df.columns:
            continue
        s = df[col]
        try:
            if agg == "count":
                val = float(s.dropna().shape[0])
            elif agg == "nunique":
                val = float(s.dropna().nunique())
            else:  # sum/mean/median/min/max
                nums = pd.to_numeric(s, errors="coerce").dropna()
                if nums.empty:
                    continue
                val = float(getattr(nums, agg)())
        except Exception:
            continue
        if fmt not in ("number", "currency", "percent"):
            fmt = "number"
        out.append(KeyMetric(label=str(label)[:60], value=val, format=fmt))
    if not out:
        return _err("no valid metrics could be computed")
    return out
```
Do NOT add `execute_key_metrics` to `TOOL_EXECUTORS` (it returns a different type; it's routed specially in Step 4).

- [ ] **Step 3: Add the tool** in `chart_tools.py` `CHART_TOOLS` list:
```python
    _t(
        "key_metrics",
        "Headline numbers shown as a stat band at the top of the report. Call this ONCE with the "
        "3–5 most important figures a reader wants first (a total, an average, a key rate, a notable count). "
        "You choose the label/column/agg; the value is computed from the data.",
        {
            "metrics": {
                "type": "array",
                "minItems": 1, "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "Short human label, e.g. 'Total revenue'."},
                        "column": {"type": "string"},
                        "agg": {"type": "string", "enum": ["sum", "mean", "median", "min", "max", "count", "nunique"]},
                        "format": {"type": "string", "enum": ["number", "currency", "percent"]},
                    },
                    "required": ["label", "column", "agg"],
                    "additionalProperties": False,
                },
            },
        },
        ["metrics"],
    ),
```

- [ ] **Step 4: Route it** in `report_generator.py`
  - In `__init__`, add: `self._key_metrics: list = []`
  - Import at top: `from chart_executor import TOOL_EXECUTORS, execute_key_metrics`
  - In `_execute_tool_calls`, at the top of the `for block` loop body (after the `tool_use` type check), before the `TOOL_EXECUTORS.get`:
```python
            if block.name == "key_metrics":
                res = execute_key_metrics(self.df, block.input)
                if isinstance(res, ToolError):
                    errors.append({"id": block.id, "reason": res.reason})
                else:
                    self._key_metrics = res   # stored, NOT counted as a chart
                continue
```
  - In `build_report`, pass it to the `Report(...)`: add `key_metrics=self._key_metrics,`

- [ ] **Step 5: Generator routing test** in `tests/integration/test_pipeline_happy.py` (or a new focused test) — use the existing fake-Claude pattern to emit a `key_metrics` tool_use + ≥1 chart, assert the built `report.key_metrics` is populated AND `len(report.charts)` excludes the key_metrics call. (Mirror an existing pipeline test's FakeClaude setup.)

- [ ] **Step 6: Verify + commit**
```bash
./venv/bin/python -m pytest tests/unit/test_key_metrics.py tests/integration/test_pipeline_happy.py -q
git add src/api/chart_tools.py src/api/chart_executor.py src/api/report_generator.py tests/
git commit -m "feat(charts): key_metrics tool + executor (computes values) + report routing"
```

---

### Task 3: KPI prompt line + KpiTiles frontend (+ PDF)

**Files:** Modify `prompts/selection_system.txt`; create `src/app/report/[id]/KpiTiles.tsx`; modify `report/[id]/page.tsx`, `report/[id]/print/page.tsx`.

- [ ] **Step 1: Prompt** — add to `selection_system.txt` (after the "Rules:" intro, as the first rule):
```
- Always call key_metrics ONCE, first, with the 3–5 headline numbers a reader most wants (a total, an average, a key rate, a notable count). These render as a stat band above the charts.
```

- [ ] **Step 2: `KpiTiles.tsx`** (light, mono numerals; format via existing `lib/format.ts`)
```tsx
'use client';
import { getFormatter } from '../../lib/format';
import type { KeyMetric } from './useReportLayout';

const FMT: Record<string, string> = { number: 'number', currency: 'currency', percent: 'percentage' };

export default function KpiTiles({ metrics }: { metrics?: KeyMetric[] }) {
  if (!metrics || metrics.length === 0) return null;
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-px bg-line border border-line rounded-2xl overflow-hidden mb-8">
      {metrics.map((m, i) => {
        const fmt = getFormatter(FMT[m.format] ?? 'number');
        return (
          <div key={i} className="bg-surface p-5">
            <div className="font-mono text-2xl font-semibold text-ink tracking-tight">{fmt(m.value)}</div>
            <div className="text-xs text-ink-3 mt-1">{m.label}</div>
          </div>
        );
      })}
    </div>
  );
}
```
(Confirm `getFormatter` accepts those keys; it's the same formatter the charts use for `y_display_type` = count/currency/percentage/number.)

- [ ] **Step 3: Render it** — in `report/[id]/page.tsx`, import `KpiTiles` and place it just inside the report container, **above** `<Toolbar>` / `<ReportSummary>`:
```tsx
        <KpiTiles metrics={report.key_metrics} />
```
And in `report/[id]/print/page.tsx`, render `<KpiTiles metrics={report.key_metrics} />` at the top of the print document (it inherits `.theme-light`).

- [ ] **Step 4: Verify + commit**
```bash
npx tsc --noEmit
git add prompts/selection_system.txt "src/app/report/[id]/KpiTiles.tsx" "src/app/report/[id]/page.tsx" "src/app/report/[id]/print/page.tsx"
git commit -m "feat(charts): KPI stat-tile band (prompt + KpiTiles + report/PDF render)"
```

---

# PHASE 2 — Bar & trend variants

### Task 4: Schema + `grouped_bar_chart` (TDD)

**Files:** `schemas.py`, `chart_tools.py`, `chart_executor.py`. Test: `tests/unit/test_grouped_bar.py`.

- [ ] **Step 1: Schema** — in `schemas.py`: extend `ChartKind` to include `"grouped_bar", "dual_axis"`; add optional fields to `ChartSpec`:
```python
    stacked: bool = False
    y_label_secondary: Optional[str] = None
```

- [ ] **Step 2: Failing test** `tests/unit/test_grouped_bar.py`
```python
import pandas as pd
from chart_executor import execute_grouped_bar_chart
from schemas import ChartSpec

def test_grouped_bar_multi_series():
    df = pd.DataFrame({
        "region": ["W", "W", "E", "E"],
        "product": ["A", "B", "A", "B"],
        "rev": [10, 20, 30, 40],
    })
    res = execute_grouped_bar_chart(df, {
        "category_col": "region", "breakdown_col": "product",
        "value_col": "rev", "agg": "sum", "mode": "grouped",
        "title": "Rev by region × product", "intent": "x",
    })
    assert isinstance(res, ChartSpec)
    assert res.kind == "grouped_bar"
    assert res.stacked is False
    assert len(res.series) == 2          # one per product
    assert res.x == ["W", "E"] or set(res.x) == {"W", "E"}
```
Run → FAIL.

- [ ] **Step 3: Implement `execute_grouped_bar_chart`** in `chart_executor.py` — group by `category_col`, pivot on `breakdown_col`, aggregate `value_col` with `agg`. Build `x` = category values, `series` = `[{"name": <breakdown value>, "data": [...aligned to x...]}]`. Set `kind="grouped_bar"`, `stacked = (mode == "stacked")`, `x_label`/`y_label`, `source_columns`, `data_point_count`. Guards: `category_col`/`breakdown_col`/`value_col` exist; breakdown cardinality ≤ 6 (else `_err` suggesting a plain `aggregation_bar_chart`); category cardinality ≤ `MAX_CATEGORIES`. Follow the existing executor style (`_err`, `_infer_display_type`).

- [ ] **Step 4: Tool** in `chart_tools.py`:
```python
    _t(
        "grouped_bar_chart",
        "Bar chart of a value aggregated by a category AND split by a second (sub)category. "
        "mode='grouped' compares side by side; mode='stacked' shows composition. Keep the breakdown to ≤6 values.",
        {
            "category_col": {"type": "string"},
            "breakdown_col": {"type": "string", "description": "Second categorical to split each bar by (≤6 values)."},
            "value_col": {"type": "string"},
            "agg": {"type": "string", "enum": ["sum", "mean", "median", "min", "max"]},
            "mode": {"type": "string", "enum": ["grouped", "stacked"]},
            **_TITLE_INTENT,
        },
        ["category_col", "breakdown_col", "value_col", "agg", "mode", "title", "intent"],
    ),
```
Register `"grouped_bar_chart": execute_grouped_bar_chart` in `TOOL_EXECUTORS`.

- [ ] **Step 5: Verify + commit**
```bash
./venv/bin/python -m pytest tests/unit/test_grouped_bar.py -q
git add src/api/schemas.py src/api/chart_tools.py src/api/chart_executor.py tests/unit/test_grouped_bar.py
git commit -m "feat(charts): grouped/stacked bar tool + executor + kind"
```

---

### Task 5: `dual_axis_chart` + `line area` flag (TDD)

**Files:** `chart_tools.py`, `chart_executor.py`. Test: `tests/unit/test_dual_axis.py`, extend the line test.

- [ ] **Step 1: Failing test** `tests/unit/test_dual_axis.py`
```python
import pandas as pd
from chart_executor import execute_dual_axis_chart
from schemas import ChartSpec

def test_dual_axis_two_series():
    df = pd.DataFrame({"month": ["Jan", "Jan", "Feb", "Feb"], "rev": [10, 20, 30, 40], "rate": [0.1, 0.2, 0.3, 0.4]})
    res = execute_dual_axis_chart(df, {
        "x_col": "month", "bar_value_col": "rev", "line_value_col": "rate",
        "bar_agg": "sum", "line_agg": "mean", "title": "Rev & rate", "intent": "x",
    })
    assert isinstance(res, ChartSpec)
    assert res.kind == "dual_axis"
    assert res.y_label_secondary is not None
    assert len(res.series) == 2
    # series carry an axis/type hint for the frontend
    kinds = {s.get("type") for s in res.series}
    assert kinds == {"bar", "line"}
```
Run → FAIL.

- [ ] **Step 2: Implement `execute_dual_axis_chart`** — group by `x_col`; aggregate `bar_value_col` (with `bar_agg`) and `line_value_col` (with `line_agg`); `x` = x values; `series = [{"name": bar_value_col, "type": "bar", "yAxisIndex": 0, "data": [...]}, {"name": line_value_col, "type": "line", "yAxisIndex": 1, "data": [...]}]`; `kind="dual_axis"`, `y_label = bar_value_col`, `y_label_secondary = line_value_col`. Guards: columns exist + numeric value cols.

- [ ] **Step 3: Tool** `dual_axis_chart`:
```python
    _t(
        "dual_axis_chart",
        "Combo chart: a bar metric and a line metric on two y-axes, sharing an x category/time. "
        "Use when two metrics on different scales are worth seeing together (e.g. revenue + conversion rate).",
        {
            "x_col": {"type": "string"},
            "bar_value_col": {"type": "string"},
            "line_value_col": {"type": "string"},
            "bar_agg": {"type": "string", "enum": ["sum", "mean", "median", "min", "max", "count"]},
            "line_agg": {"type": "string", "enum": ["sum", "mean", "median", "min", "max", "count"]},
            **_TITLE_INTENT,
        },
        ["x_col", "bar_value_col", "line_value_col", "bar_agg", "line_agg", "title", "intent"],
    ),
```
Register `"dual_axis_chart": execute_dual_axis_chart`.

- [ ] **Step 4: Area flag on line** — in `chart_tools.py` `line_chart`, add property `"area": {"type": "boolean", "description": "Fill under the line (area chart). Multi-series + area = stacked area."}` (leave it out of `required`). In `execute_line_chart`, read `params.get("area", False)` and set it on the returned `ChartSpec` (add an optional `area: bool = False` field to `ChartSpec` in `schemas.py`). Add a test asserting `execute_line_chart(..., {"area": True})` returns a spec with `area is True`.

- [ ] **Step 5: Verify + commit**
```bash
./venv/bin/python -m pytest tests/unit/test_dual_axis.py tests/unit/test_chart_executor*.py -q
git add src/api/schemas.py src/api/chart_tools.py src/api/chart_executor.py tests/
git commit -m "feat(charts): dual-axis combo tool + executor; line area flag"
```

---

### Task 6: Frontend — grouped bar, dual-axis, line area + prompt guidance

**Files:** Create `charts/GroupedBarChart.tsx`, `charts/DualAxisChart.tsx`; modify `charts/LineChart.tsx`, `charts/ChartContent.tsx`, `prompts/selection_system.txt`.

- [ ] **Step 1: Components** — build `GroupedBarChart` and `DualAxisChart` using `chartTheme` helpers (`chartBase`, `catAxis`, `valAxis`, `CHART_PALETTE`, `monoFamily`):
  - **GroupedBarChart:** map `spec.series` → ECharts bar series (`color` per index from `CHART_PALETTE`, `borderRadius:[4,4,0,0]`); if `spec.stacked`, set every series `stack: 'total'`; x from `spec.x`; legend (mono).
  - **DualAxisChart:** two `yAxis` (left `valAxis()`, right `valAxis({ position: 'right' })`); map `spec.series` honoring each series' `type` ('bar'/'line') and `yAxisIndex`; bar teal, line ochre; legend; tooltip `trigger:'axis'`.
- [ ] **Step 2: LineChart area** — in `LineChart.tsx`, when `spec.area`, add `areaStyle: { color: tealAreaGradient() }` to the (single-series) line; for multi-series + area, set `stack: 'total'` on each series.
- [ ] **Step 3: ChartContent switch** — add `case 'grouped_bar': return <GroupedBarChart spec={spec} />;` and `case 'dual_axis': return <DualAxisChart spec={spec} />;` (dynamic imports like the others).
- [ ] **Step 4: Prompt guidance** — add to `selection_system.txt`: grouped/stacked bar for a category split by a sub-category; dual-axis for two different-scale metrics; line `area:true` for cumulative/volume trends.
- [ ] **Step 5: Verify + commit**
```bash
npx tsc --noEmit
git add "src/app/report/[id]/charts/" prompts/selection_system.txt
git commit -m "feat(charts): grouped-bar, dual-axis, area frontend + selection guidance"
```

---

# PHASE 3 — Part-to-whole

### Task 7: Schema + `treemap_chart` (TDD) (+ optional sunburst)

**Files:** `schemas.py`, `chart_tools.py`, `chart_executor.py`. Test: `tests/unit/test_treemap.py`.

- [ ] **Step 1: Schema** — `ChartKind` += `"treemap"` (and `"sunburst"` if building it); `ChartSpec` += `nodes: Optional[list] = None` (hierarchical node tree).
- [ ] **Step 2: Failing test** `tests/unit/test_treemap.py`
```python
import pandas as pd
from chart_executor import execute_treemap_chart
from schemas import ChartSpec

def test_treemap_flat():
    df = pd.DataFrame({"cat": ["A", "A", "B", "C"], "rev": [10, 5, 20, 8]})
    res = execute_treemap_chart(df, {"category_col": "cat", "value_col": "rev", "agg": "sum", "title": "t", "intent": "x"})
    assert isinstance(res, ChartSpec)
    assert res.kind == "treemap"
    names = {n["name"] for n in res.nodes}
    assert names == {"A", "B", "C"}
    assert next(n for n in res.nodes if n["name"] == "A")["value"] == 15

def test_treemap_hierarchy():
    df = pd.DataFrame({"region": ["W", "W", "E"], "product": ["A", "B", "A"], "rev": [10, 20, 30]})
    res = execute_treemap_chart(df, {"category_col": "region", "subcategory_col": "product", "value_col": "rev", "agg": "sum", "title": "t", "intent": "x"})
    w = next(n for n in res.nodes if n["name"] == "W")
    assert {c["name"] for c in w["children"]} == {"A", "B"}
```
Run → FAIL.

- [ ] **Step 3: Implement `execute_treemap_chart`** — aggregate `value_col` by `category_col` (and `subcategory_col` if given) with `agg`; build `nodes` = `[{name, value}]` flat, or `[{name, value, children:[{name, value}]}]` when a subcategory is given; `kind="treemap"`; guard column existence + non-negative numeric values (treemaps need ≥0). Register `"treemap_chart": execute_treemap_chart`.
- [ ] **Step 4: Tool** `treemap_chart`:
```python
    _t(
        "treemap_chart",
        "Treemap of a category's share of a total (optionally a 2-level hierarchy). "
        "Prefer over a pie when there are many categories (>8) or a sub-category breakdown.",
        {
            "category_col": {"type": "string"},
            "subcategory_col": {"type": "string", "description": "Optional second level."},
            "value_col": {"type": "string"},
            "agg": {"type": "string", "enum": ["sum", "mean", "count"]},
            **_TITLE_INTENT,
        },
        ["category_col", "value_col", "agg", "title", "intent"],
    ),
```
- [ ] **Step 5 (optional sunburst):** only if cheap — add `sunburst_chart` tool reusing the same node builder (or a `style` param), `kind="sunburst"`. Skip if it adds risk; note the skip in the commit.
- [ ] **Step 6: Verify + commit**
```bash
./venv/bin/python -m pytest tests/unit/test_treemap.py -q
git add src/api/schemas.py src/api/chart_tools.py src/api/chart_executor.py tests/unit/test_treemap.py
git commit -m "feat(charts): treemap tool + executor + kind"
```

---

### Task 8: Frontend — Treemap (+ optional Sunburst) + prompt guidance

**Files:** Create `charts/Treemap.tsx` (+ `charts/Sunburst.tsx` if built); modify `charts/ChartContent.tsx`, `prompts/selection_system.txt`.

- [ ] **Step 1: Treemap component** — ECharts `series:[{ type:'treemap', data: spec.nodes, … }]` with `CHART_PALETTE`, `monoFamily()` labels, `breadcrumb:{show:false}`, white tile borders (`itemStyle.borderColor:'#fff'`), label `color` per the theme. Inherit `chartBase()` text/tooltip.
- [ ] **Step 2: ChartContent** — `case 'treemap': return <Treemap spec={spec} />;` (+ `case 'sunburst'` if built).
- [ ] **Step 3: Prompt** — add: a category's composition with many slices or a 2-level hierarchy → `treemap_chart` (prefer over pie when >8 categories).
- [ ] **Step 4: Verify + commit**
```bash
npx tsc --noEmit
git add "src/app/report/[id]/charts/" prompts/selection_system.txt
git commit -m "feat(charts): treemap frontend + selection guidance"
```

---

### Task 9: Build + verify + finish

**Files:** none (verification only)

- [ ] **Step 1: Full backend suite** — `./venv/bin/python -m pytest -q` (all green, incl. new executor + KPI + diversity/fallback tests).
- [ ] **Step 2: Frontend build** — `rm -rf .next && npm run build` (exit 0).
- [ ] **Step 3: Visual QA** — generate a report (or use a fixture) and confirm: the KPI band renders (app + PDF); a grouped/stacked bar, dual-axis, area line, and treemap each render with the light theme; existing charts unaffected.
- [ ] **Step 4: Finish** — use **superpowers:finishing-a-development-branch** to merge `chart-expansion` → `main` (backend + frontend deploy; production requires explicit user authorization). Backend changes (tools/executors/prompt) need a Cloud Run deploy; frontend auto-deploys via Vercel.

---

## Self-Review

**Spec coverage:** §1 KPI tiles → Tasks 1–3 (schema, tool+executor+routing, prompt+frontend+PDF); §2 bar/trend → Tasks 4 (grouped/stacked), 5 (dual-axis + area), 6 (frontend+prompt); §3 part-to-whole → Tasks 7–8 (treemap + optional sunburst); §4 schema additions → Tasks 1/4/5/7; §5 selection guidance → Tasks 3/6/8; §6 structure/theme/credits → Task 3 (band placement) + 6/8 (chartTheme) + unchanged credits/layout; §7 phasing → Phase 1/2/3; §9 verification → Task 9. Geo correctly absent. No gaps.

**Placeholder scan:** Complete code for the contract-critical bits (KeyMetric schema, `execute_key_metrics`, the `key_metrics` routing in `report_generator`, tool schemas, tests). Executor *bodies* for grouped_bar/dual_axis/treemap are specified by behavior + the test that pins them + the existing-executor style to follow (the tests are the exact contract) — not vague. Frontend components are recipes referencing `chartTheme` + the existing chart components as the pattern. No TBD/"handle edge cases".

**Type consistency:** `KeyMetric{label,value,format}` consistent backend (`schemas.py`) ↔ frontend (`useReportLayout.ts`) ↔ `KpiTiles`. New `ChartKind`s (`grouped_bar`/`dual_axis`/`treemap`/`sunburst?`) match the `ChartContent` switch cases. `ChartSpec` optional fields (`stacked`, `y_label_secondary`, `area`, `nodes`) defined in the phase that introduces them and consumed by that phase's component. `execute_key_metrics` returns `list[KeyMetric] | ToolError` and is routed by name (not via `TOOL_EXECUTORS`) — consistent with the `report_generator` change. Tool param names match executor `params.get(...)` keys.

**Risk note:** `key_metrics` is deliberately NOT in `TOOL_EXECUTORS` (it returns metrics, not a `ChartSpec`); it's special-cased by tool name in `_execute_tool_calls` so it never counts toward `MAX_CHARTS` or the charts/layout/narrative. `Report.key_metrics` defaults to `[]` and the frontend type is optional, so existing stored reports + the marketing fixture stay valid (no migration).
