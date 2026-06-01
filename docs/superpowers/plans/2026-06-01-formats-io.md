# ChartSage Export/Import (Formats I/O) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> **Subagents must NOT run `git checkout`, `git switch`, `git reset`, or `git stash`.** Stay on branch `formats-io`; commit only the files each task lists.

**Goal:** Add four export formats (PPTX, XLSX, PNG-zip, Markdown/HTML) and two import-UX upgrades (Excel sheet picker + column deselector).

**Architecture:** *Import* stays frontend-led — the uploader parses the file, lets the user pick a sheet + drop columns, and emits a CSV, so the backend is untouched. *Export* adds one shared Playwright "screenshot each chart → PNG" step (reusing the PDF browser) feeding format builders in a new `report_export.py`, exposed as per-format endpoints mirroring `export.pdf`, surfaced as a toolbar Export menu. Exports are free.

**Tech Stack:** Next.js (`xlsx`, `papaparse` already installed), FastAPI, Playwright (installed), `python-pptx` (new), `openpyxl`/`pandas` (installed).

**Spec:** `docs/superpowers/specs/2026-06-01-formats-io-design.md`
**Branch:** `formats-io` (carries the spec).

---

## File Structure

**Phase A (frontend only)**
- `src/app/app/page.tsx` — uploader: sheet picker + column deselector + emit-CSV.

**Phase B**
- `src/app/report/[id]/print/page.tsx` — render **all** charts (main + sidebar) + add `data-chart-export-id` per chart.
- `src/api/pdf_export.py` — add `render_chart_images(session_id)`.
- `src/api/report_export.py` — **new**: `build_pptx`, `build_xlsx`, `build_png_zip`, `build_markdown`, `build_html`.
- `src/api/main.py` — new endpoints `export.pptx` / `export.xlsx` / `export.zip` / `export.md` / `export.html`.
- `src/api/requirements.txt` — add `python-pptx`.
- `src/app/report/[id]/Toolbar.tsx` — Export ▾ menu.
- Tests: `tests/unit/test_report_export.py`.

---

# PHASE A — Import UX

### Task 1: Sheet picker + column deselector + emit-CSV

**Files:** Modify `src/app/app/page.tsx`. (`xlsx` and `papaparse` are already deps.)

The uploader currently sends the raw file and only previews CSVs. Rework it to: parse any file in-browser → choose sheet (xlsx) → preview with per-column checkboxes → on Generate, emit the selection as a CSV and send that. **Preserve** the 10 MB guard, the extension guard, and the entire `generate()` response handling (403 `ANON_LIMIT_REACHED`, 402 `OUT_OF_CREDITS`→modal, 503 `BUSY`, success→`router.push`, `refetch`).

- [ ] **Step 1: Parse on drop** — in `onDrop`, after the size/extension guards, read the file and populate parse state. Add `import * as XLSX from 'xlsx';` (keep `papaparse` for CSV if preferred, or use XLSX for both). New state:
```tsx
const [sheetNames, setSheetNames] = useState<string[]>([]);
const [sheet, setSheet] = useState<string>('');       // selected sheet
const [columns, setColumns] = useState<string[]>([]); // all columns in the sheet
const [excluded, setExcluded] = useState<Set<string>>(new Set());
const [rows, setRows] = useState<Record<string, any>[]>([]); // full parsed rows of the selected sheet
```
Parse: read the file into an `ArrayBuffer`; `const wb = XLSX.read(buf, { type: 'array' });` → `wb.SheetNames`. For CSV, `XLSX.read` also works (single sheet). Select the first sheet by default and call a `loadSheet(wb, name)` helper that does `XLSX.utils.sheet_to_json(wb.Sheets[name], { defval: '' })` → set `rows`, derive `columns` from the first row's keys, reset `excluded` to empty.

- [ ] **Step 2: Sheet picker UI** — when `sheetNames.length > 1`, render a labelled `<select value={sheet} onChange={e => loadSheet(wb, e.target.value)}>` of the sheet names. (Keep the parsed `wb` in a ref so re-selecting a sheet doesn't re-read the file.)

- [ ] **Step 3: Column deselector** — in the preview table header, render a checkbox per column: checked = included; toggling adds/removes from `excluded`. Show the first ~10 `rows` for preview. Selected columns = `columns.filter(c => !excluded.has(c))`. Guard: disable Generate (or block) if every column is excluded.

- [ ] **Step 4: Emit CSV on Generate** — replace `fd.append('file', file)` with a CSV built from the selection:
```tsx
const keep = columns.filter((c) => !excluded.has(c));
const csv = Papa.unparse(rows.map((r) => Object.fromEntries(keep.map((c) => [c, r[c]]))));
const blob = new File([csv], (file?.name?.replace(/\.(xlsx|csv)$/i, '') || 'data') + '.csv', { type: 'text/csv' });
const fd = new FormData();
fd.append('file', blob);
```
Everything else in `generate()` stays identical (the backend receives a `.csv`).

- [ ] **Step 5: Verify + commit**
```bash
npx tsc --noEmit
git add "src/app/app/page.tsx"
git commit -m "feat(import): Excel sheet picker + column deselector → emit CSV"
```
Manual check (later, in Task 6 QA): a multi-sheet .xlsx shows the picker; deselecting a column drops it; a plain CSV still works.

---

# PHASE B — Export

### Task 2: Print route renders all charts + screenshot targets

**Files:** Modify `src/app/report/[id]/print/page.tsx`.

Today the print page maps `mainCharts` only — so the PDF omits sidebar charts. Render **all** charts (main then sidebar) and tag each for screenshotting. This also makes the PDF complete (an intended improvement).

- [ ] **Step 1: Render all charts + export id** — replace the `mainCharts.map(...)` block. Compute the full ordered list (main charts followed by sidebar charts — both already derived in the file, or compute from `report.layout` ordered by position then order) and map it, wrapping each in a div carrying a stable per-chart id:
```tsx
{allCharts.map((c, i) => (
  <div key={c.chart_id} data-chart-export-id={c.chart_id} className="avoid-break mb-6">
    <ChartCard chartId={c.chart_id} index={i + 1} spec={c.spec} caption={c.caption} printMode />
  </div>
))}
```
(`allCharts` = main charts concatenated with sidebar charts, in layout order. Keep the existing `data-charts-ready` flag logic that signals when all charts have mounted.)

- [ ] **Step 2: Verify + commit**
```bash
npx tsc --noEmit
git add "src/app/report/[id]/print/page.tsx"
git commit -m "feat(export): print route renders all charts with per-chart export targets"
```

---

### Task 3: `render_chart_images` (shared Playwright render)

**Files:** Modify `src/api/pdf_export.py`.

- [ ] **Step 1: Add the function** — reuse `_ensure_browser`; navigate to the print route; wait for `data-charts-ready`; screenshot each `[data-chart-export-id]` element.
```python
async def render_chart_images(session_id: str) -> list[dict]:
    """Return [{"chart_id": str, "png": bytes}, ...] for every chart, in DOM order."""
    browser = await _ensure_browser()
    page = await browser.new_page(viewport={"width": 1240, "height": 1754}, device_scale_factor=2)
    try:
        await page.goto(f"{_FRONTEND_BASE}/report/{session_id}/print", wait_until="networkidle", timeout=30_000)
        try:
            await page.wait_for_selector('body[data-charts-ready="true"]', timeout=_CHARTS_READY_TIMEOUT_MS)
        except Exception:
            logging.warning("[IMG] charts-ready flag not seen — proceeding")
        out: list[dict] = []
        locators = page.locator('[data-chart-export-id]')
        count = await locators.count()
        for i in range(count):
            el = locators.nth(i)
            cid = await el.get_attribute('data-chart-export-id')
            png = await el.screenshot(type='png')
            out.append({"chart_id": cid or f"chart-{i}", "png": png})
        return out
    finally:
        await page.close()
```

- [ ] **Step 2: Verify + commit**
```bash
./venv/bin/python -c "import pdf_export; print(hasattr(pdf_export,'render_chart_images'))"   # True
git add src/api/pdf_export.py
git commit -m "feat(export): render_chart_images — screenshot each chart to PNG"
```
(Live rendering is exercised by the opt-in integration test in Task 6; here we just confirm the symbol + import.)

---

### Task 4: Format builders (`report_export.py`, TDD)

**Files:** Create `src/api/report_export.py`; `tests/unit/test_report_export.py`. Add `python-pptx` to `requirements.txt`.

All builders take a validated `Report` (from `schemas`) + the image list (for image formats) or source CSV bytes (XLSX). They are **pure** (no I/O, no Playwright) → fully unit-testable with fake images.

- [ ] **Step 1: Add the dep** — append `python-pptx==0.6.23` to `src/api/requirements.txt`; `./venv/bin/pip install python-pptx==0.6.23`.

- [ ] **Step 2: Failing tests** `tests/unit/test_report_export.py`
```python
import io, zipfile
from schemas import Report, ChartWithCaption, ChartSpec, ChartLayoutEntry, KeyMetric
import report_export as rx

def _report():
    spec = ChartSpec(kind="bar", title="By region", intent="i", x=["W","E"], y=[2,1], source_columns=["region"], data_point_count=2)
    return Report(generated_at="2024-01-01T00:00:00", summary="Sales summary.", data_quality=["1% blanks"],
                  charts=[ChartWithCaption(chart_id="c1", spec=spec, caption="West leads.")],
                  layout=[ChartLayoutEntry(chart_id="c1", position="main", order=0)],
                  key_metrics=[KeyMetric(label="Total", value=1234.0, format="currency")])

IMAGES = [{"chart_id": "c1", "png": b"\x89PNG\r\n\x1a\n" + b"0"*64}]  # not a real PNG; builders just embed bytes

def test_pptx_opens_with_slides():
    from pptx import Presentation
    data = rx.build_pptx(_report(), IMAGES)
    prs = Presentation(io.BytesIO(data))
    assert len(prs.slides) >= 1 + 1 + len(IMAGES)   # title + summary/kpi + one per chart (exact count per impl)

def test_xlsx_has_data_and_summary_sheets():
    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(rx.build_xlsx(_report(), b"region,sales\nW,2\nE,1\n")))
    assert set(wb.sheetnames) >= {"Data", "Summary"}

def test_png_zip_has_one_entry_per_chart():
    z = zipfile.ZipFile(io.BytesIO(rx.build_png_zip(IMAGES)))
    assert len(z.namelist()) == len(IMAGES)

def test_markdown_contains_summary_and_kpis():
    md = rx.build_markdown(_report(), IMAGES)
    assert "Sales summary." in md and "Total" in md and "data:image/png;base64," in md
```
Run → FAIL (module missing).

- [ ] **Step 3: Implement `report_export.py`** to satisfy the tests:
  - `build_pptx(report, images) -> bytes`: `python-pptx` `Presentation()`; slide 1 = title (`report` has no title field — use the first sentence of `summary` or "ChartSage Report" + the date); slide 2 = summary text; slide 3 = KPI tiles (`report.key_metrics` as text lines); then one slide per image (`slide.shapes.add_picture(io.BytesIO(img["png"]), ...)` + the matching chart's caption). Save to `io.BytesIO`, return bytes.
  - `build_xlsx(report, source_csv) -> bytes`: `pandas.read_csv(io.BytesIO(source_csv))` → write to a "Data" sheet via `pandas.ExcelWriter(engine="openpyxl")`; write a "Summary" sheet from a small DataFrame of `[(m.label, m.value) for m in report.key_metrics]` plus the summary text in a cell. Return bytes.
  - `build_png_zip(images) -> bytes`: zip each `img["png"]` as `{index}-{safe_title_or_chart_id}.png`.
  - `build_markdown(report, images) -> str`: `# Report` + date, the summary paragraphs, a `**KPIs**` list, then per chart: the caption + an embedded `![title](data:image/png;base64,<b64>)`.
  - `build_html(report, images) -> bytes`: same content as Markdown but minimal standalone HTML (utf-8), images as base64 `<img>`; return encoded bytes.
  - Map images to captions by `chart_id` (fall back to order).

- [ ] **Step 4: Run tests → PASS; commit**
```bash
./venv/bin/python -m pytest tests/unit/test_report_export.py -q
git add src/api/report_export.py src/api/requirements.txt tests/unit/test_report_export.py
git commit -m "feat(export): report_export builders (pptx/xlsx/png-zip/markdown/html) + tests"
```

---

### Task 5: Export endpoints + Toolbar menu

**Files:** Modify `src/api/main.py`, `src/app/report/[id]/Toolbar.tsx`.

- [ ] **Step 1: Endpoints** — add five endpoints mirroring `export_pdf` (`main.py:567`). Each: `db.get_report` (404 if missing) → for image formats `await pdf_export.render_chart_images(session_id)`; for XLSX read the source CSV (`csv_key = row["csv_storage_key"]; csv_bytes = storage.download_by_key(csv_key)`); call the builder; return `StreamingResponse` with the right `media_type` + `Content-Disposition` filename `chartsage-{id[:8]}.{ext}`; emit `posthog.capture(identity.distinct_id, "report_exported", {"reportId": session_id, "format": "<fmt>"})`. Validate the `Report` with `Report.model_validate(row["report_json"])` (or the existing helper) before passing to builders.
  - Media types: pptx `application/vnd.openxmlformats-officedocument.presentationml.presentation`; xlsx `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`; zip `application/zip`; md `text/markdown`; html `text/html`.
  - Endpoints need `storage: SupabaseStorage = Depends(get_storage)` for the XLSX one (mirror how generate-more gets storage).
  - No credit gate (consistent with `export.pdf`).

- [ ] **Step 2: Toolbar Export menu** — replace the single "Export PDF" `<button>` with an **Export ▾** dropdown. Generalize the existing `handleExportPdf` into `handleExport(ext, mime)` that `apiFetch(`/report/${sessionId}/export.${ext}`)` → blob → download as `chartsage-{id8}.{ext}` (keep the existing blob-download logic + the `exporting` state). Menu items: PDF (`pdf`), PowerPoint (`pptx`), Excel (`xlsx`), Images (`zip`), Markdown (`md`). Use a simple details/summary or a small popover; keep it on-theme (semantic classes).

- [ ] **Step 3: Verify + commit**
```bash
./venv/bin/python -m pytest -q && npx tsc --noEmit
git add src/api/main.py "src/app/report/[id]/Toolbar.tsx"
git commit -m "feat(export): per-format endpoints + Toolbar Export menu"
```

---

### Task 6: Build + verify + finish

**Files:** none (verification).

- [ ] **Step 1:** `./venv/bin/python -m pytest -q` (green incl. the new builder tests) and `rm -rf .next && npm run build` (exit 0).
- [ ] **Step 2: Import QA** (`npm run dev`): a multi-sheet `.xlsx` shows the sheet picker; switching sheets re-previews; deselecting a column excludes it from the generated report; a plain CSV still works; the all-columns-excluded guard holds.
- [ ] **Step 3: Export QA** (needs the backend running + a real report): each Export-menu item downloads an openable file — PPTX opens in PowerPoint/Keynote with a slide per chart, XLSX has Data + Summary, the zip has one PNG per chart, Markdown renders with embedded images; the PDF still works and now includes all charts.
- [ ] **Step 4: Finish** — use **superpowers:finishing-a-development-branch** to merge `formats-io` → `main`. Backend changes (new endpoints, `python-pptx`, `render_chart_images`) need a **Cloud Run deploy** (and `python-pptx` is in `requirements.txt` so the image rebuild installs it); frontend auto-deploys via Vercel. Production deploy requires explicit user authorization.

---

## Self-Review

**Spec coverage:** Import sheet picker + column deselector + emit-CSV → Task 1; export-includes-all-charts + screenshot targets → Task 2; shared chart→PNG render → Task 3; PPTX/XLSX/PNG-zip/Markdown/HTML builders → Task 4; endpoints (free, mirror export.pdf) + Export menu → Task 5; `python-pptx` dep → Task 4; verification → Task 6. No Google Sheets / new import formats / credit gating / schema changes — all respected. No gaps.

**Placeholder scan:** Complete code for the contract-critical pieces — the uploader emit-CSV snippet, `render_chart_images`, the builder test contracts (which pin each builder), the endpoint shape (mirrors the shown `export_pdf`). Builder *bodies* are specified by behavior + the failing tests that pin them + the libraries to use — not vague. Frontend Toolbar/uploader are recipes against the existing components. No TBD/"handle edge cases".

**Type/name consistency:** `render_chart_images` returns `[{chart_id, png}]`, consumed by `build_pptx`/`build_png_zip`/`build_markdown`/`build_html`; `build_xlsx` takes source CSV bytes (from `storage.download_by_key`, the method added in the earlier 404 fix). Endpoints validate `report_json` → `Report` (the same model the builders type against, with `key_metrics` from the chart-types work). `data-chart-export-id` set in Task 2 matches the selector in Task 3. The Toolbar `handleExport(ext, mime)` covers the five endpoint extensions from Task 5.

**Risk note:** Task 2 changes the PDF to include sidebar charts (all charts) — an intended improvement, not a regression; called out so it's a conscious change. `render_chart_images` opens one page and screenshots N elements on the long-lived browser (same resource model as the PDF export); device_scale_factor=2 for crisp images.
