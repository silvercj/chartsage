# ChartSage Export/Import (Formats I/O) — Design Spec

**Status:** Approved (design), pending spec review
**Date:** 2026-06-01
**Type:** Backend + frontend feature

**Goal:** Let users get reports *out* in more formats (PPTX, XLSX, PNG, Markdown/HTML) and get data *in* more flexibly — choosing the sheet in a multi-tab Excel file and deselecting columns before analysis.

**Approved scope:**
- **Export:** PPTX slide deck · XLSX (data + summary) · PNG chart images (zip) · Markdown/HTML report.
- **Import UX:** Excel multi-sheet picker + per-column deselector. *No Google Sheets, no new file formats (CSV + Excel only).*

**Context:** Today import = CSV/`.xlsx` (frontend dropzone → raw file → backend `pd.read_csv`/`pd.read_excel` → stored CSV). Export = PDF only: `pdf_export.render_report_pdf(session_id)` keeps a long-lived Chromium, navigates to `/report/{id}/print`, waits for `body[data-charts-ready="true"]`, captures `page.pdf()`. Exports are **free** (no credit gate) — keep it that way. Deps present: `playwright`, `openpyxl`, `pandas` (backend); `xlsx`, `papaparse` (frontend). New dep: `python-pptx`.

---

## Phase A — Import UX (frontend-led; backend unchanged)

The uploader (`src/app/app/page.tsx`) gains two controls; the backend keeps receiving a CSV, so nothing server-side changes.

1. **Excel sheet picker** — on file select, parse the workbook with the already-installed `xlsx` lib to list sheet names. If >1 sheet, show a picker (select/radio); default to the first. CSV files have a single implicit sheet (no picker).
2. **Column deselector** — the existing preview table gains a checkbox per column header; unchecking excludes that column. (Guard: at least one column must stay selected.)
3. **Emit CSV → send** — on Generate, the frontend builds the final dataset (chosen sheet, deselected columns dropped) as a CSV string (via `xlsx`'s `sheet_to_csv` / `papaparse` unparse) and sends *that* as the upload (a `.csv` File/Blob) to `/generate-report`. **The backend is unchanged** — it already reads CSV, and the stored CSV reflects the user's selection, so generate-more stays consistent.
   - The live preview (first ~10 rows) reflects the chosen sheet + selected columns.
   - Size: in-browser parse of the whole file is fine within the existing 10 MB cap.

**No backend changes in Phase A.** (`ALLOWED_EXTENSIONS` still `.csv`/`.xlsx`; the file the backend receives is a `.csv`.)

---

## Phase B — Export (PPTX, XLSX, PNG, Markdown/HTML)

### Shared chart-image render
Extend `pdf_export.py` with `render_chart_images(session_id) -> list[{chart_id, title, png_bytes}]`: reuse the long-lived Chromium, navigate to `/report/{id}/print`, wait for `data-charts-ready`, then screenshot each chart element. Requires **stable per-chart screenshot targets** on the print route — add `data-chart-export-id={chart_id}` (and ensure each chart's container is individually locatable) to the print page's chart cards. Returns one PNG per chart, in report order.

Exports should include **all** the report's charts (main + sidebar) — the export is the complete artifact. If the print route renders only the main column today, the image-render step (or a small dedicated export view / query param) renders the full set.

### Per-format builders (new module `src/api/report_export.py`)
All take the `Report` (from `db.get_report`) + (for image formats) the rendered chart PNGs:
- **PPTX** (`python-pptx`, new dep) — light-themed deck: (1) title slide (report title/date), (2) Insights summary slide, (3) a KPI slide rendering `report.key_metrics` as text tiles, (4) one slide per chart = the PNG + its caption. Editorial fonts where practical; clean default otherwise.
- **XLSX** (`openpyxl`/`pandas`) — workbook: a **"Data"** sheet (the source rows, read from the stored CSV via `storage.download_by_key`) + a **"Summary"** sheet (the KPI label/value pairs + the written summary text). No chart images needed.
- **PNG (zip)** — the rendered chart PNGs, zipped (filenames from chart titles).
- **Markdown / HTML** — a text report: the summary, a KPI list/table, then each chart as an embedded image (base64 data-URI) with its caption. Markdown for wikis/Notion; standalone HTML as an alternative.

### Endpoints (mirror the existing `export.pdf` style)
`GET /report/{id}/export.pptx`, `export.xlsx`, `export.zip` (images), `export.md`, `export.html` — each returns the file with the right `media_type` + `Content-Disposition`. All reuse `db.get_report` + (image formats) `render_chart_images`. **No credit gate** (consistent with `export.pdf`). PostHog: a `report_exported` event with `{ format }` (extend the existing pdf-export events).

### Frontend
`Toolbar.tsx`: replace the single "Export PDF" button with an **"Export ▾"** menu (PDF · PowerPoint · Excel · Images (zip) · Markdown). Each item fetches its endpoint via `apiFetch` and downloads the blob (the existing PDF download logic, generalized over format + filename + extension).

---

## Dependencies & schema
- **New:** `python-pptx` (backend `requirements.txt`).
- **No schema changes** — `Report` already carries `charts` (with specs/captions) + `key_metrics`; chart images are derived at export time. No DB migration.
- No new frontend deps (`xlsx` + `papaparse` already present).

---

## Phasing (plan order)
1. **Phase A — Import UX** (smaller, frontend-only): sheet picker + column deselector + emit-CSV in `page.tsx`. Ships on its own.
2. **Phase B — Export**: the shared `render_chart_images` + print-route screenshot targets → `report_export.py` builders → endpoints → the Toolbar Export menu. Ships after.

---

## Scope & non-goals
**In scope:** the four export formats + their endpoints + the Export menu; the Excel sheet picker + column deselector; `python-pptx`.

**Non-goals:**
- Google Sheets import and any new import formats (JSON/TSV/Parquet) — explicitly dropped.
- Credit-gating exports — they stay free.
- Server-side data cleaning/transforms beyond column selection (e.g. row filtering, type coercion UI) — out of scope.
- Native (editable) PPTX/Excel charts — charts are embedded as images.
- Changes to credits, auth, generation, or the credit-gated flows.

---

## Verification
- **Phase A:** `tsc` + `next build` clean. Manual: a multi-sheet `.xlsx` shows the picker and the chosen sheet's data previews; deselecting a column excludes it from the generated report (and the stored CSV); a plain CSV still works (no picker, column-deselect works); the ≥1-column guard holds.
- **Phase B:** backend tests for each builder (`report_export.py`): given a sample `Report` (+ fake chart PNGs), the PPTX opens with the expected slide count, the XLSX has Data + Summary sheets with correct values, the zip contains one PNG per chart, the Markdown/HTML contains the summary + KPIs + embedded images. `render_chart_images` returns one image per chart (smoke/integration, may be opt-in like the PDF test). Endpoints return correct `media_type` + non-empty bytes. Full `pytest` green; `next build` clean.
- Manual: each Export-menu item downloads a valid, openable file; the PDF export still works; exports remain free.
