# ChartSage Smarter Input Handling — Design Spec

**Status:** Approved (design), pending spec review
**Date:** 2026-06-02
**Type:** Backend feature (data profiling + chart execution)

**Goal:** Make catalog/tag/listing datasets (Netflix, e-commerce, surveys) produce rich reports by (1) recognizing and charting **multi-value / multi-label columns**, and (2) accepting **larger files** via row-sampling.

**Context:** Root-caused this session. The real `netflix_titles.csv` (8807×12) generates only ~3 charts because its richest columns are **comma-separated multi-value** strings — `country="United States, India, UK"`, `listed_in="Dramas, International"`, `cast="A, B, C"`. The profiler counts the distinct *combinations* (`country` cardinality=2524, `listed_in`=2549), which exceeds the categorical ceiling (`distinct ≤ 200 & ratio ≤ 0.5` in `src/api/profile.py`), so both are marked `role="unusable"` and excluded — leaving ~3 usable columns. The fix is to detect delimited columns, split/explode them, and chart the **top-N individual values**. Separately, the 10 MB upload cap is fine for now but we'll raise it with row-sampling so large files are cheap to analyze (the analysis is column-driven — a representative sample is statistically faithful).

**Approved decisions:** multi-value detection = **conservative** (correctness over coverage — never chart a free-text column); size = **raise to 50 MB + row-sample** large files.

---

## Facet 1 — Multi-value / multi-label columns

### Detection (`src/api/profile.py`)
For an object column that would otherwise be classified `unusable` (high cardinality, not an identifier/date/numeric), attempt multi-value detection **before** falling through to `unusable`. Test candidate delimiters in order: `"; "`, `" | "`/`"|"`, `", "`, `","`. Flag the column as a **multi-value categorical** only when ALL of these conservative tests pass for a delimiter:
1. **Prevalence** — the delimiter splits into ≥2 parts in a strong majority of non-null cells (≥ 60%).
2. **Low atom cardinality** — the set of distinct exploded atoms is small enough to chart as top-N: `≤ _MV_MAX_ATOMS` (≈ 50).
3. **Short atoms** — atoms are tag-like, not sentences: mean atom length ≤ `_MV_MAX_ATOM_LEN` (≈ 30 chars) and no atom is absurdly long.
4. **Repetition** — atoms recur: `distinct_atoms / total_atom_occurrences` is low (atoms are reused across rows, not unique fragments).

When matched: `ColumnInfo(role="categorical", multi_value=True, delimiter=<d>, cardinality=<distinct atom count>, top_values=<top atoms with counts>)`. Otherwise the column stays `unusable` exactly as today.

This conservatively catches `country` (~20 atoms), `listed_in` (~20 genres); rejects `cast` (thousands of distinct actors → fails atom-cardinality), `description` (long, near-unique sentences → fails short-atoms + repetition), and identifiers (caught earlier).

**Schema (`src/api/schemas.py`):** `ColumnInfo` gains `multi_value: bool = False` and `delimiter: Optional[str] = None`. `DataProfile`'s prompt rendering shows a multi-value column with its atom `top_values` and a "(multi-value — top values: …)" marker so the model knows it's a splittable tag column.

### Charting (`src/api/chart_executor.py`)
A shared helper `explode_multi_value(series, delimiter) -> pd.Series` (split on the delimiter, strip whitespace, drop empties, explode to one atom per row). The `frequency_bar_chart`, `treemap_chart`, and `pie_chart` executors: when the target column is multi-value (flagged in the profile / detected via the stored delimiter), explode first, then `value_counts` the atoms and take the top-N (respecting `MAX_CATEGORIES`). The chart renders the top individual values ("Top 10 countries"). The executor reads the delimiter from the column's profile entry (passed through to the executor), so it splits the same way detection did.

### Selection (`src/api/report_generator.py` + `prompts/selection_system.txt`)
Multi-value columns now appear as usable categoricals in the profile, so the selection model (and the reach-for-more retry) can chart them. Add one prompt line: *"Some categorical columns are multi-value/tag columns (marked multi-value, e.g. genres, countries, tags) — chart them with frequency_bar_chart or treemap_chart to show the top individual values."*

---

## Facet 2 — Larger files + row-sampling

- **Raise the cap to 50 MB:** the backend upload limit and the frontend check (`src/app/app/page.tsx` `f.size > 10*1024*1024` → 50 MB) and the "up to 10 MB" copy.
- **Row-sample for analysis (`src/api/main.py`, generation path):** a constant `MAX_ANALYSIS_ROWS` (≈ 50,000). After loading the DataFrame, if `len(df) > MAX_ANALYSIS_ROWS`, replace it with a **deterministic** sample: `df.sample(n=MAX_ANALYSIS_ROWS, random_state=0).reset_index(drop=True)`. Profiling and all chart execution then run on the sample.
- **Data-quality note:** when sampled, append to `data_quality`: *"Analyzed a representative random sample of 50,000 of {total:,} rows."*
- **Consistency:** the **sampled** CSV is what gets stored (`storage.upload_csv`), so `generate-more` / `add-chart` / `deepen` (which re-read the stored CSV) operate on the same rows — and stored-file size is bounded. (Today the original is stored; here we store what we analyzed.)
- **Frontend:** raise the size check + copy; show a one-line note for large files (e.g. "Large file — we'll analyze a representative sample.").

---

## Dependencies & schema
- **No new dependencies** (pandas already present).
- **No DB migration** — `multi_value`/`delimiter` live on the in-memory `ColumnInfo`/report JSON; sampling changes only what's stored in the CSV blob.

## Phasing (plan order)
1. **Multi-value detection + schema** (`profile.py`, `schemas.py`) — TDD on a realistic multi-value frame.
2. **Multi-value charting** (`chart_executor.py` explode helper + the 3 executors) — TDD.
3. **Selection prompt line** + wiring so the model uses multi-value columns.
4. **Size cap + row-sampling** (`main.py` + frontend) — TDD on a large frame.
5. **Build + QA + live-verify on an actual `netflix_titles.csv`** + a large-file sampling check; deploy (backend Cloud Run + frontend Vercel) with user authorization.

## Scope & non-goals
**In scope:** conservative multi-value detection; split/explode top-N charting via the existing bar/treemap/pie executors; the profile/prompt signal; 50 MB cap; deterministic row-sampling with a data-quality note; tests + real-data verification.

**Non-goals:**
- New chart *types*, or using a multi-value column as a chart **axis/group/breakdown** (e.g. revenue *by* genre) — v1 is top-N counts of the atoms only.
- Per-atom cross-tabs / co-occurrence analysis.
- Streaming/chunked parsing of huge files — the cap stays at 50 MB; sampling handles row count, not raw bytes.
- Changing chart kinds, credits, or unrelated profiler roles.

## Verification
- **Facet 1 (TDD):** a realistic multi-value DataFrame (`country`/`listed_in`/`cast`/`description`) profiles `country`+`listed_in` as `multi_value` categorical, `cast`+`description` as `unusable`; the explode helper splits correctly; `frequency_bar`/`treemap` on a multi-value column produce top-N atom counts (not raw-combo counts).
- **Facet 2 (TDD):** a >50k-row frame is sampled to `MAX_ANALYSIS_ROWS` deterministically; the data-quality note appears; a small frame is untouched.
- **Full `pytest` green.**
- **Live (post-deploy):** generate a report from an **actual `netflix_titles.csv`** → confirm `country`/`listed_in` charts appear and the report reaches a rich set (target ~8–10); upload a large file → confirm the sample note + a sensible report. (No synthetic stand-in for the Netflix check this time.)
