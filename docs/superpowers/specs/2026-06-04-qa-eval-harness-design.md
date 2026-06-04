# ChartSage QA / Eval Harness — Design Spec

**Status:** Approved (design), pending spec review
**Date:** 2026-06-04
**Type:** Internal dev tooling (quality / regression harness) — not deployed to prod

**Goal:** Catch data-handling regressions — broken/degenerate charts, type mishaps, weak/incoherent narrative, crashes — *before users hit them*, by running a corpus of datasets through the **real** generation pipeline locally and validating every output deterministically **and** with an LLM judge. Runs on-demand. (Motivating example: the year-as-int line-chart bug, which surfaced as a 1-point line in the chart spec.)

## Fidelity — runs the real pipeline (the crux)
The harness must exercise **the exact code path production uses**, or it's worthless:
- It imports the production modules — `profile.profile_dataframe`, `report_generator.ReportGenerator`, `chart_executor` — and uses the **real Anthropic Haiku** client via the existing `claude_client` + `llm_config` (key already in `.env`).
- It runs the **same two LLM passes** as chartsage.app: chart **selection** (`generate_charts()`) and the **narrative** (`generate_narrative()`), with the same prompts, `CHART_TOOLS`, executors, and model. Charts **and** the written response are produced identically to production.
- It also replicates the parts of the API endpoint that affect *output* (not just the generator core): `_load_dataframe` (CSV/Excel read) and `sample_for_analysis` (row-sampling for big files), and the report-assembly (layout + `key_metrics`). So big-file / sampling behaviour matches too.
- It deliberately **skips** the HTTP/auth/credit/storage/PDF wrapper — none of which change what the LLM produces. (Visual/PDF validation is a deferred "Hybrid" add-on.)
- A guard against drift: the runner calls a single shared helper that mirrors the endpoint's generation sequence, so if production's sequence changes, the harness changes with it.

## Architecture
A new top-level **`qa/`** package, run locally (no prod, no credits):

### 1. Corpus
- **`qa/generators.py`** — ~15 synthetic edge-case datasets, each a function returning a `pandas.DataFrame` (+ a short `name`/`intent`), each targeting a known failure mode (list below).
- **`qa/datasets/`** — drop-in folder for real CSVs (the Kaggle files etc.); every `*.csv` is auto-discovered and run.

### 2. Runner — `qa/run_eval.py`
For each dataset (synthetic + discovered): load → `profile_dataframe` → sample (mirroring the endpoint) → `ReportGenerator.generate_charts()` + `generate_narrative()` → assemble the report (chart specs + narrative + key_metrics + layout). Captures the full report **or** the exception/traceback. Writes per-dataset output (compact JSON: profile summary, chart specs, narrative, key_metrics, timings, any error) to `qa/results/<timestamp>/<dataset>.json`.

### 3. Validation — two layers
- **Deterministic gate** (`qa/validators.py`) — pure functions over (df, report) returning a list of issues:
  - Generation completed (no uncaught exception).
  - No empty/degenerate charts: every chart has non-empty x/y or series; **line charts have ≥2 distinct points**; no all-NaN / all-zero / all-identical series.
  - **Chart-data consistency**: spot-check that chart values are derivable from the source df (e.g. a count/sum bar matches a recomputed `groupby`; a line's points match the period aggregation).
  - **KPI sanity**: numeric, finite; flag a metric labelled years/count/total whose value equals a 4-digit calendar year (would have caught the "2,022 years covered" slip); flag absurd magnitudes.
  - Narrative present + non-trivial; chart count ≥ the selection floor; axis labels present.
- **LLM judge** (Haiku, in-harness) — per report, given the data profile + chart specs + narrative, scores: does each chart make sense for this data? Is any chart misleading, degenerate, or redundant? Does the narrative match the charts (no claims unsupported by a chart)? Returns structured issues with severity. This is the "agent validates" layer, baked in so one run produces both.

### 4. Results report — `qa/results/<ts>/report.md`
A scannable summary: a top-line table (dataset → **PASS / WARN / FAIL** + issue count), then per-dataset detail (the deterministic + judge issues, with the offending chart). Plus a machine-readable `summary.json`. The human (and the agent) review the FAILs/WARNs.

### 5. Trigger
- `make qa` → `python qa/run_eval.py` with flags: `--only synthetic|real|<name>`, `--no-judge` (deterministic only, free), `--limit N`.
- On-demand only (no scheduling). The agent can also run it and synthesise findings for the user.

### 6. Test-suite health (folded in — quick win first)
- Add **`pytest-timeout`** to `requirements.txt` and a default per-test timeout in `pytest.ini` (e.g. `timeout = 60`), so no test can hang a run/deploy again (today a network-bound test hung 15 min).
- Identify the straggler (run with `--timeout` to get the offending traceback) and **mock its network call** so the suite is fully offline-safe.

## Synthetic edge cases (~15, each a generator)
year-as-int · mixed date formats (DD/MM/YYYY, MM-DD-YY, ISO, Excel-serial int, "Jan 2020") · NaN-heavy · high-cardinality category (500+) · unicode/emoji labels · currency strings (£1,200 / $1.2k / "1.234,56") · single-row + tiny (2–3 rows) · duplicate column names · all-categorical · all-numeric · wide (50+ cols) · tall (100k+ rows, to exercise sampling) · boolean-ish (Yes/No, 0/1, True/False) · mixed-type column (numbers + strings) · id-like big integers (must not be charted as a measure) · negatives + extreme outliers.

## Scope & non-goals (v1)
**In:** the `qa/` package (generators, runner, deterministic validators, LLM judge, results report), the synthetic corpus + a drop-in real-CSV folder, `make qa`, and the test-suite-health fix.
**Out (deferred):** production-API/PDF/visual validation (the "Hybrid" add-on — only if chart specs prove insufficient); auto-scheduling/CI gating (on-demand only for now); auto-fixing (the harness flags; fixes are separate); seeding the corpus with every real Kaggle file (start with a handful).

## Phasing (plan order)
1. **Test-suite health** — `pytest-timeout` + default timeout + mock the straggler. (Quick; unblocks reliable runs.)
2. **Generators + runner** — the ~15 synthetic datasets + the fidelity-faithful run helper + per-dataset JSON output.
3. **Deterministic validators** — the structural + data-consistency + KPI checks, with unit tests (incl. a regression asserting the year-as-int dataset would be flagged if the parser broke).
4. **LLM judge + results report** — the Haiku judge pass + `report.md`/`summary.json`.
5. **Wire `make qa` + baseline run** — run the full corpus, triage the findings, fix or log each.

## Verification
- Unit tests for the generators (each produces the intended shape) and the deterministic validators (each check fires on a crafted bad report and passes a good one).
- A **self-test**: temporarily reverting the year-fix makes the harness FAIL the year-as-int dataset (proves it catches the real bug class).
- A first full `make qa` run produces a `report.md`; every FAIL is either fixed or logged in FUTURE-IMPROVEMENTS.
- Full `pytest` green (and now hang-proof via the timeout).
