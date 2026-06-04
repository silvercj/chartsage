# ChartSage QA / Eval Harness

On-demand regression tooling. Runs a corpus of edge-case + real datasets through
the **real** generation pipeline (the same code `/generate-report` uses), then
validates every output deterministically and with a Haiku LLM judge. **Not deployed.**

## Run

```
make qa                              # full corpus (synthetic + qa/datasets/*.csv)
make qa ARGS="--only synthetic"      # only the synthetic edge cases
make qa ARGS="--only real"           # only dropped-in CSVs
make qa ARGS="--only year_as_int"    # a single dataset by name
make qa ARGS="--no-judge"            # deterministic checks only (free, no Haiku)
make qa ARGS="--limit 5"             # first 5 datasets
```

Requires `ANTHROPIC_API_KEY` in `.env` (already present). The runner uses real
Haiku for chart selection, narrative, and the judge — the same models and prompts
production uses, so the charts **and** the written narrative are generated identically.

## Add a real dataset

Drop any `*.csv` into `qa/datasets/`. It's auto-discovered (name = filename stem).

## Output

Each run writes `qa/results/<timestamp>/`:
- `report.md` — top table (dataset → PASS/WARN/FAIL + issue counts), then per-dataset detail.
- `summary.json` — machine-readable roll-up.
- `<dataset>.json` — full per-dataset record (report, deterministic issues, judge verdict).

`qa/results/*` is gitignored.

## Verdict rule

- **FAIL** — any deterministic `fail` issue, or the judge marks any chart as not making sense.
- **WARN** — any `warn` issue (or the judge says the narrative doesn't match), no FAIL.
- **PASS** — clean.

## What it checks

Deterministic (`qa/validators.py`): generation crash; degenerate charts (empty x/y,
<2-distinct-x lines, all-NaN/zero/identical y); chart-data consistency (recomputed
count/sum groupby vs the chart's y); KPI sanity (non-finite; year-look-alike metrics);
empty narrative; chart count below the selection floor (warn); missing axis labels (warn).

Judge (`qa/judge.py`): does each chart make sense for the data, is any chart
misleading/degenerate/redundant, does the narrative match the charts.

## How it stays faithful to production

`qa/pipeline.run_report` is the anti-drift guard: it imports the **real**
`sample_for_analysis`, `MODEL_SELECTION`/`MODEL_NARRATIVE`, `profile_dataframe`, and
`ReportGenerator`, and reproduces the `/generate-report` sequence (lower-case columns
→ sample → 2-col/1-row guards → profile → `ReportGenerator(...)` → `build_report`
→ sampling note → `model_dump()`). It deliberately skips only the HTTP/auth/credit/
storage/PDF wrapper, none of which change what the LLM produces. If production's
generation sequence changes, update `run_report` to match.
