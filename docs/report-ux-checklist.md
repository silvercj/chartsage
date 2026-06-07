# Report UX checklist

Things to eyeball on a generated ChartSage report — especially **before a marketing post**
(the `event-data-post` skill, Step 6). Each is a real issue we've hit. When one recurs, fix
the *generation* (frontend chart code or backend selection) and tick the source note; add new
checks here as we find them.

## Axes & scale
- [ ] **Value axis forced to 0** when the data sits far from zero (years, large counts) → points/lines
  cram into a stripe and the trend vanishes. Scatter/line value axes should `scale: true` (fit the
  data). *Fixed for scatter 2026-06-07 (`charts/ScatterChart.tsx`); `chartTheme.ts` `valAxis` passes
  `scale` through, so any chart can opt in.*
- [ ] **Bar y-axis NOT starting at 0** — the opposite trap: bars must baseline at 0 or the lengths lie.
- [x] **Crammed / overlapping x-axis labels** — *fixed 2026-06-07*: bar labels auto-slant (length-aware
  `labelRotation` in `chartTheme.ts`, weighs count **and** label length) when they'd collide; wide bars
  also collapse to top-12. *(Other category charts — grouped bar, box, histogram — can adopt the same helper.)*

## Chart-type fit
- [x] **Scatter used for a time series** where a line/bar reads better (e.g. "total goals vs year").
  A scatter is for *relationships*, not trends over time. *Fallback fixed 2026-06-07: a year/ordinal
  column + metrics now leads with a metric-over-time **line** (`fallback.py` `_ordinal_index` /
  `_timeseries_line`), instead of a "year — distribution" histogram + a "year vs matches" scatter.*
- [x] **Line grouped by its own measure** — a `group_by` equal to the value column makes one
  one-point series per value: a tangle of coloured spikes, x-axis mislabeled with the values.
  *Fixed 2026-06-07: `execute_line_chart` drops a `group_by` that equals the value/date column.*
- [ ] Pie with too many slices; **box plot with one value per group** (renders empty).
- [ ] **Empty chart** (no x/y data) — e.g. a degenerate box plot — shouldn't be shown at all.

## Values & formatting
- [ ] **Percentage shown as `0.61` instead of `61%`** — the column needs a pct/rate/share/margin/ratio
  name (`_PERCENTAGE_KEYWORDS` in `src/api/chart_executor.py`), or the value is on the wrong scale.
- [ ] **The reverse — a plain count wrongly shown as a `%`** because its name contains a percentage
  keyword (e.g. a sports goal **`margin`** of 2.3 rendering as "2.3%"). Rename the CSV column so it
  types as a number (e.g. `goal_diff`, `goal_gap`).
- [ ] Currency/units missing where expected.
- [ ] **Mojibake** in labels (accented names garbled) — ASCII-clean the source CSV.
- [x] **Moving-average overlay mislabeled "3-mo avg"** on non-monthly lines (a 3-point avg, auto-added
  to any single-series line ≥12 pts). *Fixed 2026-06-07: `LineChart.tsx` `movingAvgLabel()` derives the
  period from `x_label` → "3-yr avg" / "3-mo avg" / neutral "3-pt avg".*

## Selection quality
- [ ] **Fallback charts** — generic `"<column> — distribution"` titles mean chart-selection under-picked.
  Track the rate via PostHog `report_charts_composed` (see `docs/analytics-events.md`).
- [x] **Fallback duplicating model charts** — when the model picks <3, the fallback re-derives from the
  same df and can re-make charts the model already chose (each line showed twice). *Fixed 2026-06-07:
  `drop_duplicates` (report_generator) drops fallback specs matching a model chart's kind + source columns.*
- [ ] **Wrong hero** — the lead chart isn't the intended analysis.
- [ ] Truncated / awkward auto-titles.

## Key metrics
- [ ] **Nonsensical key metric** at the top of the report — e.g. "Year span: 2022" (a year, not a
  span) or a decimal count ("22.23 teams/tournament"). Backend `key_metrics`; not in the post but in
  the report. *(Open — spotted on the blowouts report.)*

## Change log
- **2026-06-07** — created. Added the *scatter forced-zero* fix (the World Cup "total goals vs year"
  stripe) and banked the chart-type-fit + empty-chart checks spotted in the same report.
- **2026-06-07** — **time-series fallback**: a year/ordinal axis + metrics now leads with trend
  **lines** (hero + 2nd metric), not histograms of the year. The World Cup "blowouts by year" report
  came out all-histogram/scatter when Haiku under-picked; the fallback now produces the trend the
  data wants. Also banked the *count-wrongly-shown-as-%* check (a goal `margin` formatting as a
  percentage). (`src/api/fallback.py`, `tests/unit/test_fallback.py`.)
- **2026-06-07** — fixes surfaced by the new *generate-and-self-QA-before-publishing* loop (World Cup
  blowouts): (a) `execute_line_chart` drops a `group_by` equal to the value/date column — it was
  rendering a spiky tangle (one one-point series per value); (b) `LineChart.tsx` labels the moving
  average by real period ("3-yr avg") instead of a hardcoded "3-mo avg". Banked the nonsensical
  key-metric check. (`src/api/chart_executor.py`, `src/app/report/[id]/charts/LineChart.tsx`.)
- **2026-06-07** — more from the cards self-QA: (a) `execute_line_chart` drops a **near-unique
  group_by** (continuous metric → 12-series spiky tangle); (b) fallback **dedup** — drop fallback
  charts that duplicate a model chart (cards report showed the cards + reds lines twice). QA tool
  now understands box/heatmap series. (`chart_executor.py`, `fallback.py`, `report_generator.py`.)
