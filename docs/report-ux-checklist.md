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

## Selection quality
- [ ] **Fallback charts** — generic `"<column> — distribution"` titles mean chart-selection under-picked.
  Track the rate via PostHog `report_charts_composed` (see `docs/analytics-events.md`).
- [ ] **Wrong hero** — the lead chart isn't the intended analysis.
- [ ] Truncated / awkward auto-titles.

## Change log
- **2026-06-07** — created. Added the *scatter forced-zero* fix (the World Cup "total goals vs year"
  stripe) and banked the chart-type-fit + empty-chart checks spotted in the same report.
- **2026-06-07** — **time-series fallback**: a year/ordinal axis + metrics now leads with trend
  **lines** (hero + 2nd metric), not histograms of the year. The World Cup "blowouts by year" report
  came out all-histogram/scatter when Haiku under-picked; the fallback now produces the trend the
  data wants. Also banked the *count-wrongly-shown-as-%* check (a goal `margin` formatting as a
  percentage). (`src/api/fallback.py`, `tests/unit/test_fallback.py`.)
