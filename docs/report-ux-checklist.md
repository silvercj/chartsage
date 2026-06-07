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
- [ ] Crammed / overlapping x-axis labels (too many categories) — wide bar charts collapse to top-12;
  long category lists should rotate or truncate.

## Chart-type fit
- [ ] **Scatter used for a time series** where a line/bar reads better (e.g. "total goals vs year").
  A scatter is for *relationships*, not trends over time.
- [ ] Pie with too many slices; **box plot with one value per group** (renders empty).
- [ ] **Empty chart** (no x/y data) — e.g. a degenerate box plot — shouldn't be shown at all.

## Values & formatting
- [ ] **Percentage shown as `0.61` instead of `61%`** — the column needs a pct/rate/share/margin/ratio
  name (`_PERCENTAGE_KEYWORDS` in `src/api/chart_executor.py`), or the value is on the wrong scale.
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
