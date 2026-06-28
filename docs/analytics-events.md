# Analytics Events Dictionary

Canonical catalogue of every product-analytics event ChartSage emits (PostHog).

**Why this matters.** These events are how we see what the product *and the models* are
doing — activation, credit economics, and (since 2026-06-06) **model-output quality** such
as the chart-fallback rate. Analytics is a first-class concern: a feature isn't done until
its events exist and this file describes them.

**Maintenance rule — do this in the SAME change.** Whenever you add, rename, re-prop, or
remove an event, update this file in the same PR/commit:
- New event or new prop → add it / note it with a dated line in **Lifecycle**.
- Removed event → keep the row, mark `Removed YYYY-MM-DD` (never silently delete — dashboards
  and downstream queries depend on knowing it went away).
- Renamed event → leave the old row as `Renamed → new_name (YYYY-MM-DD)` and add the new one.

**Conventions**
- **Backend**: `posthog.capture(distinct_id, "<event>", {props})` in `src/api/main.py`, via
  `PostHogServer` (`src/api/posthog_server.py`). Analytics never breaks product flow — capture
  failures are swallowed at WARN.
- **Frontend**: `posthog.capture?.('<event>', {props})` (client). `posthog.identify` ties the
  anon id → user id on login.
- Every event carries PostHog's `distinct_id` (anon or user identity). `reportId` identifies
  the report/session where relevant.
- **"baseline"** in Lifecycle = catalogued 2026-06-06; precise creation dates before that
  weren't tracked. Date everything from here on.

---

## Model output & generation — the quality signals

| Event | Src | Trigger | Key props | Lifecycle |
|---|---|---|---|---|
| `report_generation_started` | BE | `/generate-report` accepted (after credit check) | **reportId**, rowCount, columnCount, filename, sizeBytes, deep, customPrompt | baseline · **2026-06-07: +reportId** (join key) |
| `report_generation_succeeded` | BE | report built + persisted | reportId, rowCount, columnCount, chartCount, **usedFallback**, **fallbackChartCount**, modelSelection, modelNarrative, input/output/cacheReadTokens, estCostUsd, elapsedMs, deep, customPrompt | baseline · **2026-06-06: +usedFallback, +fallbackChartCount** |
| `report_charts_composed` | BE | after a report is generated | reportId, chartCount, fallbackChartCount, modelChartCount, usedFallback, allFallback, fallbackRatio, chartKinds[], keyMetricsCount, **degenerateRejectedCount**, **curationDroppedCount**, modelSelection, deep, customPrompt, rowCount, columnCount | **Added 2026-06-06.** Primary model-output-quality event (see "Fallback rate" below). **2026-06-10: +degenerateRejectedCount** (model picks the degeneracy screen bounced into the retry round — `degenerate_reason()` in `chart_executor.py`), **+curationDroppedCount** (charts the narrative pass dropped after seeing their computed data). |
| `chart_generated` | BE | one per chart, after a report is generated | reportId, **chartId**, **kind**, **isFallback**, deep, customPrompt | **Added 2026-06-07.** One event per chart so the chart-kind mix is a simple breakdown by `kind` — see "Chart-type mix" below. |
| `report_generation_failed` | BE | generation raised | **reportId**, reason, errorClass, httpStatus, elapsedMs | baseline · **2026-06-07: +reportId** |
| `claude_overloaded` | BE | model 529/busy during selection | stage | baseline |
| `generate_more_started` | BE | `/generate-more` begins | reportId, existingChartCount | baseline |
| `generate_more_succeeded` | BE | extra charts appended | reportId, newChartCount, **chartKinds[]**, inputTokens, outputTokens, estCostUsd, elapsedMs | baseline · **2026-06-06: +chartKinds** |
| `generate_more_failed` | BE | generate-more raised | reportId, reason, errorClass?, elapsedMs | baseline |
| `deepen_started` | BE | `/deepen` begins | reportId, existingChartCount | baseline |
| `deepen_succeeded` | BE | iterative-deepening finished | reportId, newChartCount, elapsedMs | baseline |
| `deepen_failed` | BE | deepen raised | reportId, reason, … | baseline |
| `add_chart_started` | BE | `/add-chart` begins | reportId, mode, chartType | baseline |
| `add_chart_succeeded` | BE | one chart added | reportId, mode, chartKind, elapsedMs | baseline |
| `add_chart_failed` | BE | add-chart raised / produced no chart | reportId, reason, errorClass?, httpStatus?, elapsedMs | baseline |

**Joining request → report → charts (one key).** Every event in a generation request carries the same
**`reportId`**: `report_generation_started` (the request) → `report_generation_succeeded` /
`report_charts_composed` (report level) → `chart_generated` (per chart), and `report_generation_failed`
on the error path. So you can funnel or `JOIN ... USING (reportId)` start → outcome → chart mix.
`chart_generated` also carries a unique **`chartId`** (the chart's own id) for chart-level identity —
e.g. tying a chart to a later `add_chart` / `report_feedback`. *(reportId threaded through the request
2026-06-07; previously `started`/`failed` had none, so the request couldn't be joined to its report.)*

**Fallback rate (the motivating use case).** A chart is a *fallback* when chart-selection
(Haiku) returned too few charts and the deterministic heuristic (`pick_fallback_charts`) filled
in — marked by the `fallback:` `intent` prefix; computed by `chart_composition()` in
`src/api/fallback.py`. Compute the rate over `report_charts_composed`:
- **% of reports using any fallback** = share with `usedFallback = true`.
- **% of total-miss reports** (model picked *nothing*, like the 2026-06-06 hurricanes report) =
  share with `allFallback = true`.
- **avg fallback share per report** = mean of `fallbackRatio`.
Segment by `modelSelection`, `rowCount`/`columnCount`, `deep`, `customPrompt` to find where the
model under-selects.

**Chart-type mix.** `chart_generated` fires once per chart with its `kind` (one of: bar, grouped_bar,
dual_axis, histogram, scatter, line, pie, box, heatmap, treemap) and `isFallback`. Plot the
distribution with a **Trends** insight on `chart_generated` **broken down by `kind`** — a pie for the
overall split, or a stacked area / line to watch it over time. Add an `isFallback = true` filter to
split model picks from the heuristic fallback. (Added 2026-06-07 after a run of line-heavy reports —
this makes the skew visible; the report-level `chartKinds[]` array undercounts repeats within a report.)

## Sharing, exports & engagement

| Event | Src | Trigger | Key props | Lifecycle |
|---|---|---|---|---|
| `report_published` | FE + BE | Share → publish | reportId | baseline |
| `report_unpublished` | FE + BE | unpublish | reportId | baseline |
| `embed_viewed` | FE | embedded report rendered | reportId | baseline |
| `report_feedback` | FE | thumbs up/down on a report | rating, comment, reportId | baseline |
| `export_clicked` | FE | export button pressed | reportId, format | baseline |
| `report_exported` | BE | export endpoint served | reportId, format (pptx/xlsx/zip/md/html) | baseline |
| `pdf_export_started` | BE | PDF render begins | reportId, coldStart | baseline |
| `pdf_export_succeeded` | BE | PDF rendered | reportId, byteSize, elapsedMs | baseline |
| `pdf_export_failed` | BE | PDF render raised | reportId, … | baseline |

## Credits, billing & upsell

| Event | Src | Trigger | Key props | Lifecycle |
|---|---|---|---|---|
| `credits_granted` | BE | first-time profile + starter grant | amount, balance | baseline |
| `credits_spent` | BE | credits debited | amount, balance, reason | baseline |
| `out_of_credits` | BE | action blocked, balance too low | action, balance | baseline |
| `upgrade_required` | BE | anon hits a signup-gated action | action | baseline |
| `upgrade_intent_captured` | BE | upgrade-intent endpoint hit | — | baseline |
| `generate_more_upsell_cta` | FE | upsell modal CTA | method | baseline |
| `buy_credits_clicked` | FE | out-of-credits modal "buy" | source | baseline |
| `buy_pack_clicked` | FE | a credit pack selected | package_id | baseline |
| `checkout_started` | BE | Stripe checkout session created | package_id, credits | baseline |
| `checkout_cancelled` | FE | returned from a cancelled checkout | — | baseline |

## Auth, onboarding & anonymous

| Event | Src | Trigger | Key props | Lifecycle |
|---|---|---|---|---|
| `logged_in` | FE | session detected after login (also `identify`) | method | baseline |
| `signed_out` | FE | user signs out | — | baseline |
| `login_method_selected` | FE | login method chosen | method | baseline |
| `signin_cta_clicked` | FE | sign-in CTA (anon-limit page) | from | baseline |
| `onboarding_viewed` | FE | welcome page viewed | — | baseline |
| `onboarding_completed` | FE | onboarding finished | — | baseline |
| `anon_reports_claimed` | BE | anon reports merged into a new account | count | baseline |
| `anon_limit_page_viewed` | FE | anon upload-limit page shown | entryPoint | baseline |
| `anon_limit_blocked` | BE | anon blocked at the limit | — | baseline |
| `anon_cap_hit` | BE | anon IP/global cap reached | scope (ip/global) | baseline |

## Marketing & misc

| Event | Src | Trigger | Key props | Lifecycle |
|---|---|---|---|---|
| `marketing_cta_clicked` | FE | marketing CTA clicked | location (hero/pricing/nav/closing) | baseline |
| `contact_submitted` | FE | contact form submitted | — | baseline |
| `support_request` | BE | support/contact endpoint hit | (see `main.py`) | baseline |

## Landing → activation funnel

Added **2026-06-10** to locate where ad traffic drops off. The paid-ad investigation found
**0 of ~199 paid visitors reached the uploader** (`/` → `/app`); these events make that funnel
explicit: `landing_hero_viewed` → (`sample_report_clicked` | `marketing_cta_clicked`) →
`uploader_viewed` → `file_selected` → `report_generation_started` (BE) → `report_generation_succeeded` (BE).
Build a Funnel insight over them, and segment by `$device_type` / `utm_source` to compare paid vs organic.

| Event | Src | Trigger | Key props | Lifecycle |
|---|---|---|---|---|
| `landing_hero_viewed` | FE | marketing landing hero mounted | — | **Added 2026-06-10** |
| `sample_report_clicked` | FE | clicked a "See a live example" entry point (hero button or hero preview) | location (hero/hero_preview) | **Added 2026-06-10** |
| `sample_report_generated` | BE | the public showcase report was lazily generated for the first time | reportId | **Added 2026-06-10** |
| `sample_make_your_own_clicked` | FE | clicked "Make your own" on the sample report banner | — | **Added 2026-06-10** |
| `uploader_viewed` | FE | reached the `/app` uploader | — | **Added 2026-06-10** |
| `file_selected` | FE | picked/dropped a file in the uploader | sizeBytes, ext (csv/xlsx) | **Added 2026-06-10** |
