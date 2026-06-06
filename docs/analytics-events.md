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
| `report_generation_started` | BE | `/generate-report` accepted (after credit check) | rowCount, columnCount, filename, sizeBytes, deep, customPrompt | baseline |
| `report_generation_succeeded` | BE | report built + persisted | reportId, rowCount, columnCount, chartCount, **usedFallback**, **fallbackChartCount**, modelSelection, modelNarrative, input/output/cacheReadTokens, estCostUsd, elapsedMs, deep, customPrompt | baseline · **2026-06-06: +usedFallback, +fallbackChartCount** |
| `report_charts_composed` | BE | after a report is generated | reportId, chartCount, fallbackChartCount, modelChartCount, usedFallback, allFallback, fallbackRatio, chartKinds[], keyMetricsCount, modelSelection, deep, customPrompt, rowCount, columnCount | **Added 2026-06-06.** Primary model-output-quality event (see "Fallback rate" below). |
| `report_generation_failed` | BE | generation raised | reason, errorClass, httpStatus, elapsedMs | baseline |
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
