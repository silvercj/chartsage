# ChartSage — Future Improvements / Backlog

A running list of deferred work and hardening ideas. Not committed plans — capture so they aren't lost. Each can become its own brainstorm → spec → plan when picked up.

---

## Admin Console — security hardening

*Deferred 2026-06-02. Shipped as v1 with a shared `X-Admin-Token` (Secret Manager), a server-side **fail-closed** guard (`require_admin`, constant-time compare), and a gated `/admin` route. This is acceptable for a single trusted operator; the items below harden it for heavier use or more admins.*

**Residual risks in the current design:**
- **No per-admin identity** — whoever holds the token *is* the admin; no record of *who* acted; a leak grants full power until rotated.
- **Token in the browser** (`sessionStorage`) while the console is open — readable by any XSS on the `chartsage-xi.vercel.app` origin.
- **No expiry / no auto-rotation** — long-lived token; rotation is manual (add secret version + redeploy).
- **No rate limiting or grant-frequency cap** — the 100k/grant bound limits one call, not repeated calls.
- **All-or-nothing power** via the service-role DB client (bypasses RLS); no read-only vs grant split.

**Hardening, ranked by value:**
1. **Email-allowlist auth (replace the shared token).** Highest value. Gate on the caller's Supabase JWT `email` claim (already present) against an `ADMIN_EMAILS` env var. Gives per-admin attribution, inherits the account's auth (incl. any 2FA), and removes the secret from the browser entirely. Small change to `deps.py` + the frontend (use the normal authed `apiFetch`).
2. **Restrict `/admin` exposure** — keep the route localhost-only (don't deploy to prod) or put it behind an IP allowlist / Vercel WAF rule. Shrinks the XSS/exposure surface and hides the tooling.
3. **Rate limiting + abuse alert** — a Vercel WAF rate-limit on `/admin/*`, plus a PostHog alert if `admin_credit_grant` total per day exceeds a threshold (cheap abuse tripwire; fits the "everything covered by analytics" goal).
4. **Token rotation policy** — periodic rotation or short-lived tokens.
5. **Granular roles + caps** — read-only vs grant permissions; an optional per-day grant cap per admin.
6. **Search scaling** — `db.search_accounts` currently loads *all* auth users via `auth.admin.list_users()` and filters in Python (fine at current scale). Move to a server-side email filter / pagination (or a SQL RPC joining `auth.users` + `profiles`) as the user base grows.

---

## Payments — refund / chargeback clawback

*Deferred 2026-06-02. SP4 payments shipped **live** (Stripe Checkout → signature-verified, idempotent `checkout.session.completed` webhook → `grant_credits`; USD packs $5/$15/$40; full analytics funnel incl. `checkout_started`, `credits_purchased`, `checkout_cancelled`). The happy path is complete and idempotent; this is the one remaining money-impact gap.*

**Gap:** a refund or chargeback returns the customer's money but does **not** claw back the granted credits — so buy → spend → refund leaves them with free credits. Acceptable at beta volume (refunds are rare; handle manually via the admin console with a negative adjustment).

**When to build:** once there's real payment volume, or if any refund abuse appears.

**Sketch:**
- Backend handler for Stripe **`charge.refunded`** (consider **`charge.dispute.created`** for chargebacks → alert + manual handling rather than auto-clawback).
- On refund: deduct `min(credits_for_that_purchase, current_balance)` — **floor the balance at 0** (the credits may already be spent). Resolve the purchase's credit amount from the original session / the `stripe_purchase` ledger row (match on the session `ref`).
- **Idempotent** on the refund/event id (reuse the `stripe_events` pattern) so retries don't double-deduct.
- **Partial refunds:** prorate, or — simplest v1 — only act on full refunds.
- Enable `charge.refunded` on **both** the live and test webhook endpoints (the dashboard toggle is useless without the handler; the webhook already 200-ignores all non-checkout events, so this is additive and safe).

---

## Product / feature backlog

- **Computed / derived metrics in chart selection** *(raised 2026-06-07 from the event-data posts; potentially high-value, larger piece).* ChartSage's chart tools only *aggregate* existing columns (mean/sum/count by a group); they can't compute a **derived metric** — a ratio or rate across columns/tables, e.g. cards-per-game (bookings ÷ matches), pole-to-win % (wins ÷ poles per circuit), blowout % (share of 4+ goal-margin games). Today the only way to surface such an insight is to **pre-compute it into the uploaded CSV**, which forces over-distilling the dataset down to the single narrative — and a thin input table *starves* chart selection: the model under-picks → the heuristic fallback fires → duplicate/generic charts → a thin report. Since the rendered report **is** the product (and our marketing demo), this directly caps richness. **If ChartSage could derive metrics, users could upload raw/rich data and still get the derived insight — far richer reports and far less prep.** Possible shapes: (a) a pre-selection step where the model proposes derived columns (an expression over the profile) computed in pandas before selection; (b) a ratio/rate aggregation in the executors (`value = count(A)/count(B)` or `sum(A)/sum(B)` per group); (c) a "computed column" chart tool. Likely touches profile → selection → executors — its own brainstorm → spec.
- **Geo / map charts** — a chart family deferred from the chart-types expansion; warrants its own brainstorm (choropleth, point maps; data shape + geocoding considerations).
- **KPI deltas** — period-over-period change indicators on the KPI tiles (fast-follow on the existing tiles).
- **Deep-analysis pricing revisit** — `DEEP_ANALYSIS_COST` shipped at 250 credits as a placeholder; revisit against real cost/latency once there's usage data.
- **Payments** — shipped **live** (Stripe Checkout + idempotent webhook + buy-credits UI). Remaining hardening: refund/chargeback clawback (see the dedicated section above).
- **Playwright E2E suite** — automated end-to-end browser tests (raised as an idea; not started).
- **Data deletion / retention** — user-initiated data deletion + a retention policy (privacy/compliance).

## Polish / housekeeping

- **README refresh** — bring it up to date with the current product (redesign, marketing site, chart types, exports, smarter analysis, admin console).
- **Dedicated OG image** — a purpose-built social/Open Graph image for the marketing site (currently relies on defaults).

## Generation engine (from the selection-floor fix, 2026-06-02)

*Two non-blocking nits flagged in review of the chart under-selection fix (which added the reach-for-more retry + the profiler top-N rescue). Both bounded; logged for later hardening.*

- **Profiler ↔ executor cardinality mismatch.** The profiler now keeps a repeating object column `categorical` up to 200 distinct (ratio ≤ 0.5), but most chart executors cap at `MAX_CATEGORIES = 30` — only `pie_chart` renders past that (top-8 + "Other"). A borderline short-text column (31–200 repeating values) can therefore yield one weak pie. Harmless (never a crash), and partly intentional (top-N is desired), but worth aligning: either lower the profiler ceiling toward ~50–60, or add a mean-string-length heuristic to separate "tags/cities" from "sentences."
- **Unique year/temporal column mis-classified as `identifier`.** A per-season dataset's `year`
  column (unique → cardinality == row count) gets role `identifier` in the profiler, so it's dropped
  from the numeric metrics: Haiku is nudged to treat it as non-chartable (under-picks → fallback)
  **and** the time-series fallback (`_ordinal_index` in `fallback.py`, which only scans numeric-role
  columns) can't find the axis → no metric-over-year line, just generic histograms. The F1-reliability
  post (2026-06-07) needed a re-roll because of this. Fix: don't mark a temporal-named / plausible-year
  column `identifier` (it's the time axis, not an ID), and/or have `_ordinal_index` scan all columns.
  Bites any per-year dataset — i.e. half the event-data posts.
- **`key_metrics` could be recomputed on the reach-for-more / deepen round.** `_execute_tool_calls` overwrites `self._key_metrics` if a later selection round emits a `key_metrics` call (the system prompt still says "call key_metrics first"). Idempotent in practice (recomputed from the same data), but cleaner to strip the `key_metrics` tool from the reach-for-more and deepen tool lists, or skip the overwrite when metrics already exist.

---

## 2026-06-04 — QA / eval harness baseline findings

*The new `qa/` harness (`make qa`) ran its first full baseline: 16 synthetic edge-case datasets through the real generation pipeline + the Haiku judge → **PASS 4 · WARN 7 · FAIL 5**. The harness is dev tooling (not deployed); these are the issues it surfaced. Triage below.*

**Real product bug — FIXED 2026-06-04 (needs a backend deploy to reach prod):**
- **Crash on duplicate / case-colliding column names.** The `duplicate_columns` case (headers `Value`, `label`, `value`) crashed generation: `AttributeError: 'DataFrame' object has no attribute 'dtype'`. After the endpoint lower-cased headers, `Value`/`value` collided, so `df["value"]` returned a *DataFrame* (not a Series) and `profile.py:62` then called `.dtype` on it. A real upload with colliding headers (common in Excel exports, or `Date`/`date`) would have returned a 500. **Fix:** new `src/api/column_utils.normalize_columns` lower-cases **and** de-duplicates collisions pandas-style (`value`, `value.1`), wired into all four dataframe-loading endpoints in `main.py` + the harness's `qa/pipeline.py`. Unit-tested (`tests/unit/test_column_utils.py`, incl. a `profile_dataframe`-survives-collision case); the `duplicate_columns` synthetic now generates 8 charts (WARN, no crash).

**Harness fix already applied (in the baseline-triage commit):**
- **Sampling false-positive in chart-data consistency.** `tall_100k` (120k rows) is analyzed on a deterministic 50k sample, but the validator recomputed counts/sums from the *full* df → false `chart_data_mismatch`. Fixed: `run_report` now returns the analyzed (post-sample/lower-case) frame and the runner validates against it. `tall_100k` → PASS after the fix.

**Calibration notes (tune later; not bugs):**
- **Judge `makes_sense=false` → hard FAIL may be too strict.** Several FAILs (`single_row_tiny`, `boolean_ish`, `id_like_bigints`) came from the judge flagging *weak-but-not-broken* charts (a 2-row histogram, balanced 50/50 medians, a narrative overstating "consistency") as `makes_sense=false` with `severity="warn"`. Per the spec any `makes_sense=false` is a FAIL. Consider mapping FAIL to the judge's own `severity=="fail"` and treating `makes_sense=false`+`warn` as WARN, to reserve FAIL for genuine regressions.
- **`all-identical-y` warns on legitimately balanced data.** Equal category counts (a 50/50 Yes/No split → both bars = 120) trip the all-identical-y warn. Valid data, mild noise. Consider scoping it (skip pure count-bars, or require >2 categories).
- **Tiny datasets yield weak charts.** `single_row_tiny` (2 rows) produced a histogram (2 points across 5 bins) + a trivial distribution — the judge rightly flagged both. The product could skip histograms/distributions below a small row threshold.
- **Narrative can overstate.** `id_like_bigints`: the narrative claimed regional medians were "remarkably consistent" while they ranged 220–264 (~20%); the judge's narrative-vs-charts check caught it. Worth a prompt nudge toward hedged language. (The id column itself was correctly treated as an identifier, not charted as a measure — the exact thing that dataset was built to verify.)

These are logged, not yet scheduled; each can become its own brainstorm → fix when picked up.
