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
- **`key_metrics` could be recomputed on the reach-for-more / deepen round.** `_execute_tool_calls` overwrites `self._key_metrics` if a later selection round emits a `key_metrics` call (the system prompt still says "call key_metrics first"). Idempotent in practice (recomputed from the same data), but cleaner to strip the `key_metrics` tool from the reach-for-more and deepen tool lists, or skip the overwrite when metrics already exist.
