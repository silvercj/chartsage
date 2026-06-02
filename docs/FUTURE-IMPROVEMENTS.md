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

## Product / feature backlog

- **Geo / map charts** — a chart family deferred from the chart-types expansion; warrants its own brainstorm (choropleth, point maps; data shape + geocoding considerations).
- **KPI deltas** — period-over-period change indicators on the KPI tiles (fast-follow on the existing tiles).
- **Deep-analysis pricing revisit** — `DEEP_ANALYSIS_COST` shipped at 250 credits as a placeholder; revisit against real cost/latency once there's usage data.
- **SP4 — payments** — credit purchase flow (the credit economy is live; buying credits is the next monetization sub-project).
- **Playwright E2E suite** — automated end-to-end browser tests (raised as an idea; not started).
- **Data deletion / retention** — user-initiated data deletion + a retention policy (privacy/compliance).

## Polish / housekeeping

- **README refresh** — bring it up to date with the current product (redesign, marketing site, chart types, exports, smarter analysis, admin console).
- **Dedicated OG image** — a purpose-built social/Open Graph image for the marketing site (currently relies on defaults).
