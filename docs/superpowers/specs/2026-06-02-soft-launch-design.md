# ChartSage Soft-Launch Readiness (v1) — Design Spec

**Status:** Approved (design), pending spec review
**Date:** 2026-06-02
**Type:** Backend + frontend + infra (launch guardrails)

**Goal:** The minimum guardrails to safely open ChartSage to a **free public beta** — bound the cost exposure of the free tier, cover the legal basics for ingesting user data, and get visibility into failures. Payments stay deferred (SP4).

**Context:** Next.js 14 (Vercel) + FastAPI (Cloud Run) + Supabase (auth + Postgres) + PostHog + Anthropic (Haiku). Auth + the credit economy are live (signup grant 300, report 100, generate-more 40, add-chart 20, deep 250). Today the anonymous free-report cap keys on a **client-supplied `X-Anon-Id` UUID** (`db.count_anon_reports(anon_id)` in `main.py`), so rotating the UUID grants unlimited free reports → an open tap on the Anthropic bill. There is no error tracking (only PostHog `*_failed` events, no alerting) and no legal pages (`/terms`, `/privacy` don't exist). `report_generation_succeeded` already captures `estCostUsd`, so real cost-per-report is measurable for tuning the caps later.

**Approved decisions:** (1) **Keep** the anonymous-try funnel but add a **global daily hard-stop + per-IP daily cap + IP/fingerprint logging + alerting**; (2) **I draft** starting-point ToS/Privacy templates (not legal advice); (3) **Sentry** for error tracking (+ GCP budget alarm + Anthropic spend cap).

---

## Area 1 — Free-tier abuse / cost guard

Keep the 1-free-anonymous-report funnel; layer in real backstops so worst-case daily free spend is bounded and abuse is traceable.

**New table** (manual migration, in the `sp3-credits.sql` style — run once in the Supabase SQL editor):
```sql
create table if not exists anon_report_log (
  id          uuid primary key default gen_random_uuid(),
  anon_id     uuid,
  ip          text,
  fingerprint text,
  created_at  timestamptz not null default now()
);
create index if not exists anon_report_log_created_idx on anon_report_log (created_at desc);
create index if not exists anon_report_log_ip_idx on anon_report_log (ip, created_at desc);
alter table anon_report_log enable row level security;  -- service-role only; no client policies
```

**Capture (anonymous reports only):**
- **IP** — `X-Forwarded-For` first hop (the client IP on Cloud Run behind Google's LB).
- **Fingerprint** — a coarse server-side hash, `sha256(User-Agent + "|" + Accept-Language)` (first 16 hex chars). Enough to track patterns and make the per-IP cap meaningful without a client-side library. (A stronger client-side fingerprint, e.g. FingerprintJS, is a noted later enhancement, not v1.)

**Layered caps on the anonymous path** (authenticated users are untouched — they're financially bounded by their credit balance):
1. **Lifetime 1-per-`anon_id`** — unchanged (`db.count_anon_reports`).
2. **Per-IP daily cap** — `ANON_IP_DAILY_CAP` (default **5**): count `anon_report_log` rows for this IP since UTC midnight; at/over the cap → `429 {code:"RATE_LIMITED"}`. Rotating the UUID alone no longer bypasses; an abuser must also rotate IP.
3. **Global daily anonymous cap** — `ANON_GLOBAL_DAILY_CAP` (default **200**): count all `anon_report_log` rows since UTC midnight; at/over → `503 {code:"FREE_TIER_AT_CAPACITY", message:"The free tier is at capacity for today — sign in to keep going."}`. This is the **hard $ stop** on free Claude spend.

**Order:** the caps are checked **before** the Claude call (no spend on a blocked request). On a successful anonymous generation, insert one `anon_report_log` row (anon_id, ip, fingerprint).

**Alerting:** when cap 2 or 3 trips, fire `anon_cap_hit {scope:"ip"|"global"}` (PostHog) **and** a Sentry message (so it reaches the alert channel from Area 3).

**Tuning:** defaults are conservative starting points; revisit against the `estCostUsd` data once real traffic lands (200 free reports/day × cost-per-report = the bounded daily exposure).

**DB methods** (`db.py` + FakeDB): `log_anon_report(anon_id, ip, fingerprint)`; `count_anon_reports_today_by_ip(ip)`; `count_anon_reports_today()`.

**Privacy note:** IP + fingerprint are stored → must be disclosed in the Privacy Policy (Area 2).

---

## Area 2 — Legal pages + consent

Drafted **ChartSage-specific starting-point** content (clearly marked *"not legal advice — have a professional review before the paid launch"*):
- **`/terms`** — Terms of Service (server component, public/indexable).
- **`/privacy`** — Privacy Policy with a clear **Data handling & retention** section.
- **Footer links** in `MarketingFooter.tsx` (Terms · Privacy).
- **Consent line** on `/login`: "By continuing, you agree to our Terms and Privacy Policy." with links.

The Privacy Policy must disclose what is now collected/stored: uploaded **CSV data**; **AI processing by a third-party provider** (kept generic, per the established "reference AI, don't name the vendor" preference); generated **reports** + the **source CSV** in storage; **account email**; and the **IP + coarse fingerprint** captured for abuse prevention. It states retention and how to request deletion (**email request for now**, since self-serve deletion is a future item).

---

## Area 3 — Monitoring & alerting (Sentry)

- **Backend** (FastAPI/Cloud Run): add `sentry-sdk[fastapi]`; init from `SENTRY_DSN` (Secret Manager) at startup; captures unhandled exceptions with stack traces. Existing `*_failed` PostHog events stay. Init is a no-op when `SENTRY_DSN` is unset (local/dev safe).
- **Frontend** (Next/Vercel): add `@sentry/nextjs`; init from `NEXT_PUBLIC_SENTRY_DSN` (Vercel env) for client + server error capture. No-op when unset.
- **Infra alarms:** a **GCP billing budget + alert** on project `chartsage-497909` (e.g. thresholds at 50/90/100% of a monthly cap); an **Anthropic workspace spend limit** (Anthropic console).
- **Setup prerequisites (user):** create a Sentry project and provide the two DSNs; set the Anthropic spend cap. (Wiring + the GCP budget are handled in the build/deploy.)

---

## Config / env summary

- **Backend** (`main.py` / env): `ANON_IP_DAILY_CAP=5`, `ANON_GLOBAL_DAILY_CAP=200` (env-overridable, like the existing `ANON_REPORT_LIMIT`); `SENTRY_DSN` (Secret Manager). `cloudbuild.yaml`: add `SENTRY_DSN=sentry-dsn:latest` to `--set-secrets`, and the two cap env vars to `--set-env-vars`.
- **Frontend** (Vercel env): `NEXT_PUBLIC_SENTRY_DSN`.

## Dependencies & schema

- **New backend dep:** `sentry-sdk[fastapi]` (`requirements.txt`).
- **New frontend dep:** `@sentry/nextjs`.
- **Migration:** `docs/migrations/soft-launch.sql` (the `anon_report_log` table) — run manually in Supabase before the deploy.
- **Secrets/env:** `sentry-dsn` (Secret Manager) + `NEXT_PUBLIC_SENTRY_DSN` (Vercel) — created from the user-provided DSNs at deploy time.

## Phasing (plan order)
1. **Abuse guard** — migration + IP/fingerprint capture + the layered caps + DB methods + alerting hooks; TDD. Most urgent (the cost tap).
2. **Legal pages** — `/terms`, `/privacy`, footer links, login consent line; drafted content.
3. **Monitoring** — Sentry backend + frontend wiring; GCP budget alarm + Anthropic cap (deploy-time + user actions).

## Scope & non-goals
**In scope:** the layered anonymous caps + IP/fingerprint logging + alerting; the legal pages + consent; Sentry + infra alarms.

**Non-goals:**
- **Payments / credit purchase** (SP4) — the paid launch is separate.
- **Self-serve data deletion** — email-request for v1; a deletion UI is a future item.
- **Client-side fingerprinting** (FingerprintJS) — v1 uses a coarse server-side fingerprint.
- **Full Vercel WAF rule-set / captcha** — the daily caps + IP cap are the v1 backstop.
- **Per-user rate limiting on authenticated reports** — already bounded by the credit balance.
- **Changing prices or the credit economy.**

## Verification
- **Area 1 (TDD):** per-IP daily cap blocks at the threshold (429); global daily cap blocks at the threshold (503); both fire `anon_cap_hit`; an anonymous success logs an `anon_report_log` row with ip + fingerprint; authenticated generation is unaffected by all new caps; caps are checked before any Claude call. FakeDB gains the three methods + an in-memory log. Full `pytest` green.
- **Area 2:** `tsc` + `next build` clean; `/terms` + `/privacy` render; footer + login links resolve; the consent line is present.
- **Area 3:** backend boots with `SENTRY_DSN` unset (no-op) and initializes when set; `next build` clean with the Sentry SDK; a deliberately-thrown test error surfaces in Sentry post-deploy; the GCP budget alert exists.
- **Live (post-deploy):** run the migration; confirm an anonymous report logs IP/fingerprint; confirm the caps trip (lower them temporarily to test); confirm a forced error reaches Sentry; confirm the legal pages are public.
