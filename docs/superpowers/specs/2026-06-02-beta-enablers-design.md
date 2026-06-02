# ChartSage Beta Enablers — Design Spec

**Status:** Approved (design), pending spec review
**Date:** 2026-06-02
**Type:** Backend + frontend feature (beta support tooling)

**Goal:** Give the free-beta traffic two things so the £25/week marketing PoC actually teaches us something: a way to **contact support** and a way to **leave feedback** on a report.

**Context:** ChartSage = Next.js 14 (Vercel) + FastAPI (Cloud Run) + Supabase + PostHog (`posthog.capture` wired front + back). No custom domain yet (on `chartsage-xi.vercel.app`), so a `support@` inbox isn't live — hence a self-contained contact form rather than a `mailto`. The legal pages (`/terms`, `/privacy`) currently point deletion/contact requests at a placeholder `support@chartsage.app`. The marketing footer (`src/app/components/marketing/MarketingFooter.tsx`) has a link row. The report page is `src/app/report/[id]/page.tsx`. Supabase tables follow the manual-SQL migration pattern in `docs/migrations/`.

**Approved decisions:** support = **contact form** (domain-independent); feedback = **custom inline widget** (PostHog-backed).

> **Next sub-project (not designed here):** Payments (SP4).

---

## Facet 1 — Contact form

### Frontend — `/contact` page
A client component (`src/app/contact/page.tsx`), dark theme + semantic tokens. Fields: **email** (so we can reply) + **message** (required, multi-line) + a hidden **honeypot** input (`company`, visually hidden; bots fill it). On submit → `POST` via `apiFetch`; on success show a "Thanks — we'll get back to you" state; on error show an inline message. Basic client validation (message non-empty).

### Backend — `POST /contact`
Body `{ email: str|None, message: str, company: str|None }` (company = honeypot). Behavior:
- If `company` is non-empty → silently return 200 (drop as spam; don't store).
- Validate `message`: 1–4000 chars after strip (else 422 `INVALID_MESSAGE`); `email` optional, capped at 320 chars.
- Capture the caller's identity via the existing `get_identity` dependency (anon or authed — no auth required).
- Store a row in `support_messages` (`db.save_support_message(email, message, user_id, anon_id)`).
- Fire PostHog `support_request {hasEmail: bool, length: int}` (no message body in analytics — privacy).
- Return `{ ok: true }`.

### Migration — `docs/migrations/support.sql`
```sql
create table if not exists support_messages (
  id         uuid primary key default gen_random_uuid(),
  email      text,
  message    text not null,
  user_id    uuid,
  anon_id    uuid,
  created_at timestamptz not null default now()
);
create index if not exists support_messages_created_idx on support_messages (created_at desc);
alter table support_messages enable row level security;  -- service-role only; no client policies
```
Run manually in Supabase before deploy (like the other migrations). Read submissions via the Supabase table editor (an admin-console "Messages" view is a noted fast-follow).

### Wiring
- Add a **"Contact"** link (`/contact`) to `MarketingFooter.tsx` alongside Terms/Privacy.
- Update the `/terms` + `/privacy` deletion/contact references from `support@chartsage.app` to a link to **`/contact`** (since that placeholder inbox isn't live). Keep the "email" wording only if a real inbox exists; otherwise point at the form.

---

## Facet 2 — Feedback widget

A small inline component on the report page (`src/app/report/[id]/`, a new `ReportFeedback.tsx` rendered near the bottom of `page.tsx`): **"Was this report useful?"** with 👍 / 👎 buttons. Clicking a rating reveals an optional comment textarea + a **Send** button; submit → `posthog.capture('report_feedback', { rating: 'up' | 'down', comment, reportId })` → a "Thanks for the feedback!" state. **Frontend-only** — PostHog owns the data (the right home for ratings; no DB row, no endpoint). Dark theme, semantic tokens, unobtrusive. Persist a "already gave feedback for this report" flag in component state (and optionally `sessionStorage` keyed by reportId) so it doesn't nag after submission.

---

## Dependencies & schema
- **No new deps.** `support_messages` is the only schema addition (manual migration). Feedback adds no schema (PostHog event only).

## Phasing (plan order)
1. **Contact backend** — `support_messages` migration + `db.save_support_message` + `POST /contact` (validation, honeypot, identity, PostHog event). Backend TDD.
2. **Contact frontend** — `/contact` page + form + `MarketingFooter` link + legal-page reference updates. tsc.
3. **Feedback widget** — `ReportFeedback.tsx` + wire into the report page; PostHog event. tsc.
4. **Build + deploy** — full suite + `next build`; deploy backend (Cloud Run, via `CLOUDSDK_PYTHON=/opt/homebrew/opt/python@3.12/bin/python3.12 gcloud builds submit`, `_SUPABASE_URL` from `.env`) + run the `support.sql` migration in Supabase + merge/push frontend (Vercel). Smoke: submit a contact message (row appears + event) and a report feedback (event fires). Production deploy requires explicit user authorization.

## Scope & non-goals
**In scope:** the `/contact` form + endpoint + table + footer/legal links; the report feedback widget (PostHog).

**Non-goals:**
- A full helpdesk / ticketing system, email auto-replies, or an in-app admin UI for reading messages (the Supabase table + a future admin view suffice).
- NPS / survey campaigns, multi-question feedback.
- A real `support@` inbox / `mailto` (deferred until a custom domain exists — the form is the interim).
- Payments (the next sub-project).

## Verification
- **Backend (TDD):** `POST /contact` stores a `support_messages` row + fires `support_request` for a valid message; honeypot non-empty → 200 + NO row; empty/oversize message → 422; identity (anon/user) is recorded. FakeDB gains `save_support_message` + an in-memory list. Full `pytest` green.
- **Frontend:** `tsc` + `next build` clean; `/contact` renders + submits; the footer + legal links resolve; the report feedback widget renders, submits a `report_feedback` event, and shows the thanks state.
- **Live (post-deploy):** run the migration; submit a contact message → confirm a `support_messages` row + the `support_request` event; rate a report 👍/👎 + comment → confirm the `report_feedback` event in PostHog.
