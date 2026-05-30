# SP2 — Accounts Design

**Date:** 2026-05-30
**Author:** Chris Silver (with Claude)
**Status:** Approved for implementation planning

## Context

ChartSage is live (SP1): anonymous visitors get 1 free report, gated by a `chartsage_anon` cookie, with reports in Supabase Postgres + CSVs in Supabase Storage and events in PostHog. SP2 adds real accounts so logged-in users get unlimited reports, the "Generate 5 more" action (currently ungated for everyone) becomes an account feature with an upsell for anonymous users, and people can save and revisit their reports.

This is sub-project 2 of 3. SP3 adds the credit system (metering generate-more, paid tiers).

## Goals

- Sign in with **Google or magic link** (Supabase Auth).
- Logged-in users: **unlimited new reports** and **Generate-more** allowed.
- Anonymous users: keep the **1 free report** gate; clicking **Generate 5 more** → an **upsell modal** prompting account creation.
- On signup/login, **migrate the anonymous visitor's existing report(s)** into their new account.
- A **My Reports** page so logged-in users can revisit saved reports.
- An **onboarding screen** on first login showing what an account unlocks.
- Row-Level Security enabled on `reports` (the hardening SP1 deferred).
- PostHog stitches anonymous → account identity so the signup funnel is measurable.
- All 162 existing tests stay green.

## Non-goals (SP2)

- Credits / metering / paid tiers (SP3).
- Stripe / payments (SP4).
- Team/org accounts, sharing, collaboration.
- Profile editing, avatars beyond what the provider gives.
- Email/password auth (Google + magic link only).
- Deleting the `UNLIMITED_ANON_IDS` allowlist (harmless; removed in SP3 cleanup).

## Auth model & architecture

The new mechanic: the FastAPI backend must recognise logged-in users. Today every call carries `X-Anon-Id`. After SP2, authenticated calls also carry a **Supabase access token (JWT)**; the backend verifies it.

```
Browser (Supabase JS client holds the session)
   │  Authorization: Bearer <supabase JWT>   ← when logged in
   │  X-Anon-Id: <uuid>                       ← always (anon fallback)
   ▼
FastAPI (Cloud Run)
   deps.get_identity():
     1. Bearer token present → verify against Supabase JWKS (cached) →
        Identity(user_id=<uuid>, is_authenticated=True)
     2. else X-Anon-Id → Identity(anon_id=<uuid>, is_authenticated=False)
     3. else → 400
```

Decisions:

1. **Supabase Auth on the frontend** via `@supabase/supabase-js` + `@supabase/ssr` (session readable in middleware/server components). Uses the existing publishable key.
2. **Backend verifies the JWT itself** — fetch Supabase JWKS once, cache the public keys, validate signature + `exp` + audience locally. Stateless; no per-request Supabase call. New Supabase projects issue asymmetric (ES256) tokens with a public JWKS at `${SUPABASE_URL}/auth/v1/.well-known/jwks.json`, so verification needs no new secret. (Safeguard: the first implementation task confirms the project's tokens are asymmetric; if it still issues symmetric HS256 tokens, fall back to verifying with the Supabase JWT secret stored as a Cloud Run secret.)
3. **`get_identity` replaces `get_anon_id`** on gated routes. Returns `Identity(user_id?, anon_id?, is_authenticated)`.
4. **Gating collapses to:** authenticated → unlimited reports + generate-more, reports owned by `user_id`; anonymous → 1-report gate + generate-more returns `402`.
5. **RLS enabled** on `reports` as defense-in-depth (backend keeps using the service-role key, which bypasses RLS).

## Data model, migration & RLS

**No new columns** — `reports.user_id` (nullable) already exists.

### Row-Level Security
```sql
ALTER TABLE reports ENABLE ROW LEVEL SECURITY;

CREATE POLICY reports_owner_select ON reports
  FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY reports_owner_update ON reports
  FOR UPDATE USING (auth.uid() = user_id);
```
The backend uses the service-role key (bypasses RLS) so existing flows are unaffected. RLS only constrains direct browser access via the publishable key — guaranteeing one user can't read another's reports. Anon rows (`user_id IS NULL`) remain backend-only.

### Anon → account migration
On signup/login the frontend calls:
```
POST /claim-anon-reports
   Authorization: Bearer <jwt>,  X-Anon-Id: <anon cookie>
Backend:
   UPDATE reports SET user_id = <jwt.user_id>, anon_id = NULL
     WHERE anon_id = <X-Anon-Id> AND user_id IS NULL
   → { claimed: <n> }
```
Idempotent. Runs once in the auth callback. The free report a visitor just made becomes the first report in their account.

### Onboarding state
A `chartsage_onboarded` flag in `localStorage` (no DB table). The auth callback always redirects to `/welcome`; `/welcome` is a client component that redirects straight to `/` if the flag is already set, otherwise shows onboarding and sets the flag on completion. (localStorage isn't readable in the server-side callback route, so the first-visit gate lives client-side in `/welcome`.) SP3 introduces a real `profiles`/credits table when per-user economics arrive.

### Report data path
All report reads/writes stay behind the backend (service-role). **My Reports** is a backend endpoint, not a direct Supabase query. RLS is the safety net, not the primary gate.

## Components

### Backend (`src/api/`)

**New:**
- `auth.py` — Supabase JWT verification. Fetch + cache JWKS, validate signature/`exp`/audience. `verify_token(bearer, _jwks=None) -> user_id | None` (the `_jwks` injection point lets tests supply a test keypair).
- `deps.py` (extend) — `get_identity()` returns `Identity(user_id?, anon_id?, is_authenticated)`; reads `Authorization: Bearer` first, falls back to `X-Anon-Id`, else 400.

**Modified `main.py`:**
- `/generate-report` — authenticated → skip anon cap, save with `user_id`; anon → existing 1-report gate.
- `/generate-more` — authenticated → allowed; anon → `402 { code: "UPGRADE_REQUIRED" }`.
- `db.save_report` / `db.get_report` — populate `user_id` for logged-in users.
- **New** `POST /claim-anon-reports` (auth) — migration above.
- **New** `GET /my-reports` (auth) — `[{id, title, chartCount, createdAt}]`, newest first, for `user_id`.

### Frontend (`src/app/`)

**New:**
- `lib/supabase.ts` — browser Supabase client (`@supabase/ssr`); exposes `getAccessToken()`.
- `login/page.tsx` — "Continue with Google" + magic-link email field (stone/teal).
- `auth/callback/route.ts` — exchange code → session; call `/claim-anon-reports`; route to `/welcome` (first time) or `/`.
- `welcome/page.tsx` — onboarding: three unlocks (unlimited reports, Generate-more, saved reports); sets `chartsage_onboarded`; CTA → `/` or `/reports`.
- `reports/page.tsx` — My Reports list (title + date + kind badges), each → `/report/[id]`.
- `components/UpsellModal.tsx` — shown on anon `402` from generate-more; inline Google / magic-link or link to `/login`.
- `components/AuthNav.tsx` — signed-out → "Sign in"; signed-in → "My reports" + email + "Sign out".

**Modified:**
- `lib/api.ts` — `apiFetch` injects `Authorization: Bearer <token>` when a session exists (plus `X-Anon-Id` always); awaits session hydration before firing.
- `report/[id]/Toolbar.tsx` — on `402` from generate-more, open `<UpsellModal>` instead of erroring.
- `anon-limit/page.tsx` — "Sign in" button becomes a real link to `/login`.
- `layout.tsx` — mounts `<AuthNav>` + Supabase session provider.

### Dependencies
- Frontend: `@supabase/supabase-js`, `@supabase/ssr`.
- Backend: pin `pyjwt[crypto]` for JWKS verification.

## Data flow

### Sign up / log in
```
Anon (chartsage_anon cookie, maybe 1 report) → /login → Google or magic link
  → Supabase auth → redirect /auth/callback?code=...
       1. exchangeCodeForSession(code) → session stored
       2. POST /claim-anon-reports (Bearer + X-Anon-Id) → claims anon's reports
       3. redirect → /welcome (which client-side redirects to / if already onboarded)
  → PostHog: alias(anon_id → user_id); capture signed_up | logged_in
```

### Onboarding (`/welcome`, first authed visit)
On mount, redirects to `/` if `chartsage_onboarded` is already set. Otherwise: static screen, three unlocks, one CTA; sets `chartsage_onboarded` on completion. Events `onboarding_viewed` / `onboarding_completed`.

### Generate-more upsell
```
ANON clicks "Generate 5 more"
  → POST /report/{id}/generate-more (X-Anon-Id only)
  → backend not authenticated → 402 { code: "UPGRADE_REQUIRED" }
  → Toolbar opens <UpsellModal> → auth flow → returns to this report (now owned)
  → PostHog: generate_more_upsell_shown, then signed_up if converts

LOGGED-IN clicks "Generate 5 more"
  → Bearer token → backend allows → charts append (as today)
```

### My Reports (`/reports`)
```
GET /my-reports (Bearer)
  → SELECT id, title, (chart count), created_at WHERE user_id=<uid> ORDER BY created_at DESC
  → list; click → /report/{id}
```

### Token on every call
`apiFetch` adds `Authorization: Bearer <access_token>` when a Supabase session exists (auto-refreshed by the client), always keeps `X-Anon-Id`. Existing endpoints (PATCH layout, export.pdf, get report) become user-aware transparently.

## Error handling

| Where | Case | Handling |
|---|---|---|
| JWT verify | Expired | `401`; client auto-refreshes + retries; refresh fail → `/login` |
| JWT verify | Bad signature / wrong audience | `401 INVALID_TOKEN` |
| JWKS fetch | Supabase unreachable (cold) | Fail closed `401`; cached after first success |
| claim-anon-reports | No anon cookie / nothing to claim | No-op `{claimed: 0}` |
| claim-anon-reports | Called twice | Idempotent |
| Magic link | Expired / used | `/login` shows "link expired, request a new one" |
| OAuth | User cancels | Callback error → `/login` soft message |
| Same email two providers | Google + magic link | Supabase links to one identity by default |
| Token not hydrated | First paint | `apiFetch` awaits session; until then treated as anon (safe) |
| generate-more | Anonymous | `402 UPGRADE_REQUIRED` → upsell (expected) |

## Testing

**Backend unit:**
- `auth.verify_token`: valid / expired / tampered / wrong-audience, using an injected **test RSA keypair** (no real Supabase).
- `get_identity`: Bearer → authenticated; X-Anon-Id only → anon; neither → 400.

**Backend integration** (with a `FakeAuth` helper injecting an identity):
- authenticated `generate-report` × 3 → all succeed, saved with `user_id`.
- `generate-more`: authenticated → 200; anon → `402 UPGRADE_REQUIRED`.
- `claim-anon-reports` → moves anon rows to user; second call claims 0.
- `my-reports` → seeds two users, asserts each sees only their own.

**Frontend:** manual smoke per project norm.

All 162 existing tests stay green (the `get_anon_id`→`get_identity` swap preserves anon behaviour).

## Rollout

**User-provisioned (one-time runbook, provided at execution):**
1. Supabase → Auth → Providers → enable **Google** (Google OAuth client ID/secret from the existing Google Cloud project → APIs & Services → Credentials). Magic link needs no setup.
2. Supabase → Auth → URL config: Site URL + redirect allow-list → `https://chartsage-xi.vercel.app/auth/callback` (+ `http://localhost:3000/auth/callback` for dev).
3. Run the RLS SQL (above) in the SQL editor.

**Deployed by Claude:** backend → Cloud Run (`gcloud builds submit`); frontend → Vercel (auto on push). No new secrets (JWKS is public; Google client secret lives in Supabase).

## File structure

### New files
```
src/api/auth.py
tests/unit/test_auth.py
tests/helpers/fake_auth.py
src/app/lib/supabase.ts
src/app/login/page.tsx
src/app/auth/callback/route.ts
src/app/welcome/page.tsx
src/app/reports/page.tsx
src/app/components/UpsellModal.tsx
src/app/components/AuthNav.tsx
tests/integration/test_auth_gating.py
tests/integration/test_claim_and_reports.py
```

### Modified files
```
src/api/deps.py            # get_identity
src/api/main.py            # gating via identity; /claim-anon-reports; /my-reports
src/api/db.py              # claim_anon_reports, list_user_reports helpers
src/app/lib/api.ts         # Bearer token injection
src/app/report/[id]/Toolbar.tsx   # 402 → UpsellModal
src/app/anon-limit/page.tsx       # real /login link
src/app/layout.tsx         # AuthNav + session provider
requirements.txt           # pin pyjwt[crypto]
package.json               # @supabase/supabase-js, @supabase/ssr
README.md                  # SP2 auth setup notes
```

## Open questions

None — all resolved across the five design sections.
