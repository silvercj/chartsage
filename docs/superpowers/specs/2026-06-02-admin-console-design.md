# ChartSage Admin Console (v1) — Design Spec

**Status:** Approved (design), pending spec review
**Date:** 2026-06-02
**Type:** Backend + frontend feature (internal tooling)

**Goal:** A minimal, secure internal console to find accounts, view a credit balance + recent credit history, and grant credits — the foundation for account management going forward. First concrete use: grant `cj.silver@me.com` 1000 credits.

**Context:** ChartSage = Next.js 14 (Vercel) + FastAPI (Cloud Run) + Supabase (Postgres/Auth) + PostHog. Existing primitives this reuses:
- `db.grant_credits(user_id, amount, reason, ref=None) -> new_balance` — atomic RPC (`grant_credits` SQL function): increments `profiles.credits_balance` and appends a `credit_transactions` row.
- `db.get_balance(user_id)`, `db.list_transactions(user_id, limit)`, `db.ensure_profile(user_id, grant_amount)`.
- The backend Supabase client uses the **service-role key**, which **bypasses RLS** — admin endpoints can read/write any account's balance and transactions, and can call the GoTrue admin API (`auth.admin.list_users()`).
- `Identity` / `get_identity` (deps.py) resolve a Supabase JWT or anon id. Email lives in `auth.users`, not `profiles`.

**Approved decisions:** admin auth = **shared admin token** (server-checked `X-Admin-Token` header); console UI = **gated `/admin` route** inside the existing app.

---

## Auth — shared admin token (server-enforced)

- New secret **`ADMIN_API_TOKEN`** (Google Secret Manager → Cloud Run env; never committed, never sent to the frontend bundle).
- A FastAPI dependency **`require_admin`** reads the `X-Admin-Token` header and compares it to `ADMIN_API_TOKEN` using a constant-time compare (`hmac.compare_digest`).
- **Fail-closed:** if `ADMIN_API_TOKEN` is unset/empty in the environment, OR the header is missing or mismatched → `403 {code: "FORBIDDEN"}`. There is no fallback path.
- The token is never logged. All admin traffic is over HTTPS (Cloud Run).
- **Accepted limitation:** a shared token has no per-admin attribution. Acceptable for a single-admin internal tool; upgradeable later to an email allowlist (the Supabase JWT already carries an `email` claim).

---

## Backend — endpoints under `/admin` (all behind `require_admin`)

1. **`GET /admin/accounts?q=<email substring>&limit=50`** — search accounts.
   - Resolve users via `supabase.auth.admin.list_users()` (service-role, paginated up to a sane cap), filter by case-insensitive email substring (empty `q` → first `limit`), and join `profiles` for each user's `credits_balance` (0 if no profile row yet).
   - Returns `[{user_id, email, credits_balance, created_at}]`.
2. **`GET /admin/accounts/{user_id}`** — account detail.
   - `{user_id, email, credits_balance, transactions: [{delta, reason, ref, created_at}]}` via `get_balance` + `list_transactions` (+ email from the admin API). `404` if the user does not exist.
3. **`POST /admin/accounts/{user_id}/grant`** — grant credits.
   - Body `{amount: int, reason?: str}`. Validate `1 ≤ amount ≤ 100000` (else `422 INVALID_AMOUNT`); `404` if the user does not exist.
   - Ensure the `profiles` row exists (so `grant_credits` updates a real row), then call `grant_credits(user_id, amount, reason or "admin_grant", ref="admin")`.
   - Fire the `admin_credit_grant` analytics event (below). Returns `{user_id, credits_balance}`.

---

## Frontend — gated `/admin` route

- A **token gate:** an input to paste the admin token, stored in `sessionStorage` (per-tab, never committed). A small **`adminFetch`** helper sends it as the `X-Admin-Token` header — it does **not** use the regular `apiFetch` (which attaches the Supabase Bearer).
- **Flow:** search box → results list (email + balance) → click a row → detail panel (balance + recent transactions) + a **grant form** (amount + optional reason) → submit → success toast + refresh the balance/transactions.
- **States:** `403` → "Enter a valid admin token" prompt; empty results → "No accounts match"; in-flight spinners on search/grant.
- Dark theme, semantic design-system classes (no hex). The page is marked `noindex` (robots) so it is not crawled. Reachable in prod, but inert without a valid token (the backend is the boundary).

---

## Analytics

- Every grant fires **`admin_credit_grant`** — `distinct_id = target user_id`; properties `{amount, newBalance, reason, targetEmail, source: "admin_console"}` (camelCase, no underscore leakage, matching the existing event convention).
- This is in addition to the durable `credit_transactions` ledger row that `grant_credits` already writes (reason `admin_grant`). Two records: one analytical (PostHog), one financial (ledger).

---

## Bootstrap — granting the first 1000 credits

1. Generate a strong random `ADMIN_API_TOKEN` (e.g., `openssl rand -hex 32`).
2. Store it in Secret Manager (`admin-api-token`) and wire it into Cloud Run (`cloudbuild.yaml --set-secrets` adds `ADMIN_API_TOKEN=admin-api-token:latest`).
3. Resolve `cj.silver@me.com` → `user_id` (the search endpoint / `list_users`).
4. `POST /admin/accounts/{user_id}/grant {amount: 1000}` with the `X-Admin-Token` header — run server-side without printing the token.
5. The user retrieves the token from Secret Manager (`gcloud secrets versions access latest --secret=admin-api-token`) to use the console UI.

---

## Security

- Strong random token; Secret Manager only (never git, never the frontend bundle).
- Constant-time comparison; fail-closed when the env var is unset.
- Token never logged; HTTPS only.
- Service-role reads/writes are confined to the three admin endpoints behind `require_admin`.
- `/admin` is reachable but useless without the token.

---

## Verification

- **Backend (TDD):** `require_admin` → `403` with no token, `403` with a wrong token, pass with the correct token; grant happy-path (balance increases by `amount`, `admin_credit_grant` fired, ledger row written); amount validation (`422` for `0`, negative, > max); unknown user (`404`); search filters by email substring and returns balances.
- **Frontend:** `tsc --noEmit` + `next build` clean; manual token-gate → search → click → grant → balance refresh.
- **Live (post-deploy):** set the secret, grant 1000 to `cj.silver@me.com`, confirm the balance + the `admin_credit_grant` event + the ledger row; confirm a wrong/absent token is rejected.

---

## Scope & non-goals

**In scope (v1):** the shared-token `require_admin` guard; the three admin endpoints (search / detail / grant); the gated `/admin` UI (search → view → grant); the `admin_credit_grant` event; the secret wiring + the bootstrap grant.

**Non-goals:**
- Deduct/refund UI, account suspension/ban, role management, plan changes.
- A dedicated audit-log viewer (the `credit_transactions` ledger + PostHog already record every grant).
- Bulk operations, CSV import of grants, scheduled grants.
- Real pagination/scale tuning of search (v1 uses `list_users()` + a cap; a SQL search RPC is a future optimization).
- Per-admin attribution (deferred with the shared-token choice).
- Changes to the public credit prices, the signup grant, or any user-facing credit flow.

---

## Phasing (plan order)

1. **Backend** — `require_admin` (+ `ADMIN_API_TOKEN` config), the three `/admin` endpoints, the `admin_credit_grant` event; TDD. Plus `cloudbuild.yaml` secret wiring.
2. **Frontend** — the `/admin` route: token gate + `adminFetch`, search, detail, grant form.
3. **Deploy + bootstrap** — set the secret, deploy backend (Cloud Run) + frontend (Vercel), grant `cj.silver@me.com` 1000 credits, verify.
