# ChartSage Payments (SP4) — Design Spec

**Status:** Approved (design), pending spec review
**Date:** 2026-06-02
**Type:** Backend + frontend feature (Stripe payments → credit top-ups)

**Goal:** Let signed-in users **buy credit packs** with a card via **hosted Stripe Checkout**, crediting their account automatically through a signature-verified, idempotent webhook. This converts the existing "out of credits → notify me" placeholder into a real self-serve purchase, monetising the £25/week marketing PoC traffic.

**Context:** ChartSage already has a complete credit economy (SP3): `profiles` + a `credit_transactions` ledger, with atomic Postgres functions `grant_credits` / `spend_credits` / `ensure_profile`. The admin console already calls `grant_credits` for manual top-ups. Today, when a user runs out, `OutOfCreditsModal` and `/credits` only capture *intent* (`POST /upgrade-intent` → `upgrade_intent` table). Stripe is greenfield (not in `requirements.txt`, no code references). Reference costs: report = 100 credits, deep analysis = 250, generate-more = 40, add-chart = 20, signup grant = 300 (≈3 free reports). Backend = FastAPI on Cloud Run (`chartsage-backend-112026133429.us-central1.run.app`); frontend = Next.js on Vercel (`chartsage.app`); Supabase Postgres. Secrets are backend-only (Secret Manager → Cloud Run via `cloudbuild.yaml --set-secrets`). Deploy uses `CLOUDSDK_PYTHON=/opt/homebrew/opt/python@3.12/bin/python3.12` for gcloud.

**Approved decisions:**
- **Model:** one-time **3-tier credit packs** (no subscription).
- **Pricing (GBP base):** `Starter £5 → 600 credits` · `Standard £15 → 2,000 credits` (best value) · `Pro £40 → 6,000 credits`.
- **International:** credits are **currency-decoupled** (granted from server-side config, never derived from the amount paid), so packs credit identically in any currency. Local-currency display via **Stripe Adaptive Pricing** (Dashboard toggle, no code). Hand-picked per-currency prices (`currency_options`) are a deferred fast-follow.
- **Integration:** **hosted Stripe Checkout** (`mode=payment`) — Stripe owns the card UI, wallets, 3DS/SCA, receipts, Adaptive Pricing, and fraud (Radar). **No Stripe key on the frontend** (redirect to the session URL).
- **Rollout:** build + smoke in **test mode** first, then swap to live keys.

> **Prior sub-projects:** SP1 foundation, SP2 accounts, SP3 credits, plus marketing/redesign/chart/export/smarter-analysis/admin/soft-launch/multi-value/beta-enablers. This is the monetisation layer on top of SP3.

---

## Facet 1 — Packages (server-side source of truth)

A new `src/api/billing.py` holds the pack catalogue — **credits live here; prices live in Stripe**:

```python
PACKAGES = {
    "starter":  {"credits": 600,  "label": "Starter",  "price_env": "STRIPE_PRICE_STARTER",  "gbp": 5},
    "standard": {"credits": 2000, "label": "Standard", "price_env": "STRIPE_PRICE_STANDARD", "gbp": 15},
    "pro":      {"credits": 6000, "label": "Pro",      "price_env": "STRIPE_PRICE_PRO",      "gbp": 40},
}
```

- The Stripe **Price ID** for each pack is read from an env var (`STRIPE_PRICE_*`; Price IDs are not secret → plain `--set-env-vars`).
- `gbp` is a display hint only (the authoritative amount lives in Stripe; Adaptive Pricing may localise it). The client **never** sends an amount — only a `package_id`.
- Helper `get_package(package_id) -> dict | None` for validation (unknown id → 400).

## Facet 2 — Backend endpoints

Three endpoints in `main.py`, following existing patterns (`get_identity`, `get_db`, `get_posthog` providers; `ContactIn`-style Pydantic bodies).

### `GET /billing/packages`
Public. Returns the catalogue for the frontend to render: `[{id, label, credits, gbp}]`. Single source of truth — no duplicated pricing in the frontend.

### `POST /billing/checkout` — body `{ package_id: str }`
**Auth required** (a purchase must credit a real account). Behavior:
- `get_identity`; if anonymous (no `user_id`) → 401 `{code:"AUTH_REQUIRED"}`.
- `get_package(package_id)`; unknown → 400 `{code:"UNKNOWN_PACKAGE"}`.
- Create a Stripe Checkout Session:
  - `mode="payment"`, `line_items=[{price: <pack price id>, quantity: 1}]`
  - `client_reference_id=<user_id>`
  - `metadata={user_id, credits, package_id}` (credits resolved **server-side** from `PACKAGES`)
  - `success_url=f"{FRONTEND_BASE_URL}/credits?purchase=success&session_id={{CHECKOUT_SESSION_ID}}"`
  - `cancel_url=f"{FRONTEND_BASE_URL}/credits?purchase=cancelled"`
- Fire PostHog `checkout_started {package_id, credits}`.
- Return `{ url: session.url }`.

### `POST /billing/webhook` — Stripe → us
**No auth; signature-verified.** Behavior:
- Read the **raw** request body (`await request.body()`) — do NOT use parsed JSON (signature is over the raw bytes).
- `stripe.Webhook.construct_event(payload, request.headers["stripe-signature"], STRIPE_WEBHOOK_SECRET)`. On `ValueError` (bad payload) or `SignatureVerificationError` → **400**.
- If `event["type"] == "checkout.session.completed"` and `session["payment_status"] == "paid"`:
  - Extract `user_id`, `credits` from `session["metadata"]`.
  - Call the **atomic, idempotent** `db.process_stripe_purchase(event_id, user_id, credits, session_id)` → `{granted, balance}` (Facet 3).
  - Fire PostHog `credits_purchased {package_id, credits, amount_total, currency}` (server-side) **only when `granted` is true** (so webhook replays don't double-count).
- Always return **200 `{received: true}`** for verified events (even ignored types/duplicates) so Stripe stops retrying. Only signature failures return 400.

## Facet 3 — Idempotency (exactly-once crediting)

Stripe delivers webhooks **at-least-once**, so the same `checkout.session.completed` can arrive multiple times. Crediting must be exactly-once **and crash-safe** — so idempotency and the grant live in **one Postgres transaction**.

### Migration — `docs/migrations/payments.sql`
```sql
create table if not exists stripe_events (
  event_id   text primary key,
  created_at timestamptz not null default now()
);
alter table stripe_events enable row level security;  -- service-role only; no client policies

-- Atomic: record the event and grant in a single transaction. Returns
-- {granted, balance}. On a duplicate event_id it grants nothing (granted=false)
-- and returns the current balance, so the webhook fires analytics exactly once.
-- Reuses grant_credits() for the ledger entry (reason 'stripe_purchase',
-- ref = checkout session id).
create or replace function process_stripe_purchase(
  p_event text, p_user uuid, p_credits int, p_ref text
) returns jsonb language plpgsql as $$
declare new_balance int; n_inserted int;
begin
  insert into stripe_events(event_id) values (p_event)
    on conflict (event_id) do nothing;
  get diagnostics n_inserted = row_count;     -- 1 = new, 0 = duplicate
  if n_inserted = 0 then
    select credits_balance into new_balance from profiles where user_id = p_user;
    return jsonb_build_object('granted', false, 'balance', coalesce(new_balance, 0));
  end if;
  insert into profiles(user_id, credits_balance) values (p_user, 0)
    on conflict (user_id) do nothing;          -- safety: profile must exist to credit
  perform grant_credits(p_user, p_credits, 'stripe_purchase', p_ref);
  select credits_balance into new_balance from profiles where user_id = p_user;
  return jsonb_build_object('granted', true, 'balance', new_balance);
end; $$;
```
Run manually in Supabase before deploy (like the other migrations).

### `db.py`
- `process_stripe_purchase(event_id, user_id, credits, session_id) -> dict` → `self.client.rpc("process_stripe_purchase", {...}).execute()`, returning `{"granted": bool, "balance": int}` (mirrors the existing `grant_credits` RPC wrapper).
- **FakeDB:** an in-memory `set` of seen `event_id`s; first sight grants into the fake balance (via the existing fake `grant_credits`), records the id, returns `{granted: true, balance}`; repeat sight is a no-op returning `{granted: false, balance}`.

The ledger row carries `reason='stripe_purchase'`, `ref=<session_id>`, so purchases appear in `/credits` history and are traceable to a Stripe session.

## Facet 4 — Frontend

No new Stripe dependency or key (hosted Checkout = redirect only).

- **`/credits` page (`src/app/credits/page.tsx`):** replace the "notify me" CTA with a **three pack cards** row fetched from `GET /billing/packages` (label, credits, "£X", "best value" badge on Standard). Click → `POST /billing/checkout {package_id}` → `window.location.href = url`. Fire `buy_pack_clicked {package_id}`.
- **Return handling:** on `/credits?purchase=success`, show a "Payment received — adding your credits…" banner and **refresh the balance, polling 2–3× over a few seconds** to cover brief webhook lag (the balance updates when the webhook lands). `?purchase=cancelled` → a quiet "Checkout cancelled" note.
- **`OutOfCreditsModal` (`src/app/components/OutOfCreditsModal.tsx`):** swap the primary "Notify me" button for **"Buy credits"** → routes to `/credits` (where the pack cards + purchase flow live). Keep a secondary "Maybe later"/close. (Single purchase surface = `/credits`; the modal funnels there.)
- Anonymous users never reach these (they get `UpsellModal` → sign in first); the buy path is auth-gated end-to-end.

## Dependencies, secrets & provisioning

- **Backend dep:** add `stripe` to `requirements.txt`.
- **Secrets (Secret Manager → `cloudbuild.yaml --set-secrets`, backend-only):** `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`.
- **Env vars (`--set-env-vars`, not secret):** `STRIPE_PRICE_STARTER`, `STRIPE_PRICE_STANDARD`, `STRIPE_PRICE_PRO`. The redirect base reuses the existing `FRONTEND_BASE_URL` (already `https://chartsage.app`).
- **No frontend secret** — hosted Checkout needs no publishable key.
- **User provisioning runbook (test mode first):**
  1. In Stripe **test mode**, create 3 Products with one-time GBP Prices (£5/£15/£40); copy the `price_…` IDs.
  2. Register a webhook endpoint → `https://<backend>/billing/webhook`, event `checkout.session.completed`; copy the signing secret (`whsec_…`).
  3. Put `STRIPE_SECRET_KEY` (test `sk_test_…`) + `STRIPE_WEBHOOK_SECRET` into Secret Manager; set the 3 `STRIPE_PRICE_*` env vars.
  4. (Optional) Toggle **Adaptive Pricing** on for local-currency display.
  5. Deploy; smoke a test-mode purchase with card `4242 4242 4242 4242`.
  6. Repeat 1–3 with **live** keys/prices and redeploy to go live.

## Phasing (plan order)
1. **Package config + `/billing/packages`** — `billing.py` + the public catalogue endpoint. TDD.
2. **Idempotency layer** — `payments.sql` migration + `db.process_stripe_purchase` + FakeDB. TDD (new event grants; duplicate event no-ops).
3. **Checkout endpoint** — `POST /billing/checkout` (auth-gate, package lookup, session creation with mocked Stripe, `checkout_started`). TDD.
4. **Webhook endpoint** — `POST /billing/webhook` (raw body, signature verify, paid-session → `process_stripe_purchase`, `credits_purchased`; bad signature → 400; replay → no double-grant; irrelevant type → 200 ignore). TDD with mocked `construct_event`.
5. **Frontend** — `/credits` pack cards + purchase redirect + success/cancel return handling + balance poll; `OutOfCreditsModal` "Buy credits". `tsc`.
6. **Deps + cloudbuild wiring** — `stripe` in requirements; `--set-secrets`/`--set-env-vars` additions.
7. **Build + QA + deploy (test mode) + smoke** — full `pytest` + `next build`; run `payments.sql` in Supabase; deploy backend (Cloud Run, `CLOUDSDK_PYTHON=3.12`) + frontend (Vercel); live test-mode purchase → credits land; replay the webhook (Stripe CLI/dashboard resend) → balance unchanged. **Production deploy + live-key swap require explicit user authorization.**

## Scope & non-goals
**In scope:** the 3 endpoints; `billing.py` catalogue; idempotent crediting (`stripe_events` + `process_stripe_purchase`); pack-buying UI on `/credits` + the modal funnel; secrets/env wiring; tests + a test-mode live smoke.

**Non-goals (v1):**
- **Subscriptions / recurring billing** (cancellation, dunning, proration) — packs only.
- **In-app refunds** — handled via the Stripe dashboard + an admin `grant_credits` adjustment if needed.
- **Invoices / VAT / tax automation** — deferred until VAT-registered (Adaptive Pricing ≠ tax). Flagged in `docs/FUTURE-IMPROVEMENTS.md`.
- **Hand-picked per-currency prices** (`currency_options`) — Adaptive Pricing covers v1; revisit for round local numbers.
- **Anonymous purchase** — sign-in required (no account to credit otherwise).
- **Storing a Stripe customer** on the profile — YAGNI for one-time packs.

## Verification
- **TDD (backend):**
  - `billing.get_package` returns/None correctly; `/billing/packages` returns the 3 packs.
  - `/billing/checkout`: anonymous → 401; unknown package → 400; valid → calls Stripe with the right price + `metadata.credits` (resolved server-side) + `client_reference_id`, returns the session URL (Stripe mocked); `checkout_started` fired.
  - `process_stripe_purchase`: a new event grants the pack's credits + writes a `stripe_purchase` ledger row; the **same event id again grants nothing** (balance unchanged).
  - `/billing/webhook`: bad signature → 400; `checkout.session.completed`+paid → credits granted + `credits_purchased` fired; **replayed event → no double-grant**; non-checkout event → 200 + ignored.
  - `FakeDB` gains `process_stripe_purchase` + the seen-events set.
- **Full `pytest` green; `tsc` + `next build` clean.**
- **Live (test mode, post-deploy):** complete a Checkout with `4242…` → redirected to `/credits?purchase=success`, balance increases by the pack amount, a "Standard"/etc. row appears in history. Resend the same webhook event from the Stripe dashboard → balance does **not** change again.
