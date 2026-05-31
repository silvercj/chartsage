# SP3 — Credits Design

**Date:** 2026-05-31
**Author:** Chris Silver (with Claude)
**Status:** Approved for implementation planning

## Context

ChartSage is live. SP1 shipped the serverless foundation (Supabase Postgres + Storage, PostHog, Cloud Run, Vercel) with an anonymous 1-free-report gate. SP2 shipped accounts (Supabase Auth, JWT-verified backend identity, anon→account report migration, My Reports, a logged-in nav) and gave logged-in users **unlimited** reports + generate-more. SP3 replaces "unlimited" with a **credit economy**: logged-in users get a starter grant, generating reports and extra charts draws it down, and running out surfaces an "out of credits" wall that captures upgrade intent.

This is sub-project 3 of 3 in the original roadmap. **Payments are out of scope** — buying credits (Stripe) is SP4. SP3's job is to stand up the credit accounting + metering + UX and **measure demand** (how many people hit the wall and ask to be notified) before payment rails are built.

## The economy (researched, psychologically tuned)

Credit-pricing research is clear that a **1:1 model is the wrong choice**: credits work precisely because they abstract money (chunky "game-chip" numbers dampen the pain-of-paying that a literal "3 → 2 → 1 reports" counter triggers), and variable per-action costs that track real cost/value are the norm in AI tools. So:

| Action | Cost | Rationale |
|---|---|---|
| New report | **100 credits** | Core, expensive action (full 2-pass, 10-chart pipeline) |
| Generate 5 more | **40 credits** | Lighter add-on (~1 Claude pass); clean 5:2 ratio tracking cost + value |
| Anonymous 1st report | **free** (unchanged) | Existing top-of-funnel hook |
| Signup starter grant | **300 credits** | ~3 reports — succeed a few times, then hit the wall *while engaged* (the demand signal) |

All three numbers are **env-tunable** (`REPORT_COST`, `GENERATE_MORE_COST`, `SIGNUP_GRANT`) so the economy retunes without a code change. No expiry, no rollover (simplest + friendly). Sources informing this: m3ter credit-pricing guide, schematichq AI-credits analysis, standard freemium-onboarding research.

## Goals

- Logged-in users get a one-time 300-credit starter grant on first login.
- Generating a report costs 100 credits; "Generate 5 more" costs 40; both gate when the balance is short.
- A credit **ledger** (append-only) is the source of truth; a cached balance powers fast reads.
- Credits are spent **atomically** and **only on success** (a failed generation never costs credits).
- A proper logged-in **app header** (Reports · Credits · account menu) with a live **balance pill** — consolidating the SP2 floating nav and the SP2.1 toolbar account-controls into one extensible shell.
- An **out-of-credits** modal that captures "notify me" intent (the SP3 demand signal); a **/credits** page with balance, costs, and history.
- Anonymous behaviour unchanged (1 free report; generate-more → create-account upsell).
- PostHog tracks the full economy funnel (grant, spend, out-of-credits, upgrade-intent).
- Existing test suite stays green (two SP2 "unlimited" tests updated to the credit reality).

## Non-goals (SP3)

- **Stripe / payments / self-serve purchase** (SP4) — "out of credits" captures intent only.
- Credit expiry, rollover, or refunds.
- Paid tiers / subscription plans.
- A full Account page (the header account menu — email + Sign out — is enough for now).
- Admin tooling for granting credits (do it via SQL/ledger insert if needed).
- Metering PDF export or report viewing (free).

## Section 1 — Data model & ledger

Two new tables plus a tiny intent table; `reports` is untouched.

```sql
-- Fast-read balance + per-user profile (created on first login)
create table profiles (
  user_id        uuid primary key references auth.users(id),
  credits_balance int not null default 0,
  created_at     timestamptz not null default now()
);

-- Append-only audit trail; source of truth for every credit movement
create table credit_transactions (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users(id),
  delta       int  not null,            -- +grant / −spend
  reason      text not null,            -- 'signup_grant' | 'report' | 'generate_more' | 'adjustment'
  ref         text,                     -- e.g. the report_id it paid for
  created_at  timestamptz not null default now()
);
create index credit_transactions_user_idx on credit_transactions (user_id, created_at desc);

-- "Notify me when paid plans launch" — the SP3 demand signal
create table upgrade_intent (
  user_id    uuid primary key references auth.users(id),
  email      text,
  created_at timestamptz not null default now()
);
```

**Atomic spend (race-safe):** the conditional `UPDATE` decrements only when the balance covers the cost, and the ledger insert records it — both in one transaction.

```sql
create or replace function spend_credits(p_user uuid, p_amount int, p_reason text, p_ref text default null)
returns int language plpgsql as $$
declare new_balance int;
begin
  update profiles set credits_balance = credits_balance - p_amount
    where user_id = p_user and credits_balance >= p_amount
    returning credits_balance into new_balance;
  if new_balance is null then
    raise exception 'INSUFFICIENT_CREDITS';
  end if;
  insert into credit_transactions(user_id, delta, reason, ref)
    values (p_user, -p_amount, p_reason, p_ref);
  return new_balance;
end; $$;

create or replace function grant_credits(p_user uuid, p_amount int, p_reason text, p_ref text default null)
returns int language plpgsql as $$
declare new_balance int;
begin
  update profiles set credits_balance = credits_balance + p_amount
    where user_id = p_user
    returning credits_balance into new_balance;
  insert into credit_transactions(user_id, delta, reason, ref)
    values (p_user, p_amount, p_reason, p_ref);
  return new_balance;
end; $$;

-- Idempotent first-login setup: create the row, grant the starter once.
create or replace function ensure_profile(p_user uuid, p_grant int)
returns int language plpgsql as $$
declare new_balance int;
begin
  insert into profiles(user_id, credits_balance) values (p_user, 0)
    on conflict (user_id) do nothing;
  if not exists (select 1 from credit_transactions
                 where user_id = p_user and reason = 'signup_grant') then
    perform grant_credits(p_user, p_grant, 'signup_grant', null);
  end if;
  select credits_balance into new_balance from profiles where user_id = p_user;
  return new_balance;
end; $$;
```

**RLS:** enable on all three tables; owner-only `SELECT` on `profiles` and `credit_transactions` (`auth.uid() = user_id`); `upgrade_intent` has no public policy (backend-only). All writes go through the backend's service-role key (bypasses RLS) calling the functions above — same model as SP2.

```sql
alter table profiles enable row level security;
alter table credit_transactions enable row level security;
alter table upgrade_intent enable row level security;
create policy profiles_owner_select on profiles for select using (auth.uid() = user_id);
create policy txns_owner_select on credit_transactions for select using (auth.uid() = user_id);
```

## Section 2 — Backend

**New `src/api/credits.py`** — service layer over the Supabase RPCs:
- `ensure_profile(user_id) -> balance` — calls `ensure_profile(user, SIGNUP_GRANT)`; idempotent.
- `get_balance(user_id) -> int`.
- `spend(user_id, amount, reason, ref)` — calls `spend_credits`; raises `InsufficientCredits` on the `INSUFFICIENT_CREDITS` error.
- `grant(user_id, amount, reason, ref)`.

**`src/api/db.py`** gains the thin Supabase calls these use: `get_profile`, `list_transactions(user_id, limit)`, `record_upgrade_intent(user_id, email)`, and `.rpc(...)` wrappers for `spend_credits` / `grant_credits` / `ensure_profile`. `FakeDB` mirrors all of them with in-memory atomic equivalents.

**Costs config** (env-tunable, defaults shown): `REPORT_COST=100`, `GENERATE_MORE_COST=40`, `SIGNUP_GRANT=300`.

**Gating in `main.py`** (anonymous paths unchanged):
- `POST /generate-report` — anon → existing 1-free cap; **authenticated** → `ensure_profile`, pre-check `balance ≥ REPORT_COST`; if short → `402 {code: "OUT_OF_CREDITS", balance, cost}` (no Claude call, no spend); **on success**, `spend(REPORT_COST, "report", ref=report_id)`.
- `POST /report/{id}/generate-more` — anon → `402 {code: "UPGRADE_REQUIRED"}` (unchanged); **authenticated** → pre-check `balance ≥ GENERATE_MORE_COST`; short → `402 OUT_OF_CREDITS`; on success → `spend(GENERATE_MORE_COST, "generate_more", ref=report_id)`.

The **two 402 codes are distinct** — `UPGRADE_REQUIRED` (anon → create account) vs `OUT_OF_CREDITS` (logged-in → out of credits) — so the frontend shows the correct modal.

**New endpoints** (all require auth → `401 AUTH_REQUIRED` otherwise):
- `GET /me` → `{credits_balance}` — tiny; the nav calls it on every page. Calls `ensure_profile` so the starter grant lands on first authenticated load.
- `GET /credits/history` → `[{delta, reason, ref, created_at}]`, newest first.
- `POST /upgrade-intent` (body `{email}`, the logged-in session's email; `user_id` comes from the token) → upsert `upgrade_intent(user_id, email)`; idempotent; fires PostHog.

**PostHog events:** `credits_granted {amount}`, `credits_spent {amount, balance, reason}`, `out_of_credits {action, balance}`, `upgrade_intent_captured`.

## Section 3 — Frontend

**App header (the logged-in nav shell).** A new **in-flow** top header for logged-in users: brand left; **Reports · Credits** nav; a **credits balance pill** + an **account menu** (email · Sign out) right. It **replaces** both the SP2 floating `AuthNav` and the SP2.1 toolbar account-controls. Because it's in normal flow (not a floating overlay), the report page's sticky toolbar sits *below* it (no overlap), and the toolbar reverts to just Generate/Export. It is **not rendered on `/report/[id]/print`**, so the exported PDF stays clean. It's the extensible spine for future menu items (Account, etc.).

**`CreditsBadge`** (fed by a `useCredits()` hook hitting `GET /me`): shows e.g. `⚡ 300`, links to `/credits`, turns amber when the balance can't afford a report (< `REPORT_COST`), and refetches after any spend.

**`/credits` page:** large balance, a plain-language cost table (Report = 100, Generate 5 more = 40 — the transparency the research insists on), the transaction history (`GET /credits/history`), and a "paid plans coming soon — notify me" CTA.

**Two modals:**
- Anonymous wall → existing `UpsellModal` ("create a free account").
- Logged-in `OUT_OF_CREDITS` → new `OutOfCreditsModal` ("You're out of credits — top-ups coming soon, notify me" → `POST /upgrade-intent`, then a confirmation).

**Transparency, not friction:** action buttons show the cost inline — "Generate report · 100" / "Generate 5 more · 40" — with the live pill; no blocking confirm dialog, no surprise burns.

**Copy rewrites:** onboarding (`/welcome`) "Unlimited reports" → "**300 credits to start** (~3 reports)"; the home upload CTA and any "unlimited" wording reflect credits.

**Account:** kept minimal for SP3 — the header account menu (email + Sign out). No separate Account page yet.

## Section 4 — Data flow

**First login → starter grant:** after the SP2 auth flow lands on `/reports`, the header mounts → `useCredits` → `GET /me` → `ensure_profile()` creates the profile + grants 300 → returns `300`. Idempotent. PostHog `credits_granted`.

**Report (balance ≥ 100):** upload → `POST /generate-report` → pre-check ≥100 → pipeline → save → `spend(100, "report", ref)` → `{session_id}`; pill refetches → 200. PostHog `credits_spent {100, 200}`.

**Report (balance < 100):** `POST /generate-report` → pre-check fails → `402 OUT_OF_CREDITS {balance, cost:100}` **before any Claude call or spend** → home shows `OutOfCreditsModal`. PostHog `out_of_credits {action:"report"}`.

**Generate 5 more:** "Generate 5 more · 40" → `POST …/generate-more` → ≥40 → append → `spend(40, "generate_more", ref)` → pill refetches. Short → `402 OUT_OF_CREDITS` → modal.

**Notify-me:** modal "Notify me" → `POST /upgrade-intent` → upsert + PostHog `upgrade_intent_captured` → modal confirms.

**Balance freshness:** `useCredits` refetches after every successful spend and on `/credits`, so the pill is always current.

**Anonymous (unchanged):** 1 free report (`403 ANON_LIMIT_REACHED` on the 2nd), generate-more → `402 UPGRADE_REQUIRED` → `UpsellModal`. No credits/profile until sign-in.

## Section 5 — Error handling, testing & rollout

### Error handling

| Where | Case | Handling |
|---|---|---|
| `ensure_profile` | parallel first-requests | `insert ... on conflict do nothing` + grant guarded by "no `signup_grant` row" → single grant |
| `spend` | balance < cost at debit (lost race after pre-check) | RPC raises `INSUFFICIENT_CREDITS`; report already made → serve it, log it (rare free slip, not a 500) |
| `spend` | DB error mid-debit | report saved but debit failed → serve report, log + PostHog `credit_spend_failed` (don't punish the user for our error) |
| `ensure_profile` | profile create fails | `503 "couldn't check credits, retry"` — fail safe (don't silently free or hard-block) |
| `/upgrade-intent` | called twice | upsert on `user_id` → idempotent |
| credit endpoints | anonymous caller | `401 AUTH_REQUIRED` (nav only calls `/me` with a session) |

### Testing

**Backend unit (`tests/unit/test_credits.py`):** `ensure_profile` grants exactly once (second call no double-grant); `spend` decrements + writes a ledger row; `spend` insufficient raises `InsufficientCredits`; `grant` adds. Exercised against `FakeDB`'s in-memory atomic spend/grant; the real Postgres functions are validated in the live smoke.

**Backend integration** (`tests/integration/test_credit_gating.py`, `test_credits_endpoints.py`, FakeDB + FakeAuth):
- authed report with 300 → succeeds, balance 200, ledger has `report −100`.
- report at balance < 100 → `402 OUT_OF_CREDITS`, **no report saved, no spend**.
- generate-more ≥40 → succeeds, −40; < 40 → `402 OUT_OF_CREDITS`.
- first `/me` → grants 300; second `/me` → still 300 (no double-grant).
- `/credits/history` → returns the caller's ledger, isolated by user.
- `/upgrade-intent` → records a row; second call idempotent.
- a generation that fails (Claude error) → **no debit** (balance unchanged).
- anon paths unchanged (1-free; generate-more `402 UPGRADE_REQUIRED`).

**SP2 test updates:** the two SP2 "authenticated = unlimited" tests now hit credit gating. `test_authenticated_user_unlimited_reports` (3 reports) still passes (300 ÷ 100 = exactly 3) but is **renamed** to reflect metering and gains a 4th-report → `402 OUT_OF_CREDITS` assertion; `test_authenticated_generate_more_allowed` keeps passing (40 of 300) and asserts the post-spend balance. Same pattern as SP2 updating the anon generate-more tests — the suite stays green, those two are updated to the new reality.

**Frontend:** manual smoke per project norm.

### Rollout

**You run one SQL migration** in Supabase (SQL editor) — the full Section 1 SQL: the three tables, the `spend_credits` / `grant_credits` / `ensure_profile` functions, and the RLS policies. (I'll hand you the exact block, like the SP2 RLS step.)

**No new secrets.** Costs are env vars with defaults (`REPORT_COST=100`, `GENERATE_MORE_COST=40`, `SIGNUP_GRANT=300`) — set via cloudbuild substitutions or left at defaults. Deploy backend (Cloud Run) + frontend (Vercel), coordinated like SP2.

**Live smoke:** sign up → 300 granted; report → 200; generate-more → 160; drain below 100 → out-of-credits modal → notify-me; `/credits` shows the history; nav pill updates; PDF export still clean (header excluded from `/print`).

## File structure

### New files
```
src/api/credits.py                          # credit service (ensure_profile, get_balance, spend, grant)
tests/unit/test_credits.py
tests/integration/test_credit_gating.py
tests/integration/test_credits_endpoints.py
src/app/components/AppHeader.tsx             # logged-in nav shell (brand · Reports · Credits · account)
src/app/components/CreditsBadge.tsx          # live balance pill
src/app/components/OutOfCreditsModal.tsx     # logged-in out-of-credits → notify me
src/app/lib/useCredits.ts                    # balance hook (GET /me) + refetch
src/app/credits/page.tsx                     # balance + cost table + history + notify-me
```

### Modified files
```
src/api/db.py                 # profile/ledger/upgrade_intent helpers + spend/grant/ensure RPC wrappers
src/api/main.py               # credit gating in generate-report/generate-more; /me, /credits/history, /upgrade-intent
tests/helpers/fake_db.py      # in-memory profiles, transactions, upgrade_intent + atomic spend/grant/ensure
tests/integration/test_auth_gating.py   # update the 2 SP2 "unlimited" tests to credit metering
src/app/layout.tsx            # mount AppHeader (replaces floating AuthNav)
src/app/components/AuthNav.tsx           # removed — superseded by AppHeader
src/app/report/[id]/Toolbar.tsx          # drop SP2.1 account controls; cost-on-buttons; OUT_OF_CREDITS → OutOfCreditsModal
src/app/report/[id]/page.tsx             # refetch balance after generate-more
src/app/page.tsx              # home: cost hint on generate button; OUT_OF_CREDITS → OutOfCreditsModal
src/app/welcome/page.tsx      # copy: "unlimited" → credits
README.md                     # SP3 credits + SQL migration docs
cloudbuild.yaml               # optional _REPORT_COST/_GENERATE_MORE_COST/_SIGNUP_GRANT substitutions (defaults fine)
```

## Open questions

None — resolved across the five design sections.
