# ChartSage

Drop a CSV or Excel file. Get a narrated data report with charts in under 10 seconds.

## What it does

ChartSage profiles your data, asks Claude to pick the 5-7 charts that tell the most useful story, renders them with ECharts, and wraps the result in a written executive summary plus a data-quality callout when something looks off in your data.

## Tech Stack

- **Frontend:** Next.js 14, React, TypeScript, Tailwind CSS, ECharts
- **Backend:** FastAPI, Python 3.11+, pandas, Pydantic v2
- **AI:** Claude via Anthropic SDK (Haiku 4.5 default; switchable)
- **Storage:** Redis (24-hour session TTL)

## Getting started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Redis running on localhost:6379 (`brew install redis && brew services start redis`)
- An Anthropic API key

### Setup

```bash
# Backend
cp .env.example .env
# edit .env to add ANTHROPIC_API_KEY
pip install -r requirements.txt

# Frontend
npm install
```

### Run

In one terminal:

```bash
make dev          # FastAPI on :8000
```

In another:

```bash
npm run dev       # Next.js on :3000
```

Open `http://localhost:3000`, drop a CSV, see a report.

## Switching models

Default is `haiku-4-5` (~$0.01 per report). Switch by setting one env var:

```bash
CLAUDE_MODEL=sonnet-4-6 make dev          # ~$0.035/report
CLAUDE_MODEL=opus-4-7 make dev            # ~$0.04/report
```

Per-pass overrides (cheap selection, smarter narrative):

```bash
CLAUDE_MODEL_SELECTION=haiku-4-5
CLAUDE_MODEL_NARRATIVE=sonnet-4-6
```

## Tests

```bash
make test         # unit + integration (~4s, no API calls)
make test-e2e     # real Claude smoke tests (~60s, ~$0.06)
```

## Internal BI (analytics)

Local, free Evidence.dev dashboards over ChartSage's own data — signups, reports, revenue/credits,
free→paid conversion, plus PostHog product funnels. Lives in [`bi/`](bi/):

```bash
cd bi && npm install && npm run sources && npm run dev -- --port 4000   # → http://localhost:4000
```

See [bi/README.md](bi/README.md) for one-time setup (the read-only DB role), refreshing data, and
how to add new charts and pages.

## Architecture

See [docs/superpowers/specs/2026-05-23-chartsage-rebuild-design.md](docs/superpowers/specs/2026-05-23-chartsage-rebuild-design.md).

## License

MIT.

## Deploying

Production runs on Vercel (frontend) + Google Cloud Run (backend) + Supabase (Postgres + Storage + auth) + PostHog (analytics).

### One-time provisioning

1. **Supabase**
   - Create project at supabase.com (US-East recommended).
   - SQL editor → run the schema from [the SP1 design](docs/superpowers/specs/2026-05-24-sp1-foundation-design.md#data-model).
   - Storage → create a private bucket named `csv-inputs`.
   - Settings → copy `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`.

2. **PostHog**
   - Create a free account + project at posthog.com.
   - Copy the project API key (`phc_...`).

3. **Google Cloud**
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com
   gcloud iam service-accounts create chartsage-runner
   ```
   Push secrets:
   ```bash
   echo -n "$ANTHROPIC_API_KEY" | gcloud secrets create anthropic-key --data-file=-
   echo -n "$SUPABASE_SERVICE_ROLE_KEY" | gcloud secrets create supabase-srk --data-file=-
   echo -n "$POSTHOG_API_KEY" | gcloud secrets create posthog-key --data-file=-
   ```

4. **Vercel**
   - Import this repo at vercel.com.
   - Add env vars: `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `NEXT_PUBLIC_POSTHOG_KEY`, `NEXT_PUBLIC_POSTHOG_HOST`.

### Deploy

Backend:
```bash
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_SUPABASE_URL=https://YOUR.supabase.co,_FRONTEND_BASE_URL=https://chartsage.vercel.app
```

Frontend:
```bash
git push origin main   # Vercel auto-deploys
```

### Smoke test

After the first deploy, visit your Vercel URL. Drop a CSV. Verify the report renders, check the Supabase `reports` table has a row, and check the PostHog dashboard for `report_generation_succeeded` events.

## Accounts (SP2)

Logged-in users (Google or magic link) get unlimited reports and "Generate more"; anonymous visitors keep one free report and see an upsell modal on "Generate 5 more". On sign-in, the visitor's anonymous report is migrated into their account, and a **My Reports** page lists saved reports.

**How auth works:** the browser uses `@supabase/ssr` to hold the session; `apiFetch` attaches `Authorization: Bearer <supabase JWT>` to every API call (plus the anon `X-Anon-Id` fallback). The FastAPI backend (`src/api/auth.py`) verifies the JWT locally against Supabase's public JWKS (asymmetric ES256/RS256, audience + expiry) — no per-request call to Supabase and no new secret. `deps.get_identity` resolves Bearer→authenticated, else X-Anon-Id→anonymous. Gating: authenticated → unlimited; anonymous → 1-report cap and `402 UPGRADE_REQUIRED` on generate-more.

### One-time auth provisioning

1. **Google OAuth client** (Google Cloud Console → APIs & Services):
   - OAuth consent screen → configure (External; app name; support email).
   - Credentials → Create credentials → OAuth client ID → Web application.
   - Authorized redirect URI: `https://YOUR_PROJECT.supabase.co/auth/v1/callback`.
   - Copy the Client ID + Client secret.
2. **Supabase → Authentication → Providers → Google:** enable, paste the Client ID + secret. (Magic link / Email is on by default.)
3. **Supabase → Authentication → URL Configuration:**
   - Site URL: `https://chartsage-xi.vercel.app`
   - Redirect URLs: `https://chartsage-xi.vercel.app/auth/callback` and `http://localhost:3000/auth/callback`.
4. **Row-Level Security** (Supabase SQL editor) — defense-in-depth; the backend uses the service-role key which bypasses RLS, so existing flows are unaffected:
   ```sql
   ALTER TABLE reports ENABLE ROW LEVEL SECURITY;
   CREATE POLICY reports_owner_select ON reports FOR SELECT USING (auth.uid() = user_id);
   CREATE POLICY reports_owner_update ON reports FOR UPDATE USING (auth.uid() = user_id);
   ```
5. **Vercel env:** ensure `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY` (the `sb_publishable_...` key) are set for all environments.

**JWKS note:** verification needs no new backend secret — Supabase exposes a public JWKS at `${SUPABASE_URL}/auth/v1/.well-known/jwks.json`. Confirm it returns a non-empty `keys` array (new projects sign with asymmetric ES256). If a project still signs with symmetric HS256, fall back to verifying with the Supabase JWT secret stored as a Cloud Run secret.

No new Cloud Run secrets or env vars are required for SP2 (`SUPABASE_URL` is already set). Deploy the backend (`gcloud builds submit ...`) and frontend (`git push origin main`) together — they share the `402` generate-more contract.

## Credits (SP3)

Logged-in users run on a credit economy instead of "unlimited": a one-time **300-credit** starter grant on first login, **100 credits per report**, **40 per "Generate 5 more"** — debited only on success (a failed generation never charges). When the balance is short, generating returns `402 OUT_OF_CREDITS` and the UI shows an "out of credits — notify me" modal that records upgrade intent (the demand signal for SP4 paid top-ups). Anonymous visitors are unchanged (1 free report, then a create-account upsell).

**Architecture:** an append-only `credit_transactions` ledger (source of truth) plus a cached `profiles.credits_balance`, mutated only through atomic Postgres functions (`spend_credits` / `grant_credits` / `ensure_profile`). The backend exposes these as DB methods alongside the report methods and gates `generate-report` / `generate-more`; a logged-in app header shows the live balance and links to a `/credits` page (balance, costs, history). RLS is owner-only on the credit tables; the backend uses the service-role key.

### One-time provisioning

Run the migration in **Supabase → SQL editor**: paste the entire contents of [`docs/migrations/sp3-credits.sql`](docs/migrations/sp3-credits.sql) — it creates `profiles`, `credit_transactions`, `upgrade_intent`, the three credit functions, and RLS. It is safely re-runnable (`if not exists` / `create or replace`). Run it **before** deploying the SP3 backend (authenticated credit checks call these functions; without them, those calls fail safe with `503 CREDITS_UNAVAILABLE`).

**Costs are env-tunable** (Cloud Run env vars, defaults shown): `REPORT_COST=100`, `GENERATE_MORE_COST=40`, `SIGNUP_GRANT=300`. No new secrets. Deploy backend (`gcloud builds submit ...`) + frontend (`git push origin main`).
