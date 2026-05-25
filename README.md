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
