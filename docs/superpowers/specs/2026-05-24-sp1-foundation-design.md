# SP1 — Production Foundation Design

**Date:** 2026-05-24
**Author:** Chris Silver (with Claude)
**Status:** Approved for implementation planning

## Context

ChartSage runs locally today: Next.js + FastAPI + Playwright + Redis on one machine. The full production roadmap (deployment, accounts, credits, paywall, onboarding, PostHog) decomposes into three sub-projects that ship sequentially:

- **SP1 (this spec):** Deploy. Replace Redis with persistent storage. Anonymous cookie identity. 1-report limit per anon. PostHog wired.
- **SP2:** Supabase Auth, login/signup, onboarding screen, migrate anon reports to user accounts on signup, unlimited reports for logged-in users.
- **SP3:** Credit ledger, gated generate-more with upsell modal, starter credits on signup.

Each sub-project is independently shippable and useful. SP1 makes the app live on the internet with a 1-report-per-visitor gate; SP2 layers identity; SP3 layers economics.

## Goals (SP1)

- The app runs on production infrastructure (Cloud Run + Vercel + Supabase + PostHog) at <$10/mo at launch traffic.
- Reports persist permanently — no more 24-hour TTL.
- Each anonymous visitor (identified by a long-lived UUID cookie) can generate exactly one report. Subsequent attempts route to a friendly placeholder page.
- All cost-bearing actions emit PostHog events with token counts and dollar estimates so spend tracking is live from day one.
- No regressions to existing functionality: 10-chart generation, generate-more, PDF export, drag-and-drop, collapsible summary all keep working.
- 123 existing tests stay green; ~10 new tests cover the database, storage, anon limit, and event emission paths.

## Non-goals (SP1)

- Accounts, login, signup (SP2)
- Credits, upsell modal, paywall (SP3)
- Stripe / payments (SP4)
- Row-level security (deferred to SP2 when users exist)
- Per-user rate limiting (Cloud Run max-instances is the only throttle for SP1)
- Report deletion / retention policy (SP3 follow-up)
- Custom domain (cosmetic; stays on `*.vercel.app` until SP2)

## Architecture

```
                       Browser
                          │
              ┌───────────┴────────────────────┐
              │  Cookie: chartsage_anon=<UUID> │
              └───────────┬────────────────────┘
                          │
       ┌──────────────────▼──────────────────┐
       │  Vercel (Next.js)                    │
       │  - middleware sets/reads anon cookie │
       │  - posthog-js fires events           │
       │  - PostHog $user_id = anon cookie    │
       └──┬─────────────────────────────────┬─┘
          │ API calls include               │
          │ X-Anon-Id: <uuid>               │
          ▼                                 ▼
┌─────────────────────────┐    ┌──────────────────────────┐
│  Cloud Run (FastAPI)     │    │  PostHog cloud           │
│  - 1 vCPU, 1GB, scale-0  │    │  - server-side capture   │
│  - Playwright + Chromium │◄───┤    (cost-bearing events) │
│  - reads anon_id header  │    └──────────────────────────┘
│  - enforces anon limit   │
└──┬────────────┬──────────┘
   │            │
   ▼            ▼
┌──────────┐  ┌──────────────────┐
│ Supabase │  │ Supabase Storage │
│ Postgres │  │  - raw CSV blobs │
│  reports │  │  bucket: csv-    │
│          │  │  inputs (private)│
└──────────┘  └──────────────────┘
```

### Stack

- **Frontend host:** Vercel Hobby (free)
- **Backend host:** Google Cloud Run (~$0/mo at launch traffic; scales to zero)
- **Database:** Supabase Postgres (free 500MB)
- **File storage:** Supabase Storage `csv-inputs` bucket (free 1GB)
- **Analytics:** PostHog cloud (free 1M events/mo)
- **Anon identity:** httpOnly `chartsage_anon` UUID cookie, sameSite=Lax, secure, 1-year expiry

### Cost estimate at launch (~10 reports/day)

- Vercel: $0
- Cloud Run: ~$0 (free tier covers 2M requests/mo + 360K vCPU-seconds)
- Supabase: $0
- PostHog: $0
- Anthropic API: ~$3/mo (10 × $0.011 × 30)

**Total infra <$5/mo** through ~100 reports/day. Past that, Cloud Run and Anthropic become the dominant costs and scale linearly.

### What goes away

- Redis (replaced by Postgres for reports + Supabase Storage for CSV blobs)
- Local `src/api/logs/` files (replaced by Cloud Run stdout → Cloud Logging)

### What stays

- All chart generation logic (executors, profile, generate-more, Playwright PDF)
- Pydantic Report/ChartSpec schemas
- 123 existing tests

## Data model

### Postgres schema

```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE reports (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    anon_id           UUID,
    user_id           UUID,                            -- populated in SP2
    report_json       JSONB NOT NULL,                  -- full Report shape
    csv_storage_key   TEXT,                            -- path in Storage; NULL = unavailable
    title             TEXT,                            -- denormalized for list views
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (anon_id IS NOT NULL OR user_id IS NOT NULL)
);

CREATE INDEX reports_anon_id_idx
    ON reports (anon_id, created_at DESC)
    WHERE user_id IS NULL;

CREATE INDEX reports_user_id_idx
    ON reports (user_id, created_at DESC)
    WHERE user_id IS NOT NULL;
```

- `id` is the existing `session_id` — no rename.
- `anon_id` and `user_id` are mutually exclusive in practice. SP2 migration is just `UPDATE` flipping one for the other.
- `report_json` carries the whole Pydantic-validated `Report`. Pydantic is source of truth; DB just stores it.
- `title` is denormalized from the summary's first sentence at write time, for the eventual "my reports" list.
- No row-level security yet (everything via backend service role); SP2 introduces RLS.

### Supabase Storage

- Bucket: `csv-inputs` (private)
- Key pattern: `{report_id}.csv`
- One file per report — the original upload, as UTF-8 CSV (XLSX converted to CSV before upload)
- 10MB max per file (matches existing upload limit)
- No TTL — files persist as long as the report does
- Service-role-only access (backend uses Supabase admin key)

**Why Storage instead of Postgres `bytea`:**
- Cheap (Supabase free 1GB; Postgres free is 500MB, reserved for metadata)
- S3-compatible — migrate later if needed
- Streamable downloads for future "re-export original" feature

### Anon limit query

```sql
SELECT COUNT(*) FROM reports WHERE anon_id = $1 AND user_id IS NULL;
```

If `>= 1`, return `403 ANON_LIMIT_REACHED`. Frontend routes to `/anon-limit`.

### Migration from current Redis state

1. SP1 deployment writes only to Postgres / Storage going forward
2. Existing local Redis data is dev-only (TTL 24h) — no production data exists yet
3. `redis-py` stays in `requirements.txt` for now (some tests still use it via FakeRedis); removed in SP2 cleanup

## Components

### Backend (`src/api/`)

**New modules:**

- `db.py` — Supabase Postgres client (`supabase-py`). Exposes `save_report`, `get_report`, `update_report_json`, `update_layout`, `count_anon_reports`. Single abstraction over what used to be raw Redis calls. Returns Pydantic-validated `Report` objects.
- `storage.py` — Supabase Storage wrapper. `upload_csv(report_id, csv_bytes)`, `download_csv(report_id) → bytes`, `delete_csv(report_id)`. Service-role key only.
- `posthog_server.py` — thin wrapper around `posthog` SDK for server-side event capture. One function: `capture(distinct_id, event, properties)`. Fires async; never blocks the request. Silently swallows errors.
- `deps.py` — FastAPI dependency that extracts `X-Anon-Id` from request headers, validates as UUID, returns it. Raises 400 if missing/malformed.

**Modified modules:**

- `main.py` — every endpoint that touched Redis now calls `db.py`. `/generate-report` gains an anon-limit check at the top. `/generate-report` writes the CSV to Storage instead of Redis. Removes `r = Depends(get_redis)`. Adds `anon_id: UUID = Depends(get_anon_id)` to gated endpoints. Fires PostHog events on `report_generation_*`, `generate_more_*`, `pdf_export_*`.
- `report_generator.py` — unchanged. Pure orchestration.
- `pdf_export.py` — unchanged. Cloud Run image has Chromium pre-installed.

### Frontend (`src/`)

**New files:**

- `src/middleware.ts` — Next.js root middleware. If no `chartsage_anon` cookie, generate UUID and set it (httpOnly, sameSite=Lax, secure in prod, 1-year expiry).
- `src/app/lib/anon.ts` — `getAnonId(): string` reads the cookie. Used by every fetch wrapper.
- `src/app/lib/posthog.ts` — initializes PostHog browser SDK on first import. Sets `$user_id` from the anon cookie so client/server events stitch.
- `src/app/lib/api.ts` — single `apiFetch(path, init)` wrapper that injects `X-Anon-Id: <uuid>` into every request. Replaces scattered bare `fetch(${NEXT_PUBLIC_API_URL}/...)` calls.
- `src/app/anon-limit/page.tsx` — placeholder shown when anon hits the 1-report limit. Friendly copy + disabled "Sign in" button ("coming soon"). Real signup arrives in SP2.

**Modified files:**

- `src/app/page.tsx` — uses `apiFetch`; on `403 ANON_LIMIT_REACHED`, redirects to `/anon-limit`.
- `src/app/report/[id]/useReportLayout.ts` — uses `apiFetch` for PATCH calls.
- `src/app/report/[id]/Toolbar.tsx` — uses `apiFetch` for generate-more and PDF export; fires PostHog events.
- `src/app/layout.tsx` — imports `lib/posthog` once for global init.

### Deployment artifacts

- `Dockerfile` (repo root) — base `mcr.microsoft.com/playwright/python:v1.42.0-jammy`. Copies `src/api/`, installs `requirements.txt`, runs `uvicorn`. Final image ~600MB compressed.
- `.dockerignore` — excludes tests/, docs/, .git/, venv/, node_modules/.
- `cloudbuild.yaml` (repo root) — declarative Cloud Build that builds + deploys to Cloud Run.
- `vercel.json` — Vercel build config.
- `.env.production.example` — template listing every env var the deployment needs.

### Tests

- `tests/helpers/fake_db.py` — in-memory dict-backed db client matching `db.py` interface.
- `tests/helpers/fake_storage.py` — in-memory storage. Same pattern.
- `tests/helpers/fake_posthog.py` — captures events without sending; tests assert on captured events.
- `tests/integration/test_anon_limit.py` — verifies 1-report-per-anon enforcement.
- `tests/integration/test_db.py` — `save_report` + `get_report` round-trip; `count_anon_reports` correctness.
- `tests/integration/test_storage_failure.py` — Storage upload error → no orphaned report row.
- `tests/integration/test_posthog_events.py` — assert event names + camelCase property keys.
- Update `tests/integration/test_api_layout.py` + `test_api_errors.py` — swap `FakeRedis` → `FakeDB` + `FakeStorage`; add `X-Anon-Id` header to all test requests.

## Data flow

### First-time visit

```
Browser → GET /
  Next.js middleware: no cookie → set chartsage_anon=<uuid> (httpOnly, sameSite=Lax, secure, 1yr)
  posthog-js identifies as that UUID
  → fires $pageview
```

### First report generation

```
Browser → POST /generate-report  (X-Anon-Id: <uuid>, multipart CSV)

Backend:
  1. get_anon_id dependency parses header → UUID
  2. db.count_anon_reports(anon_id) → 0
  3. Read CSV bytes, build DataFrame, profile
  4. ReportGenerator.build_report() → Report with chart_ids + layout
  5. db.save_report(id, anon_id, user_id=None, report_json, csv_key, title)
  6. storage.upload_csv(report_id, csv_bytes)
  7. posthog_server.capture(anon_id, "report_generation_succeeded", {
       reportId, rowCount, columnCount, chartCount,
       modelSelection, modelNarrative,
       inputTokens, outputTokens, cacheReadTokens,
       estCostUsd, elapsedMs
     })
  8. Return { session_id }

Browser → router.push(`/report/${session_id}`)
  posthog.capture("report_viewed", { reportId, chartCount, hasDataQualityNotes, loadMs })
```

### Second report attempt (anon limit hit)

```
Browser → POST /generate-report (same anon cookie, second time)

Backend:
  1. db.count_anon_reports(anon_id) → 1
  2. posthog_server.capture(anon_id, "anon_limit_blocked")
  3. Return 403 with body { code: "ANON_LIMIT_REACHED", detail: "..." }

Browser:
  apiFetch sees 403 + code ANON_LIMIT_REACHED → router.push("/anon-limit")
  /anon-limit page fires "anon_limit_page_viewed"
```

### PATCH layout

```
Browser → PATCH /report/{id}/layout (debounced 500ms)

Backend:
  1. db.get_report(id) → Report  (no auth check in SP1)
  2. validate every chart_id in new layout against report.charts
  3. db.update_layout(id, new_layout) — single jsonb_set update
  4. Return 204

PostHog: not captured per PATCH (too noisy). Once-per-session "layout_edited" on first drag instead.
```

### Generate-more

```
Browser → POST /report/{id}/generate-more

Backend:
  1. db.get_report(id)
  2. storage.download_csv(id) → bytes
  3. pd.read_csv(bytes) → DataFrame
  4. gen.generate_more(existing.charts) → (new_charts, new_layout)
  5. db.update_report_json(id, updated_report)
  6. posthog_server.capture(anon_id, "generate_more_succeeded", {
       reportId, newChartCount, inputTokens, outputTokens, estCostUsd, elapsedMs
     })
  7. Return updated Report
```

### PDF export

```
Browser → window.open(`${API}/report/${id}/export.pdf`)

Backend:
  1. db.get_report(id) — 404 if missing
  2. posthog_server.capture(anon_id, "pdf_export_started", { reportId, coldStart })
  3. render_report_pdf(id) — Playwright navigates to FRONTEND_BASE/report/{id}/print
  4. posthog_server.capture(anon_id, "pdf_export_succeeded", { reportId, byteSize, elapsedMs })
  5. Stream PDF
```

## PostHog event taxonomy

Events: `snake_case`. Properties: `camelCase`. Server-side events are source-of-truth for cost-bearing actions (can't be ad-blocked).

### Naming convention rules

1. Events: `snake_case`, verb-ish (`csv_dropped`, `report_viewed`).
2. Properties: `camelCase` (`reportId`, `inputTokens`, `estCostUsd`).
3. Cost-bearing events include `inputTokens`, `outputTokens`, `estCostUsd`.
4. Failure events include `reason`, `httpStatus` (when applicable), and `errorClass` (Python exception class name).
5. Server captures use `distinct_id = anonId` so client/server events stitch into one user timeline.

### Lifecycle / funnel (client)

| Event | Properties |
|---|---|
| `landing_viewed` | `referrer` |
| `dropzone_focused` | — |
| `csv_dropped` | `filename`, `sizeKb`, `kind` (`csv`\|`xlsx`) |
| `csv_rejected_client` | `reason` (`tooLarge`\|`wrongType`), `sizeKb?`, `kind?` |
| `csv_preview_shown` | `rowCount`, `columnCount` |
| `generate_clicked` | `filename`, `sizeKb` |

### Report generation (server, cost-bearing)

| Event | Properties |
|---|---|
| `report_generation_started` | `rowCount`, `columnCount`, `filename`, `sizeBytes` |
| `report_generation_succeeded` | `reportId`, `rowCount`, `columnCount`, `chartCount`, `modelSelection`, `modelNarrative`, `inputTokens`, `outputTokens`, `cacheReadTokens`, `estCostUsd`, `elapsedMs` |
| `report_generation_failed` | `reason`, `errorClass`, `httpStatus`, `elapsedMs` |
| `claude_overloaded` | `stage` (`selection`\|`narrative`) |

### Report viewing (client)

| Event | Properties |
|---|---|
| `report_viewed` | `reportId`, `chartCount`, `hasDataQualityNotes`, `loadMs` |
| `report_load_failed` | `reportId`, `httpStatus`, `errorMessage` |
| `summary_toggled` | `reportId`, `action` (`collapsed`\|`expanded`) |

### Layout edits (client)

| Event | Properties |
|---|---|
| `chart_dragged` | `reportId`, `chartId`, `fromIndex`, `toIndex`, `sameContext` |
| `chart_hidden` | `reportId`, `chartId`, `fromMainIndex` |
| `chart_promoted` | `reportId`, `chartId`, `fromSidebarIndex` |
| `sidebar_toggled` | `reportId`, `action` |
| `layout_save_failed` | `reportId`, `httpStatus`, `errorMessage` |
| `layout_save_blocked` | `reportId` |

### Generate-more

| Event | Side | Properties |
|---|---|---|
| `generate_more_clicked` | client | `reportId`, `currentChartCount` |
| `generate_more_started` | server | `reportId`, `existingChartCount` |
| `generate_more_succeeded` | server | `reportId`, `newChartCount`, `inputTokens`, `outputTokens`, `estCostUsd`, `elapsedMs` |
| `generate_more_failed` | server | `reportId`, `reason`, `httpStatus`, `elapsedMs` |

### PDF export

| Event | Side | Properties |
|---|---|---|
| `export_pdf_clicked` | client | `reportId` |
| `pdf_export_started` | server | `reportId`, `coldStart` |
| `pdf_export_succeeded` | server | `reportId`, `byteSize`, `elapsedMs` |
| `pdf_export_failed` | server | `reportId`, `reason`, `errorClass` |

### Anon limit (conversion funnel)

| Event | Side | Properties |
|---|---|---|
| `anon_limit_blocked` | server | — |
| `anon_limit_page_viewed` | client | `entryPoint` (`directLink`\|`afterUpload`) |
| `signin_cta_clicked` | client | `from` (`anonLimit`\|`toolbar`) |

### Outbound

| Event | Properties |
|---|---|
| `new_report_link_clicked` | `reportId` |

## Deployment

### Supabase setup (one-time)

1. Create project on supabase.com (US-East).
2. Run the SQL schema (Data Model section).
3. Storage → create `csv-inputs` bucket → set private.
4. Project Settings → grab three values:
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY` (Vercel only)
   - `SUPABASE_SERVICE_ROLE_KEY` (Cloud Run only, never frontend)

### Cloud Run setup (one-time)

```bash
gcloud config set project chartsage-prod
gcloud services enable run.googleapis.com cloudbuild.googleapis.com
gcloud iam service-accounts create chartsage-runner
gcloud projects add-iam-policy-binding chartsage-prod \
  --member=serviceAccount:chartsage-runner@... --role=roles/run.invoker
```

**`Dockerfile`** at repo root:

```dockerfile
FROM mcr.microsoft.com/playwright/python:v1.42.0-jammy
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/api ./src/api
ENV PYTHONPATH=/app/src/api
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--app-dir", "src/api"]
```

**`cloudbuild.yaml`** at repo root:

```yaml
steps:
  - name: gcr.io/cloud-builders/docker
    args: [build, -t, gcr.io/$PROJECT_ID/chartsage-backend:$SHORT_SHA, .]
  - name: gcr.io/cloud-builders/docker
    args: [push, gcr.io/$PROJECT_ID/chartsage-backend:$SHORT_SHA]
  - name: gcr.io/google.com/cloudsdktool/cloud-sdk
    entrypoint: gcloud
    args:
      - run
      - deploy
      - chartsage-backend
      - --image=gcr.io/$PROJECT_ID/chartsage-backend:$SHORT_SHA
      - --region=us-central1
      - --platform=managed
      - --allow-unauthenticated
      - --memory=1Gi
      - --cpu=1
      - --min-instances=0
      - --max-instances=10
      - --concurrency=4
      - --timeout=120s
      - --service-account=chartsage-runner@$PROJECT_ID.iam.gserviceaccount.com
      - --set-secrets=ANTHROPIC_API_KEY=anthropic-key:latest,SUPABASE_SERVICE_ROLE_KEY=supabase-srk:latest,POSTHOG_API_KEY=posthog-key:latest
      - --set-env-vars=SUPABASE_URL=https://xxxx.supabase.co,FRONTEND_BASE_URL=https://chartsage.vercel.app,CLAUDE_MODEL=haiku-4-5
```

Deploy: `gcloud builds submit --config cloudbuild.yaml`

### Vercel setup (one-time)

1. Import repo at vercel.com — Next.js auto-detected.
2. Environment Variables:
   - `NEXT_PUBLIC_API_URL=https://chartsage-backend-xxxx-uc.a.run.app`
   - `NEXT_PUBLIC_SUPABASE_URL=https://xxxx.supabase.co`
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJh...`
   - `NEXT_PUBLIC_POSTHOG_KEY=phc_...`
   - `NEXT_PUBLIC_POSTHOG_HOST=https://us.i.posthog.com`
3. Push to `main` → auto-deploys.

### PostHog setup (one-time)

1. Create account + project on posthog.com.
2. Copy project API key (`phc_...`).
3. Autocapture off — events are explicit.
4. Add to Cloud Run secrets as `posthog-key`.
5. Add to Vercel env as `NEXT_PUBLIC_POSTHOG_KEY`.

### Secrets management

Backend secrets in **Google Secret Manager**. Frontend in **Vercel env vars**. Nothing in git.

```bash
echo -n "$ANTHROPIC_API_KEY" | gcloud secrets create anthropic-key --data-file=-
echo -n "$SUPABASE_SERVICE_ROLE_KEY" | gcloud secrets create supabase-srk --data-file=-
echo -n "$POSTHOG_API_KEY" | gcloud secrets create posthog-key --data-file=-
```

### CORS

Update `main.py`:

```python
allow_origin_regex=r"https://chartsage(-[a-z0-9-]+)?\.vercel\.app|http://localhost:3000"
```

### Local dev — what changes

`make dev` and `npm run dev` still work — they hit the production Supabase project. `.env.local` carries the Supabase + PostHog + Anthropic keys. For pure-offline dev, `supabase start` runs a Dockerized local instance (optional polish; not required for SP1).

### Domain (deferred)

Stays on `*.vercel.app` + `*.a.run.app` for SP1. Custom domain after SP2.

## Error handling

### New failure modes

| Where | What can go wrong | Response |
|---|---|---|
| **Supabase Postgres unreachable** | Network/Supabase outage | 503 `STORAGE_UNAVAILABLE`; toast "Service degraded, try again" |
| **Supabase Storage timeout (upload)** | Slow link, big file | 30s timeout. On fail: 502; transaction reverts so no orphan row |
| **Supabase Storage download fail (generate-more)** | Object missing | 404 `SOURCE_DATA_UNAVAILABLE`; toast "Source data is no longer available" |
| **PostHog ingest fails** | Network/outage | Silent swallow; WARN log. Analytics must never block product flow. |
| **PostHog blocked client-side** | Ad blocker | Same — silent. Server-side events still capture cost-bearing actions. |
| **Cloud Run cold-start >120s** | Very rare | 504; toast "Service is starting up, try again in 30s" |
| **`X-Anon-Id` missing** | Bot, curl, frontend bug | 400 `MISSING_ANON_ID` |
| **`X-Anon-Id` malformed** | Tampered cookie | 400 `INVALID_ANON_ID` |
| **Anon hits limit + POSTs again** | Expected | 403 `ANON_LIMIT_REACHED`; frontend → `/anon-limit` |
| **Concurrent uploads from same anon** | Race condition | Last-write-wins per request; both succeed; next attempt blocked. Acceptable for v1. |

### What we deliberately don't handle in SP1

- Ownership checks per report — UUIDs unguessable (128-bit); SP2 adds RLS tied to `user_id`.
- Per-anon rate limiting beyond the 1-report cap — Cloud Run `--max-instances=10` is the only throttle.
- CSV deletion / retention — CSVs persist forever in SP1 (storage cost negligible).

## Testing

### Backend test additions

- `tests/helpers/fake_db.py` — in-memory dict, matches `db.py` interface
- `tests/helpers/fake_storage.py` — in-memory bytes store, matches `storage.py` interface
- `tests/helpers/fake_posthog.py` — captures events, exposes `.events` list for assertions
- `tests/integration/test_anon_limit.py` — 2nd POST same anon → 403 with the right code
- `tests/integration/test_db.py` — save + get round-trip; count_anon_reports correctness
- `tests/integration/test_storage_failure.py` — upload error → no orphaned report row
- `tests/integration/test_posthog_events.py` — assert names + property keys (camelCase) for each cost-bearing call
- Update `test_api_layout.py` + `test_api_errors.py` — `FakeRedis` → `FakeDB` + `FakeStorage`; add `X-Anon-Id` to all requests

### Manual smoke test plan (after first deploy)

- [ ] Visit prod URL fresh browser → cookie `chartsage_anon` set with a UUID
- [ ] Drop a CSV → report generates
- [ ] PostHog shows `report_generation_succeeded` with token counts + estCostUsd
- [ ] Reload report URL → same data renders (Postgres-backed)
- [ ] Drag a chart → PATCH returns 204
- [ ] Try second CSV upload → redirects to `/anon-limit`
- [ ] PostHog shows `anon_limit_blocked` then `anon_limit_page_viewed`
- [ ] Clear cookie + refresh → can upload again
- [ ] "Export PDF" → works, warm-start under 3s
- [ ] Supabase dashboard: `reports` table has rows; `csv-inputs` bucket has the CSV

## File structure

### New files

- `src/api/db.py`
- `src/api/storage.py`
- `src/api/posthog_server.py`
- `src/api/deps.py`
- `src/middleware.ts`
- `src/app/lib/anon.ts`
- `src/app/lib/posthog.ts`
- `src/app/lib/api.ts`
- `src/app/anon-limit/page.tsx`
- `Dockerfile`
- `.dockerignore`
- `cloudbuild.yaml`
- `vercel.json`
- `.env.production.example`
- `tests/helpers/fake_db.py`
- `tests/helpers/fake_storage.py`
- `tests/helpers/fake_posthog.py`
- `tests/integration/test_anon_limit.py`
- `tests/integration/test_db.py`
- `tests/integration/test_storage_failure.py`
- `tests/integration/test_posthog_events.py`

### Modified files

- `src/api/main.py` — endpoints swap Redis → db.py; anon-limit check; PostHog calls; CORS regex
- `requirements.txt` — `+supabase`, `+posthog`; can drop `redis` if FakeRedis is fully replaced
- `src/app/page.tsx` — `apiFetch`; redirect on `ANON_LIMIT_REACHED`
- `src/app/report/[id]/useReportLayout.ts` — `apiFetch`
- `src/app/report/[id]/Toolbar.tsx` — `apiFetch` + PostHog events
- `src/app/layout.tsx` — import PostHog initializer
- `tests/integration/test_api_layout.py` — `FakeDB` + `FakeStorage` + `X-Anon-Id`
- `tests/integration/test_api_errors.py` — same
- `README.md` — add "Deploying" section

## Open questions

None — all decisions resolved in the sections above.
