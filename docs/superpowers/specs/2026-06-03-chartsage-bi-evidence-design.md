# ChartSage BI (Evidence.dev) — Design Spec

**Status:** Approved (design), pending spec review
**Date:** 2026-06-03
**Type:** Internal tooling — local, free BI on ChartSage's own business + product data

**Goal:** Stand up [Evidence.dev](https://evidence.dev) **locally and for free** as a code-based BI layer over ChartSage's own data, so founder-facing questions ("how many signups / reports / purchases this week, what's converting") are answered from versioned SQL + Markdown instead of clicking around Supabase and PostHog. Deliver the scaffold plus **one Overview page that exercises both data sources end-to-end**; everything after that is the owner writing more pages.

**Why Evidence / why free:** Evidence's core framework is **MIT-licensed and free to self-host / run locally forever** ([GitHub](https://github.com/evidence-dev/evidence), [self-host docs](https://docs.evidence.dev/deployment/self-host)). The only paid products are *Evidence Cloud* hosting, *Core Support* ($500/mo/seat), and a *Pro* SSO plan ([pricing](https://evidence.dev/pricing)) — **none are used here**. Local flow is `npm install` → `npm run sources` → `npm run dev` ([install docs](https://docs.evidence.dev/install-evidence)).

**Mental model (important):** Evidence is **not** a live-query dashboard. SQL "source queries" run when you execute `npm run sources`; results are cached locally as Parquet and queried in-browser via DuckDB. Pages are Markdown + SQL + chart components. **"Refresh data" = re-run `npm run sources`.** This is standard for Evidence and keeps the app fast and offline.

**Approved decisions (from brainstorming):**
- **Approach A — hybrid native sources:** Postgres connector (live business tables) + an Evidence **JavaScript data source** that calls PostHog's HogQL query API. Both first-class; no separate ETL script.
- **Location:** a **`bi/` subfolder** in this repo, versioned with the product, isolated from the Next.js tooling.

---

## Facet 1 — Project location & isolation

Scaffold Evidence into `bi/` (from the official template, `evidence-dev/template`). `bi/` is a self-contained Node/SvelteKit project with its own `package.json` and `node_modules` — it must not leak into the Next.js app's build or typecheck.

Isolation changes to the **root** repo:
- **`tsconfig.json`** — add `"bi"` to `exclude` (root `include` globs `**/*.ts`/`**/*.tsx`, which would otherwise pull in Evidence/SvelteKit TS).
- **`.gitignore`** — the Evidence template ships its own `bi/.gitignore` (ignores `node_modules`, `.evidence`, `build`, `.svelte-kit`). Add a belt-and-braces root block for `bi/node_modules/`, `bi/.evidence/`, `bi/build/`, `bi/.svelte-kit/` so nothing heavy is ever staged from the repo root. `bi/.env` is already covered by the existing root `.env` pattern.
- **`next lint`** scopes to `pages`/`app`/`src` only, so `bi/` is not linted — no change needed. The backend Docker image already excludes `docs`/`node_modules`; `bi/` is frontend-only and never shipped to Cloud Run.

`bi/` **is** committed (the SQL/Markdown definitions are the value). Only its build artifacts and deps are ignored.

## Facet 2 — Data sources

### `sources/supabase` — Postgres connector (business data)
`bi/sources/supabase/connection.yaml` of type `postgres`; credentials supplied via env (`EVIDENCE_SOURCE__supabase__<var>`) read from `bi/.env`. Source queries (`.sql`) materialised on `npm run sources`:

| Query | Source table(s) | Notes |
|---|---|---|
| `signups` | `profiles` (`created_at`) | one profile per registered user; daily counts |
| `reports` | `reports` (`created_at`, `user_id`) | reports/day; split anon (`user_id is null`) vs signed-in |
| `credit_purchases` | `credit_transactions` where `reason='stripe_purchase'` | count + `sum(delta)` credits granted; `ref` = checkout session |
| `credit_spend` | `credit_transactions` where `delta < 0` | credits consumed (report generation) |
| `anon_volume` | `anon_report_log` (`created_at`) | anonymous attempts (abuse/cap view) |

Primary tables are in the `public` schema (no special grants). `auth.users` is **not** required (use `profiles` as the registered-user spine). **$ revenue is not stored in Postgres** (only credit `delta`); see Open Items.

### `sources/posthog` — JavaScript data source (product data)
A JS source ([docs](https://docs.evidence.dev/core-concepts/data-sources/javascript)) that POSTs HogQL to PostHog's query API and returns rows. Reuses the existing `POSTHOG_PERSONAL_API_KEY` + `POSTHOG_PROJECT_ID` from the app's `.env` (copied into `bi/.env`). Shape (to be finalised against the docs at build):
- `POST {posthog_api_host}/api/projects/{project_id}/query/`, header `Authorization: Bearer {personal_api_key}`, body `{"query": {"kind": "HogQLQuery", "query": "<sql>"}}`.
- **Host caveat:** the app's `NEXT_PUBLIC_POSTHOG_HOST` (`us.i.posthog.com`) is the *ingestion* host; the *query* API is the app host (e.g. `https://us.posthog.com`). Confirm at build.
- Queries: (1) `events_daily` — `SELECT toDate(timestamp) d, event, count() FROM events GROUP BY d, event` for trends; (2) a `checkout_cancelled`-over-time series; (3) a visit→upload→report→checkout funnel. **Exact event names are unknown** and will be discovered first via a `SELECT event, count() FROM events GROUP BY event ORDER BY 2 DESC` introspection query.

Cross-source joins (e.g. PostHog activation vs Postgres purchases) happen at the **page/query layer** — Evidence's DuckDB engine lets one query reference another's result.

## Facet 3 — Credentials & safety

- **Postgres:** recommended path is a dedicated **read-only Postgres role** in Supabase (BI can never write). To move fast we *may* start with the standard Supabase connection string — owner's call at build time.
- **Connection method:** use Supabase's **Session pooler** connection (IPv4-friendly, `...pooler.supabase.com`), `sslmode=require`. Direct `db.<ref>.supabase.co` is often IPv6-only. Confirm host/port at build.
- All secrets live in **`bi/.env` (gitignored)**. Nothing secret is committed. `bi/.env.example` documents the required keys.

## Facet 4 — First deliverable: `pages/index.md` (Overview)

One page proving both sources render with **real data**:
- **Business (Postgres):** `<BigValue>` tiles (total signups, reports, credit purchases, credits spent); `<LineChart>` signups/day and reports/day; reports split anon vs signed-in; free→paid conversion (% of `profiles` with ≥1 `stripe_purchase`).
- **Product (PostHog):** top events bar/table; `checkout_cancelled` trend; a funnel once event names are confirmed.

YAGNI: no extra pages, themes, or auth in v1 — just the scaffold + Overview. Owner expands from there.

## Facet 5 — Run & refresh workflow (documented in `bi/README.md`)

```bash
cd bi
npm install
npm run sources                 # pull Postgres + PostHog → local Parquet
npm run dev -- --port 4000      # http://localhost:4000  (4000, NOT 3000 — Next.js owns 3000)
```
Refresh data = re-run `npm run sources`. `bi/README.md` records this plus the `.env` keys.

## Facet 6 — Implementation order

1. **Pre-flight:** confirm Evidence's current supported Node range vs local Node v23.11.0; if unsupported, `nvm install --lts` (20/22) for the `bi/` work. Scaffold `bi/` from the template; `npm install`.
2. Add root isolation (tsconfig `exclude`, `.gitignore` block).
3. Configure `sources/supabase` (Postgres) + `bi/.env`; run `npm run sources`; confirm business queries return rows.
4. Build `sources/posthog` JS source; introspect events; add trend/funnel queries; re-run sources.
5. Write `pages/index.md` Overview; `npm run dev -- --port 4000`; **verify it renders real data in the browser**.
6. Add `bi/README.md` + `bi/.env.example`.

## Open items (resolved at build, not blocking)
- **$ revenue:** not in Postgres. v1 reports **credits purchased** + **purchase count** from Postgres; true $ revenue is derived from the credits→pack-price map ($5/$15/$40) found in the app code, or read from a PostHog purchase event if one carries a value. Note the chosen basis on the page.
- **PostHog event names** + **query API host** + **Supabase pooler host/port** — confirmed empirically during steps 1, 3–4.
- **Evidence Node support** — verified before scaffolding.

## Non-goals
- Deploying Evidence anywhere (local-only by request).
- Real-time/live dashboards (Evidence refreshes on `npm run sources`).
- Replacing PostHog or Supabase as systems of record.
- Building an exhaustive dashboard suite now (just the Overview).
