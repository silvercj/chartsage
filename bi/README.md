# ChartSage BI (Evidence.dev)

Local, free, code-based BI for ChartSage. SQL + Markdown → interactive charts.
Built with [Evidence](https://evidence.dev) (MIT-licensed; self-hosting/local is free).

Two data sources:
- **`posthog`** — product analytics via the HogQL query API (a JavaScript source).
- **`supabase`** — business metrics from the Postgres DB (a read-only role).

## Run it

```bash
cd bi
npm install                     # first time only
npm run sources                 # pull PostHog + Supabase → local Parquet
npm run dev -- --port 4000      # http://localhost:4000   (4000, not 3000 — the app owns 3000)
```

**Evidence is not live** — it queries a local snapshot. To refresh the numbers, re-run
`npm run sources`. Node ≥18 (works on Node 23).

## One-time setup

Secrets live in `bi/.env` (gitignored). Start from the template:

```bash
cp .env.example .env
```

### PostHog (works immediately)
`bi/.env` already has the PostHog key/project copied from the app's root `.env`.
Query host is `https://us.posthog.com` (the app host, **not** the `us.i.posthog.com` ingestion host).

### Supabase Postgres (needs a read-only role)
1. **Create the role.** In Supabase → SQL editor, run
   [`docs/migrations/bi-readonly-role.sql`](../docs/migrations/bi-readonly-role.sql)
   (set a strong password first). It creates `chartsage_readonly` with SELECT-only access
   **and** the RLS read policies it needs (otherwise RLS returns zero rows).
2. **Get the connection.** Supabase → **Connect** → **Session pooler**. Use **port 5432** —
   *not* the 6543 transaction pooler (Evidence uses server-side cursors, which it doesn't support).
3. **Fill `bi/.env`:**
   ```
   EVIDENCE_SOURCE__supabase__host=aws-0-<region>.pooler.supabase.com
   EVIDENCE_SOURCE__supabase__user=chartsage_readonly.<project-ref>
   EVIDENCE_SOURCE__supabase__password=<the role password>
   ```
   - The pooler **user needs the `.<project-ref>` suffix** (direct connection instead?
     then `host=db.<ref>.supabase.co` and `user=chartsage_readonly` with no suffix).
   - ⚠️ **Escape any `$` in the password as `\$`.** Evidence loads `.env` through Vite, which
     *expands* `$` — e.g. a password `pa$1word` must be written `pa\$1word`. Quoting does **not** help.
4. `npm run sources` — the Business charts populate.

Connection settings (port/database/schema/SSL) live in `sources/supabase/connection.yaml` (committed —
no secrets). TLS uses `rejectUnauthorized: false`, which works with Supabase out of the box; for strict
CA verification swap it for `ssl: true`.

## Layout

```
bi/
  pages/index.md            ← the Overview dashboard (edit this / add pages here)
  sources/posthog/*.js      ← HogQL queries → tables posthog.event_totals, .events_daily, .funnel
  sources/supabase/*.sql    ← Postgres queries → tables supabase.signups_daily, .reports_daily, …
  evidence.config.yaml      ← theme + enabled datasource plugins
  .env                      ← secrets (gitignored)
```

Add a chart: write a ```sql``` block in a page, then drop a component
(`<LineChart>`, `<BarChart>`, `<BigValue>`, `<DataTable>`) below it.
See the [components docs](https://docs.evidence.dev/components/all-components/).
