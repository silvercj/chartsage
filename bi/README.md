# ChartSage BI (Evidence.dev)

Local, free, code-based BI for ChartSage. SQL + Markdown → interactive charts.
Built with [Evidence](https://evidence.dev) (MIT-licensed; self-hosting/local is free).

Two data sources:
- **`posthog`** — product analytics via the HogQL query API (a JavaScript source).
- **`supabase`** — business metrics from the Postgres DB (a read-only role).

## Run / launch it

```bash
cd bi
npm install                     # first time only
npm run sources                 # pull PostHog + Supabase → local Parquet snapshot
npm run dev -- --port 4000      # launch → http://localhost:4000  (4000, not 3000 — the app owns 3000)
```

- **Evidence is a snapshot, not live** — refresh the numbers by re-running `npm run sources`.
- The dev server **hot-reloads page edits** — leave `npm run dev` running while you work; stop with `Ctrl-C`.
- Node ≥18 (works on Node 23).

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
  pages/index.md            ← the Overview dashboard
  pages/*.md                ← each file becomes a route + sidebar entry
  sources/posthog/*.js      ← HogQL queries → tables posthog.event_totals, .events_daily, .funnel
  sources/supabase/*.sql    ← Postgres queries → tables supabase.signups_daily, .reports_daily, …
  evidence.config.yaml      ← theme + enabled datasource plugins
  .env                      ← secrets (gitignored)
```

## Adding charts & metrics

**The model:** every file in `sources/<src>/` becomes a table named `<src>.<filename>` (e.g.
`sources/supabase/signups_daily.sql` → `supabase.signups_daily`). A page is Markdown containing
named `sql` blocks; you pass a block's results into a chart component via `data={…}`.

> Page (`.md`) edits hot-reload in the browser. **Whenever you add or change a file under
> `sources/`, re-run `npm run sources`** — you can do it in a second terminal while `npm run dev`
> is running, and the page updates when it finishes.

### 1. Add a chart from data that already exists

Edit `pages/index.md` (or any page): add a named SQL block, then a component beneath it. The
block name is how the component finds the data.

````markdown
```sql credits_over_time
select day, credits_purchased
from supabase.purchases_daily
order by day
```

<BarChart data={credits_over_time} x=day y=credits_purchased title="Credits sold per day"/>
````

### 2. Add a new business metric (Postgres)

1. Create `sources/supabase/<name>.sql` — e.g. `report_visibility.sql`:
   ```sql
   select
     case when is_public then 'Public' else 'Unlisted' end as visibility,
     count(*) as reports
   from reports
   group by 1
   ```
2. Run `npm run sources` ← this is what builds the `supabase.report_visibility` table.
3. Use it on a page:

   ````markdown
   ```sql visibility
   select * from supabase.report_visibility
   ```

   <BarChart data={visibility} x=visibility y=reports swapXY=true/>
   ````

### 3. Add a new product metric (PostHog)

Don't hand-write the fetch — copy an existing query file and tweak it:

```bash
cp sources/posthog/event_totals.js sources/posthog/signups_by_day.js
```

Edit the `query` (HogQL) string and the columns it returns, then `npm run sources`. Two rules for
these JS sources:
- **Always return ≥1 row** — the connector reads `data[0]`, so an empty array throws.
- **No backtick `${}`** in the file — Evidence expands `$`; use plain string concatenation
  (the existing files already do).

PostHog dates arrive as strings, so cast them on the page: `select cast(day as date) as day, …`.

### 4. Add a whole new page

Create `pages/revenue.md` — it's served at `/revenue` and added to the sidebar automatically.
Subfolders become sections: `pages/growth/funnel.md` → `/growth/funnel`.

### Components you'll reach for most

| Component | For |
|---|---|
| `<BigValue data={q} value=col/>` | a single headline number (KPI tile) |
| `<LineChart data={q} x= y= series=/>` | trends over time |
| `<BarChart data={q} x= y= series= type=stacked/>` | comparisons / stacked breakdowns |
| `<DataTable data={q} rows=10 search=true/>` | raw rows (add `<Column/>` to format) |
| `<Value data={q} column=col/>` | drop a number inline in a sentence |

Number formatting: `fmt=usd0`, `fmt=pct1`, `fmt='#,##0'`. Full reference:
[components](https://docs.evidence.dev/components/all-components/) ·
[markdown + SQL syntax](https://docs.evidence.dev/core-concepts/syntax/).
