# ChartSage Publish / Shareable Reports (SP5-A) — Design Spec

**Status:** Approved (design), pending spec review
**Date:** 2026-06-03
**Type:** Backend + frontend feature (SEO growth — the backlink/share engine)

**Goal:** Let a signed-in report owner **publish** a report to a public, **search-indexable, embeddable** page with a per-report social preview image — turning ChartSage's own output into the compounding SEO + backlink + share engine the marketing strategy identified as the #1 growth lever. Everything not explicitly published stays exactly as today (viewable only by its secret link) **and is now explicitly `noindex`**.

**Context:** From `docs/marketing-strategy.md`: at a £25/week budget with one-time pricing, paid acquisition is structurally unviable; growth must be organic/product-led, and ChartSage's auto-generated dashboards make free-tool/programmatic SEO genuinely viable (the Canva / Zapier / HubSpot / remove.bg pattern). **Current state (verified in code):** `GET /report/{id}` (`main.py:439`) has **no auth/ownership check** — any report is already viewable by anyone holding its UUID link, and there is no `noindex`, so a leaked link could be indexed. The report page (`src/app/report/[id]/page.tsx`) is a `'use client'` component that fetches `/report/{id}` on mount. Reports live in the `reports` table (`id`, `anon_id`, `user_id`, `report_json`, `csv_storage_key`, `title`, `created_at`, `updated_at`); the backend reads via the service-role client (RLS is not the gate — the API layer is). Playwright rendering already exists (`pdf_export.py`, `render_chart_images` from the export feature) and is reused here for the OG image.

> This is **SP5-A**. SP5-B (intent-matched tool landing pages) and SP5-C (homepage positioning copy) are separate cycles after this.

**Approved decisions:**
- **Visibility model:** unlisted-by-link **(default, now `noindex`)** + explicit owner opt-in **Publish** → `public` (indexed, OG preview, in sitemap, embeddable). No new "owner-only private" state in v1 (that's a separate future hardening).
- **Publishing is owner-gated:** requires a signed-in owner; anonymous reports must be **claimed** first.
- **Social preview:** a **per-report chart preview image**, generated on publish by reusing the existing Playwright renderer.
- **SEO architecture:** server-render the report page's **metadata only** (Next `generateMetadata`); keep the chart body client-rendered (Google renders JS; only meta + robots must be server-side).

---

## Facet 1 — Data model

Migration `docs/migrations/publish.sql` (run manually in Supabase, like the others):
```sql
alter table reports add column if not exists is_public    boolean not null default false;
alter table reports add column if not exists og_image_key text;
alter table reports add column if not exists published_at  timestamptz;
create index if not exists reports_public_idx on reports (is_public, updated_at desc) where is_public;
```
`is_public=false` (default) = today's unlisted-by-link behavior, now explicitly `noindex`. No backfill needed (existing reports default to unlisted).

## Facet 2 — Backend endpoints (`main.py`)

All publish/unpublish endpoints are **owner-gated**: `get_identity` must be authenticated **and** `row["user_id"] == str(identity.user_id)`, else `403 {code:"NOT_OWNER"}` (anon → `401 {code:"AUTH_REQUIRED"}`; missing report → `404`).

- **`POST /report/{id}/publish`** — owner-gated. Generates the OG image (Facet 3), uploads it, then `db.set_report_visibility(id, is_public=True, og_image_key=key, published_at=now)`. Returns `{public_url, embed_url, og_image_url}`. Idempotent (re-publish regenerates the image + url).
- **`POST /report/{id}/unpublish`** — owner-gated → `db.set_report_visibility(id, is_public=False)`. (Leave the og image blob; it's harmless and `noindex` resumes immediately.) Returns `{ok: true}`.
- **`GET /report/{id}/meta`** (public, **optional identity**) — returns `{is_public, title, description, og_image_url, owned}` for Next `generateMetadata` + the share UI. **It must NOT 400 on missing auth headers**, because Next's server-side `generateMetadata` calls it with none — so add a soft `get_identity_optional` dependency (valid Bearer → user; otherwise anonymous; never raises). `title`/`description` derive from the report's `title` + a one-line narrative summary. `owned` is true **only when an authenticated owner calls it** (the Toolbar's client-side `apiFetch` sends the Bearer); the server-side `generateMetadata` call gets `owned=false`, which is fine — metadata doesn't need it. Leaves the existing `GET /report/{id}` (returns `report_json`) **unchanged**.
- **`GET /reports/public`** (public, cacheable) — `[{id, updated_at}]` for the sitemap (public reports only, capped/paginated).

**`db.py` additions:** `set_report_visibility(report_id, is_public, og_image_key=None, published_at=None) -> bool`; `list_public_reports(limit=5000) -> list[dict]`. `get_report` already returns the row (now incl. the new columns). **FakeDB** mirrors these (an `is_public`/`og_image_key` on the in-memory row + the two methods).

## Facet 3 — OG preview image (reuse Playwright)

On publish, render a **1200×630 PNG** of the report's headline chart(s) and upload it to a **public** Supabase Storage bucket (e.g. `og-images`, public-read), keyed by report id; store the key in `og_image_key`; expose its public URL via `og_image_url`.

Implementation: **reuse the embed view as the capture target** — `/publish` sets `is_public=true` **first**, then Playwright loads `{FRONTEND_BASE_URL}/report/{id}/embed` at a 1200×630 viewport and screenshots it (the embed view, Facet 4, is already a clean chrome-less chart render and serves public reports, so it must be public before capture). This keeps one rendering path for both iframe embeds and OG capture. Falls back to the generic ChartSage OG card if rendering fails (publish still succeeds). Reuses `pdf_export.py`'s Playwright browser/session + `storage.py` upload pattern (add `storage.upload_public_image(key, png_bytes) -> public_url`).

## Facet 4 — Frontend (Next 14)

- **Report page** (`src/app/report/[id]/page.tsx`): restructure into a **server component** that (a) exports `async function generateMetadata({params})` — fetches `${API}/report/${id}/meta` server-side and returns: if `is_public` → real `title`/`description`, `openGraph` + `twitter` (`summary_large_image`) with `og_image_url`, canonical = the public report URL, `robots: { index: true, follow: true }`; else → `robots: { index: false, follow: false }` (noindex); and (b) renders the existing client view, moved verbatim into a `'use client'` child `ReportClient.tsx`. **No change to the interactive behavior** — only the `<head>` becomes server-correct.
- **Toolbar** (`src/app/report/[id]/Toolbar.tsx`, owner only — gated on `meta.owned`): a **"Share / Publish"** control → a confirmation modal (*"Publishing makes this report and its charts public and indexable by search engines. Your uploaded file is never shared. You can make it private again anytime."*) → on confirm calls `/publish`, then shows the **public link** (copy) + an **`<iframe>` embed snippet** (copy) + a **"Make private"** toggle (calls `/unpublish`). Fires PostHog `report_published` / `report_unpublished`.
- **Embed view** (`src/app/report/[id]/embed/page.tsx`): minimal, chrome-less render of the report's charts for iframes; serves **public reports only** (non-public → a small "This report isn't public" placeholder, no data). Metadata: `robots: noindex` + the **`indexifembedded`** directive so it passes SEO value to the embedding page without ranking standalone. Must be **frame-embeddable** (do not send `X-Frame-Options: DENY`; allow `frame-ancestors`). Fires `embed_viewed`.
- **Sitemap** (`src/app/sitemap.ts`): static marketing routes + (SP5-B later) tool pages + public report URLs from `GET /reports/public`.
- **`robots.txt`** (`src/app/robots.ts` if not present): allow crawling; point to the sitemap.

## Facet 5 — Privacy guarantee (the crux)

Nothing becomes indexable, sitemapped, or embeddable until the **owner explicitly publishes** (signed-in, with the warning). Unlisted-by-link behavior is **unchanged** from today; default reports now additionally carry `noindex` (a strict improvement — a leaked link can no longer be indexed). Publishing exposes only the `report_json` (the charts + narrative the owner already sees) — **never the raw uploaded CSV** (it stays in private storage, untouched by this feature).

## Dependencies & schema
- **No new backend deps** (Playwright + supabase storage already present). One new **public Storage bucket** (`og-images`) — created during provisioning.
- **Schema:** the three `reports` columns above (manual migration). No other tables.

## Phasing (plan order)
1. **Migration + DB layer** — `publish.sql` + `set_report_visibility` / `list_public_reports` + FakeDB. TDD on FakeDB.
2. **Publish/unpublish/meta/public endpoints** — owner-gating + the meta + public-list endpoints. Backend TDD (Stripe-style: owner 200 / non-owner 403 / anon 401; meta reflects visibility; public list filters).
3. **OG image generation** — `storage.upload_public_image` + Playwright capture of the embed view on publish (wired into `/publish`). TDD with the renderer mocked.
4. **Embed view route** — chrome-less public-only render + frame headers + `indexifembedded`. tsc.
5. **Report page `generateMetadata`** (server) + `ReportClient.tsx` split + Toolbar publish/share UI + confirmation. tsc.
6. **`sitemap.ts` + `robots.ts`.** tsc.
7. **Build + QA + deploy** — full `pytest` + `next build` (Vercel gate); create the `og-images` public bucket + run `publish.sql` in Supabase; deploy backend (Cloud Run, `CLOUDSDK_PYTHON=3.12`) + frontend (Vercel). Smoke: publish a report → public URL gets `index` meta + OG image + appears in `/sitemap.xml`; a non-public report has `noindex`; embed renders in an iframe; `unpublish` flips it back. Production deploy requires explicit user authorization.

## Scope & non-goals
**In scope:** the visibility column; publish/unpublish/meta/public-list endpoints (owner-gated); per-report OG image (Playwright); server-rendered report metadata (index vs noindex); the owner share/publish UI + embed snippet; the embed view; sitemap + robots; analytics events.

**Non-goals (v1):**
- A **public discovery gallery** / browse page (later; the strategy's "example gallery" lever).
- **Custom share slugs** (keep the UUID URL).
- **Per-chart** embeds (whole-report embed only).
- **OG image regeneration on every edit** (regenerate on publish only; an edited-but-not-re-published report may show a slightly stale preview).
- A true **owner-only "private"** visibility state (separate future hardening; `GET /report/{id}` stays open-by-link as today).
- SP5-B (tool landing pages) and SP5-C (positioning copy) — separate specs.

## Verification
- **Backend TDD:** publish → owner 200 (sets is_public + og_image_key) / non-owner 403 `NOT_OWNER` / anon 401 `AUTH_REQUIRED` / missing 404; unpublish flips is_public false (owner-gated same way); `GET /meta` returns `is_public` + `owned` correctly for owner vs other vs anon; `GET /reports/public` returns only public ids; OG generation invoked on publish (renderer mocked); `FakeDB` visibility methods. Full `pytest` green.
- **Frontend:** `tsc` + `next build` clean; a **public** report's page emits `index` + OpenGraph/Twitter meta with the per-report image; a **non-public** report emits `noindex`; the embed route renders charts for a public report and refuses a non-public one; `/sitemap.xml` lists public reports.
- **Live (post-deploy):** publish a real report → confirm (a) the public URL's `<head>` shows index + the OG image (view-source / a social-preview debugger), (b) it appears in `/sitemap.xml`, (c) the `<iframe>` embed renders on a third-party page, (d) a non-published report shows `noindex`, (e) `unpublish` reverts it. `report_published` fires in PostHog.
