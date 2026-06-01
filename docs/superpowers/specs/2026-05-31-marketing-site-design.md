# ChartSage Marketing Site / Landing + FAQ — Design Spec

**Status:** Approved (via full landing mockup), pending spec review
**Date:** 2026-05-31
**Type:** Marketing sub-project (frontend only)

**Goal:** A marketing landing page that explains ChartSage and converts logged-out visitors into trials/signups — and answers the common questions (FAQ). Inherits the live "Analyst's Instrument" design system.

**Approved direction:** Comprehensive single-page conversion scroller, **dark** marketing shell (it's for logged-out visitors), with a **light** embedded example report (accurately shows real output and pops against the dark). Plain, concrete, non-AI-slop copy.

Reference mockup (gitignored, local): `design/landing.html`.

---

## 1. Routing & structure (the one architectural change)

Today `/` *is* the upload tool. We make `/` **conditional** and move the tool to `/app`.

- **`/app`** (new route) — the **upload tool**: exactly today's `/` page content (dropzone, upload, generate, multi-step progress, cost label, 402→OutOfCreditsModal, anon free-report flow). Used by **both** anonymous (free first report) and logged-in users. Pure move — no behavior change.
- **`/`** (marketing landing) — a **server component** that reads the Supabase session from cookies (via the `@supabase/ssr` server client):
  - **Logged in** → `redirect('/app')` (no marketing flash).
  - **Logged out / anonymous** → render the marketing landing (static, no auth needed).
- **AppHeader** (root layout) — extend its hide rule so it does **not** render on `/` (the marketing page has its own nav). Currently hidden on `/report/*/print`; add `pathname === '/'`. (Logged-in `/` redirects to `/app` before paint, so AppHeader still shows on app routes.)
- **CTA / redirect audit** — every "use the tool" destination becomes `/app`:
  - All marketing CTAs ("Upload a CSV…") → `/app`.
  - `anon-limit` page CTA, post-login redirect default, `welcome` safeNext, and the generate flow's internal `router.push` — audit each; anywhere that meant "the uploader / home tool" now means `/app`. The brand-logo link in AppHeader stays `/` (→ redirects logged-in to `/app`, shows marketing to logged-out — correct).
  - Login `?next=` handling unchanged (still respects an explicit next).

**Marketing page = its own nav + footer** (below), independent of AppHeader.

---

## 2. Page sections (single scroll, anchor nav)

Dark shell, semantic tokens/fonts from the design system. Order:

1. **Nav** (sticky) — brand mark + "ChartSage"; anchor links How it works · Example · Pricing · FAQ; "Sign in" (`/login`); primary "Upload a CSV" (`/app`).
2. **Hero** — eyebrow `CSV → report · no SQL, no pivot tables`; H1 (Fraunces) **"Ten charts and a *written summary*, from one spreadsheet."** (italic teal accent on "written summary"); sub "Upload a CSV or Excel file. ChartSage profiles the data, picks the ten charts that actually say something, and writes the analysis to match."; CTAs **"Upload a CSV — first one's free"** (`/app`) + "See an example ↓" (#example); a light report-card visual.
3. **Trust strip** — mono, factual: `Private to your account` · `Not used to train AI models` · `CSV & Excel` · `Nothing to install, no SQL`. *(All verified true; no deletion claim — none exists.)*
4. **How it works** — "Three steps. About thirty seconds." → (01) **Upload a file**: "Drop in a CSV or Excel export. No cleaning, no formatting rules, no column naming — it reads what you have." (02) **Claude reads it**: "It profiles every column, works out the types and relationships, and chooses the ten charts worth showing for this particular dataset." (03) **You get a report**: "Interactive charts, a written summary of what stands out, and a flag on anything that looks off in the data. Export to PDF or keep it in your account."
5. **Example report** (#example) — "Here's a report it made." + sub "Output from a sample sales dataset — charts, a written summary, and a note on the one column that didn't look right. Nothing staged." → a **real, interactive report** embedded (see §3).
6. **What you get** — "A report, not a chart dump." → 6 feature cards: **Ten charts, chosen for your data** ("Not a fixed template. Claude picks the chart types that fit your columns — distributions, trends, correlations, breakdowns."); **A written summary** ("The headline findings in plain English, up top — so you can read the takeaway before you read the charts."); **A data-quality flag** ("It tells you when something looks wrong — blank fields, odd outliers, mixed types — instead of quietly charting it anyway."); **PDF export** ("One click to a clean PDF you can send as-is."); **More charts on demand** ("Want a different cut? Generate five more and they're added to the report."); **Saved to your account** ("Every report you make stays in one place. Come back to it, re-export it, or pick up where you left off.").
7. **Who it's for** — "Bring the data you already have." + sub "If it opens in a spreadsheet, it works here." → tiles: **Sales** (pipeline · revenue by region · win rates · order values); **Marketing** (campaign performance · channel mix · spend vs. conversion); **Finance & ops** (spend · headcount · throughput · month-over-month); **Research** (survey responses · breakdowns · correlations between fields).
8. **Pricing** (#pricing) — "Your first report is free." + sub "No card, no account. Make an account and you get credits to keep going." → blurb "Try it, then top up later." / "Reports and extra charts run on credits, so you only pay for what you make. Paid top-ups are coming soon — for now, the free report plus your starter credits are enough to put it through its paces." + CTA `/app`; table: First report → **Free · no account**; New account starts with → **300 credits**; A report → **100 credits**; Five more charts → **40 credits**; Paid top-ups → **Coming soon** (muted). *(Numbers must read from the same source of truth as the app: `REPORT_COST=100`, `GENERATE_MORE_COST=40`, `SIGNUP_GRANT=300`.)*
9. **FAQ** (#faq) — "Questions worth asking." → 6 Q&As (verbatim):
   - **What files can I upload?** "CSV and Excel (.xlsx). Drop in an export from your database, CRM, analytics tool, or just a spreadsheet you've been keeping by hand."
   - **What happens to my data?** "Your file is stored privately and used only to build your report. It's tied to your account, isn't shared, and isn't used to train AI models. We don't sell it and we don't look at it." *(Confirmed acceptable by owner.)*
   - **How accurate are the charts?** "The charts are drawn from your actual numbers — they're your numbers, plotted. The written summary is generated by Claude reading those numbers: usually sharp, occasionally worth a second look. Treat it as a fast first pass, not the final word."
   - **How do credits work?** "Your first report is free with no account. A new account starts with 300 credits; a report costs 100 and generating five more charts costs 40. Paid top-ups are on the way."
   - **Do I need to know SQL or set anything up?** "No. There's nothing to install and no query to write. Upload a file in the browser and you have a report."
   - **Can I share or export it?** "Every report exports to a clean PDF. Saved reports stay in your account so you can come back and re-export any time."
10. **Closing CTA** — "Stop screenshotting spreadsheets." / "Upload one file and see what's in it. The first report is on us." + "Upload a CSV — free" (`/app`); subtle teal glow.
11. **Footer** — brand + links (How it works · Example · Pricing · FAQ · Sign in) + `© ChartSage`.

Entrance motion: light staggered reveals (respect `prefers-reduced-motion`). Smooth-scroll anchor nav.

---

## 3. The embedded example report

The strongest conversion element, so it must be the **real renderer**, not a screenshot:

- Commit a **sample report JSON fixture** (`src/app/(marketing)/_data/sample-report.json` or similar) — ideally captured from a real generated report (a clean sales-style dataset) for authenticity.
- Render it with the **existing report components** (`ChartCard` + `charts/*`, light `.theme-light` scope) in a **read-only embed**: no Toolbar, no dnd-kit, no API calls — just the charts + summary + data-quality note. Reuses the post-redesign chart theme, so it looks identical to a real report.
- Keep it to ~3–4 charts + summary + the data-quality line for page weight; it's a teaser, not the full 10.

---

## 4. Design system & componentization

- Inherits the live tokens/fonts (dark default; `.theme-light` for the embedded report). No new colors/fonts.
- New marketing components live together under a marketing route group, e.g. `src/app/(marketing)/` with `MarketingNav`, `Hero`, `TrustStrip`, `HowItWorks`, `ExampleReport`, `Features`, `UseCases`, `Pricing`, `Faq`, `ClosingCta`, `MarketingFooter` — small, focused, one section each. Section copy lives in the components (or a single `content.ts` constants file for easy editing).
- Pricing/credit numbers imported from the existing `src/app/lib/credits.ts` constants (single source of truth) — never hard-coded.

---

## 5. SEO & analytics

- **Metadata:** route-level `metadata` for `/` — title (e.g. "ChartSage — Ten charts and a written summary, from one spreadsheet"), description, canonical; OpenGraph + Twitter tags (title/description/image). OG image: a static asset (a report-card render) — placeholder acceptable for v1, flagged for a real one.
- **Analytics (light):** PostHog `$pageview` is already auto-captured. Add a `marketing_cta_clicked` event with a `{ location }` property (hero / pricing / closing / nav) on the primary CTAs, so the funnel (landing → /app → report) is measurable. No other new events.

---

## 6. Scope & non-goals

**In scope:** the marketing landing at `/`, the `/app` move + conditional redirect + CTA/redirect audit, the read-only example-report embed, SEO/meta, light CTA analytics. Frontend only.

**Non-goals:**
- No backend/API changes. No changes to the upload/generate/credit/auth logic (the tool is *moved*, not modified).
- Separate `/pricing` and `/faq` routes — single page with anchors for v1.
- Testimonials / social proof — none exist pre-launch.
- Payments (SP4) and a data-deletion / retention feature (flagged as future) — out of scope; copy avoids promising either.
- A blog / docs / multi-page marketing site.

---

## 7. Verification

- `npx tsc --noEmit` clean; `npm run build` exit 0 (all routes incl. new `/app`).
- Manual: logged-out `/` shows the landing; logged-in `/` redirects to `/app`; anonymous `/app` can upload + get the free report; every CTA lands on `/app`; the embedded example renders (light) with real charts; anchor nav scrolls; PDF/export references accurate.
- Behavior regression: login, anon free report, generate, generate-more, credits, sign out — all unchanged (the tool moved to `/app` intact).
- Pricing numbers on the page match `credits.ts`.
- Lighthouse/meta sanity: title + description + OG present on `/`.
