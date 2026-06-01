# ChartSage Marketing Site — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
>
> **Subagents must NOT run `git checkout`, `git switch`, `git reset`, or `git stash`.** Stay on the current branch (`marketing-site`); commit only the files each task lists.

**Goal:** A dark marketing landing page at `/` for logged-out visitors (logged-in users redirect to the app), explaining ChartSage and converting visitors to try it — with the upload tool moved to `/app`.

**Architecture:** `/` becomes a server component that checks the Supabase session and either `redirect('/app')` (logged-in) or renders the marketing landing (logged-out). The current uploader moves verbatim to `/app`. Marketing sections are small client components composed into `MarketingLanding`, with copy in one `content.ts`. The example report reuses the real chart components (light, read-only) fed a synthetic JSON fixture. Inherits the live design system; no backend changes.

**Tech Stack:** Next.js 14 App Router, `@supabase/ssr`, Tailwind (existing token system), ECharts (existing chart components), PostHog.

**Spec:** `docs/superpowers/specs/2026-05-31-marketing-site-design.md`
**Visual + copy reference:** `design/landing.html` (the approved mockup — exact markup, styling, and copy to port).
**Branch:** `marketing-site` (already checked out; carries the spec).
**Verification:** frontend, no styling unit tests — each task: implement → `npx tsc --noEmit` clean → commit; final `npm run build` + visual QA.

---

## File Structure

**New**
- `src/app/lib/supabase-server.ts` — server-side Supabase client (reads session from cookies).
- `src/app/app/page.tsx` — the upload tool, moved from `src/app/page.tsx` (imports re-pathed).
- `src/app/components/marketing/content.ts` — all marketing copy (single source of truth).
- `src/app/components/marketing/MarketingLanding.tsx` — composes the sections.
- `src/app/components/marketing/{MarketingNav,Hero,TrustStrip,HowItWorks,ExampleReport,Features,UseCases,Pricing,Faq,ClosingCta,MarketingFooter}.tsx` — one section each.
- `src/app/components/marketing/sampleReport.ts` — synthetic example-report fixture (typed as `Report`).
- `src/app/report/[id]/charts/ChartContent.tsx` — extracted kind→component switch (shared by ChartCard + ExampleReport).

**Modified**
- `src/app/page.tsx` — replaced with the conditional server component (redirect or marketing).
- `src/app/components/AppHeader.tsx` — also hide on `/`.
- `src/app/report/[id]/ChartCard.tsx` — import the extracted `ChartContent`.
- CTA/redirect audit: `src/app/anon-limit/page.tsx`, `src/app/reports/page.tsx` (+ verify `welcome`, `login`) — repoint any "home = the tool" link to `/app`.

**Semantic classes available** (from the redesign; dark by default): `bg-canvas`, `bg-surface`/`bg-surface-2`, `text-ink`/`text-ink-2`/`text-ink-3`, `border-line`/`border-line-2`, `bg-accent`/`text-on-accent`/`text-accent`, `text-ember`, `font-display`/`font-mono`, `card`, `btn`/`btn-primary`/`btn-ghost`, `eyebrow`, `shadow-card`/`shadow-card-lg`/`shadow-glow`, `.theme-light` (scope for the example report).

---

### Task 1: Routing foundation — server client, move tool to /app, conditional `/`

**Files:**
- Create: `src/app/lib/supabase-server.ts`
- Move: `src/app/page.tsx` → `src/app/app/page.tsx` (re-path imports)
- Create/replace: `src/app/page.tsx` (conditional server component)
- Create: `src/app/components/marketing/MarketingLanding.tsx` (temporary stub; fleshed out in Tasks 2–5)
- Modify: `src/app/components/AppHeader.tsx`

- [ ] **Step 1: Server Supabase client** — create `src/app/lib/supabase-server.ts`

```ts
import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';

// Read-only server client for Server Components (session check / redirects).
export function getSupabaseServer() {
  const cookieStore = cookies();
  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        get(name: string) {
          return cookieStore.get(name)?.value;
        },
        set() {},    // no-op: Server Components can't set cookies
        remove() {},
      },
    },
  );
}
```

- [ ] **Step 2: Move the upload tool to `/app`**

```bash
mkdir -p src/app/app
git mv src/app/page.tsx src/app/app/page.tsx
```
Then in `src/app/app/page.tsx`, fix the four relative imports (one level deeper now):
- `'./lib/api'` → `'../lib/api'`
- `'./components/OutOfCreditsModal'` → `'../components/OutOfCreditsModal'`
- `'./lib/useCredits'` → `'../lib/useCredits'`
- `'./lib/credits'` → `'../lib/credits'`

Nothing else in that file changes (it stays the `'use client'` uploader, identical behavior). *(Optional, low-risk copy fix: its hero says "5–7 charts … under 10 seconds"; the app makes 10 charts. Leave as-is unless trivial — out of scope for this move.)*

- [ ] **Step 3: Temporary MarketingLanding stub** — create `src/app/components/marketing/MarketingLanding.tsx`

```tsx
export default function MarketingLanding() {
  return (
    <main className="min-h-screen bg-canvas text-ink flex items-center justify-center">
      <p className="font-display text-2xl">ChartSage — landing coming together…</p>
    </main>
  );
}
```

- [ ] **Step 4: Conditional `/`** — create `src/app/page.tsx`

```tsx
import { redirect } from 'next/navigation';
import { getSupabaseServer } from './lib/supabase-server';
import MarketingLanding from './components/marketing/MarketingLanding';

export default async function Home() {
  const supabase = getSupabaseServer();
  const { data: { user } } = await supabase.auth.getUser();
  if (user) redirect('/app');
  return <MarketingLanding />;
}
```

- [ ] **Step 5: Hide AppHeader on `/`** — in `src/app/components/AppHeader.tsx`, extend the early-return so the app header never shows on the marketing page (it has its own nav). Add, alongside the existing print check:

```tsx
  // Marketing landing has its own nav; app header is for app routes only.
  if (pathname === '/') return null;
```

- [ ] **Step 6: Verify**

Run: `npx tsc --noEmit`
Expected: clean. (`/app` = the tool; `/` = redirect-or-stub; logged-in `/` → `/app`.)

- [ ] **Step 7: Commit**

```bash
git add src/app/lib/supabase-server.ts src/app/app/page.tsx src/app/page.tsx src/app/components/marketing/MarketingLanding.tsx src/app/components/AppHeader.tsx
git commit -m "feat(marketing): move tool to /app; conditional / (redirect logged-in, else landing)"
```

---

### Task 2: Copy constants + Nav + Hero + Trust strip

**Files:**
- Create: `src/app/components/marketing/content.ts`
- Create: `src/app/components/marketing/MarketingNav.tsx`, `Hero.tsx`, `TrustStrip.tsx`
- Modify: `src/app/components/marketing/MarketingLanding.tsx`

- [ ] **Step 1: Copy constants** — create `content.ts` (verbatim approved copy; single source of truth)

```ts
import { REPORT_COST, GENERATE_MORE_COST, SIGNUP_GRANT } from '../../lib/credits';

export const HERO = {
  eyebrow: 'CSV → report · no SQL, no pivot tables',
  // headline rendered with an italic accent on "written summary"
  headlineA: 'Ten charts and a ',
  headlineEm: 'written summary',
  headlineB: ', from one spreadsheet.',
  sub: 'Upload a CSV or Excel file. ChartSage profiles the data, picks the ten charts that actually say something, and writes the analysis to match.',
  ctaPrimary: "Upload a CSV — first one's free",
  ctaSecondary: 'See an example ↓',
};

export const TRUST = [
  'Private to your account',
  'Not used to train AI models',
  'CSV & Excel',
  'Nothing to install, no SQL',
];

export const HOW = [
  { n: '01', h: 'Upload a file', p: 'Drop in a CSV or Excel export. No cleaning, no formatting rules, no column naming — it reads what you have.' },
  { n: '02', h: 'Claude reads it', p: 'It profiles every column, works out the types and relationships, and chooses the ten charts worth showing for this particular dataset.' },
  { n: '03', h: 'You get a report', p: 'Interactive charts, a written summary of what stands out, and a flag on anything that looks off in the data. Export to PDF or keep it in your account.' },
];

export const FEATURES = [
  { h: 'Ten charts, chosen for your data', p: 'Not a fixed template. Claude picks the chart types that fit your columns — distributions, trends, correlations, breakdowns.' },
  { h: 'A written summary', p: 'The headline findings in plain English, up top — so you can read the takeaway before you read the charts.' },
  { h: 'A data-quality flag', p: 'It tells you when something looks wrong — blank fields, odd outliers, mixed types — instead of quietly charting it anyway.' },
  { h: 'PDF export', p: 'One click to a clean PDF you can send as-is.' },
  { h: 'More charts on demand', p: "Want a different cut? Generate five more and they're added to the report." },
  { h: 'Saved to your account', p: 'Every report you make stays in one place. Come back to it, re-export it, or pick up where you left off.' },
];

export const USES = [
  { h: 'Sales', p: 'pipeline · revenue by region · win rates · order values' },
  { h: 'Marketing', p: 'campaign performance · channel mix · spend vs. conversion' },
  { h: 'Finance & ops', p: 'spend · headcount · throughput · month-over-month' },
  { h: 'Research', p: 'survey responses · breakdowns · correlations between fields' },
];

export const PRICING = {
  blurbH: 'Try it, then top up later.',
  blurbP: 'Reports and extra charts run on credits, so you only pay for what you make. Paid top-ups are coming soon — for now, the free report plus your starter credits are enough to put it through its paces.',
  rows: [
    { k: 'First report', v: 'Free · no account', muted: false },
    { k: 'New account starts with', v: `${SIGNUP_GRANT} credits`, muted: false },
    { k: 'A report', v: `${REPORT_COST} credits`, muted: false },
    { k: 'Five more charts', v: `${GENERATE_MORE_COST} credits`, muted: false },
    { k: 'Paid top-ups', v: 'Coming soon', muted: true },
  ],
};

export const FAQ = [
  { q: 'What files can I upload?', a: "CSV and Excel (.xlsx). Drop in an export from your database, CRM, analytics tool, or just a spreadsheet you've been keeping by hand." },
  { q: 'What happens to my data?', a: "Your file is stored privately and used only to build your report. It's tied to your account, isn't shared, and isn't used to train AI models. We don't sell it and we don't look at it." },
  { q: 'How accurate are the charts?', a: 'The charts are drawn from your actual numbers — they’re your numbers, plotted. The written summary is generated by Claude reading those numbers: usually sharp, occasionally worth a second look. Treat it as a fast first pass, not the final word.' },
  { q: 'How do credits work?', a: `Your first report is free with no account. A new account starts with ${SIGNUP_GRANT} credits; a report costs ${REPORT_COST} and generating five more charts costs ${GENERATE_MORE_COST}. Paid top-ups are on the way.` },
  { q: 'Do I need to know SQL or set anything up?', a: 'No. There’s nothing to install and no query to write. Upload a file in the browser and you have a report.' },
  { q: 'Can I share or export it?', a: 'Every report exports to a clean PDF. Saved reports stay in your account so you can come back and re-export any time.' },
];

export const CLOSING = {
  h: 'Stop screenshotting spreadsheets.',
  p: 'Upload one file and see what’s in it. The first report is on us.',
  cta: 'Upload a CSV — free',
};

export const APP_HREF = '/app';
```

- [ ] **Step 2: Build `MarketingNav.tsx`, `Hero.tsx`, `TrustStrip.tsx`**

Port the `<nav>`, hero `<header>`, and trust strip from `design/landing.html` into these client components, translating the mockup's inline styles to the semantic classes (table above). Use `content.ts` for all text. Specifics:
- **MarketingNav:** brand mark (the 3-bar SVG from `AppHeader`) + "ChartSage" (`font-display`); anchor links `#how`/`#example`/`#pricing`/`#faq`; "Sign in" → `/login`; primary "Upload a CSV" → `APP_HREF`. Sticky, `bg-canvas/80 backdrop-blur border-b border-line`.
- **Hero:** `eyebrow`, `font-display` H1 with `<em className="italic font-medium text-accent">{HERO.headlineEm}</em>`, sub, the two CTAs (primary → `APP_HREF`), and the light report-card visual (port the `.report` card markup from the mockup; wrap it in `theme-light` so its `bg-surface`/`text-ink` resolve light; the inline SVG area chart can be copied as-is).
- **TrustStrip:** the four `TRUST` items, mono, with the teal dot bullets, on `bg-surface-2` between hairlines.

- [ ] **Step 3: Compose** — update `MarketingLanding.tsx` to render `<MarketingNav/> <Hero/> <TrustStrip/>` inside a `bg-canvas text-ink` root.

- [ ] **Step 4: Verify + commit**

```bash
npx tsc --noEmit
git add src/app/components/marketing/
git commit -m "feat(marketing): copy constants, nav, hero, trust strip"
```

---

### Task 3: How it works + Features + Use cases

**Files:**
- Create: `src/app/components/marketing/HowItWorks.tsx`, `Features.tsx`, `UseCases.tsx`
- Modify: `MarketingLanding.tsx`

- [ ] **Step 1: Build the three sections** from `design/landing.html` (the `#how` band, the "What you get" band, the "Who it's for" band), using `content.ts` (`HOW`, `FEATURES`, `USES`) and semantic classes.
  - **HowItWorks** (`id="how"`): eyebrow "How it works", `font-display` heading "Three steps. About thirty seconds.", a 3-col grid of `step` cards (mono number, `font-display` h3, `text-ink-2` p).
  - **Features:** eyebrow "What you get", heading "A report, not a chart dump.", a 3-col grid of 6 `fcard`s (icon tile `bg-surface-2`, `font-display` h3, `text-ink-2` p). Icons: reuse simple glyphs/SVGs (the mockup uses placeholder glyphs — a small inline SVG per card is fine).
  - **UseCases:** eyebrow "Who it's for", heading "Bring the data you already have.", sub, a 4-col grid of `ucard`s (`font-display` h4, `font-mono text-ink-3` p), on `bg-surface-2` between hairlines.

- [ ] **Step 2: Compose** — add `<HowItWorks/> <Features/> <UseCases/>` to `MarketingLanding` (order per spec §2).

- [ ] **Step 3: Verify + commit**

```bash
npx tsc --noEmit
git add src/app/components/marketing/
git commit -m "feat(marketing): how-it-works, features, use-cases sections"
```

---

### Task 4: Example report (real charts, read-only)

**Files:**
- Create: `src/app/report/[id]/charts/ChartContent.tsx` (extracted)
- Modify: `src/app/report/[id]/ChartCard.tsx` (use the extracted component)
- Create: `src/app/components/marketing/sampleReport.ts`
- Create: `src/app/components/marketing/ExampleReport.tsx`
- Modify: `MarketingLanding.tsx`

- [ ] **Step 1: Extract `ChartContent`** — create `src/app/report/[id]/charts/ChartContent.tsx` with the exact kind→component switch currently inside `ChartCard.tsx` (the `dynamic(() => import('./BarChart'))` imports and the `ChartContent({ spec })` switch). Note paths shift: inside `charts/`, imports are `'./BarChart'` etc. (drop the `charts/` prefix).

```tsx
'use client';
import dynamic from 'next/dynamic';

const BarChart = dynamic(() => import('./BarChart'), { ssr: false });
const HistogramChart = dynamic(() => import('./HistogramChart'), { ssr: false });
const ScatterChart = dynamic(() => import('./ScatterChart'), { ssr: false });
const LineChart = dynamic(() => import('./LineChart'), { ssr: false });
const PieChart = dynamic(() => import('./PieChart'), { ssr: false });
const BoxPlot = dynamic(() => import('./BoxPlot'), { ssr: false });
const Heatmap = dynamic(() => import('./Heatmap'), { ssr: false });

export default function ChartContent({ spec }: { spec: any }) {
  switch (spec.kind) {
    case 'bar': return <BarChart spec={spec} />;
    case 'histogram': return <HistogramChart spec={spec} />;
    case 'scatter': return <ScatterChart spec={spec} />;
    case 'line': return <LineChart spec={spec} />;
    case 'pie': return <PieChart spec={spec} />;
    case 'box': return <BoxPlot spec={spec} />;
    case 'heatmap': return <Heatmap spec={spec} />;
    default: return <p className="text-sm text-ember">Unsupported chart kind: {String(spec.kind)}</p>;
  }
}
```

- [ ] **Step 2: Use it in `ChartCard.tsx`** — delete the local `dynamic(...)` chart imports and the local `ChartContent` function; add `import ChartContent from './charts/ChartContent';`. Leave everything else (the dnd-kit `useSortable`, the card markup, `<ChartContent spec={spec} />` call) unchanged.

- [ ] **Step 3: Synthetic fixture** — create `sampleReport.ts`. Author a believable **synthetic sales** report (do NOT use any real user data). Use 3 charts. The `line` and `bar` spec shapes below are confirmed against the components; for the 3rd, open `src/app/report/[id]/charts/PieChart.tsx` and match its expected `spec` fields exactly.

```ts
import type { Report } from '../../report/[id]/useReportLayout';

export const SAMPLE_REPORT: Report = {
  generated_at: '2024-12-31T00:00:00Z',
  summary: 'Revenue grew 22% year over year, led by the West region at 31% of total. Order volume rose alongside average order value, and Q4 closed above target. One data note: 8% of rows have a blank discount_code.',
  data_quality: ['8% of rows in "discount_code" are blank — those orders are counted as "none".'],
  charts: [
    {
      chart_id: 'c1',
      caption: 'Revenue rose steadily through the year and closed December above target.',
      spec: {
        kind: 'line', title: 'Revenue by month',
        x: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
        y: [320, 340, 360, 355, 390, 410, 430, 425, 470, 510, 560, 620],
        x_label: 'Month', y_label: 'Revenue ($k)', y_display_type: 'number',
      },
    },
    {
      chart_id: 'c2',
      caption: 'The West region leads at roughly a third of total revenue.',
      spec: {
        kind: 'bar', title: 'Revenue by region',
        x: ['West','East','North','South','Intl'],
        y: [1490, 1180, 940, 760, 430],
        x_label: 'Region', y_label: 'Revenue ($k)', y_display_type: 'number',
      },
    },
    // c3: a pie/composition chart — match the exact spec fields to PieChart.tsx
    // (e.g. { kind:'pie', title:'Orders by channel', data:[{name, value}, ...] }).
  ],
  layout: [
    { chart_id: 'c1', position: 'main', order: 0 },
    { chart_id: 'c2', position: 'main', order: 1 },
    { chart_id: 'c3', position: 'main', order: 2 },
  ],
  metadata: {},
};
```
After confirming the pie shape, add the `c3` chart object to `charts`.

- [ ] **Step 4: ExampleReport component** — create `ExampleReport.tsx`. Render a read-only light report: wrap in `theme-light`, show the section eyebrow/heading/sub (`content.ts` could hold these or inline per spec §2), then a light report card containing the `summary` (`font-display`), each chart in a simple read-only card via `ChartContent`, and the `data_quality` note (`font-display italic text-ink-2`). NO Toolbar, NO dnd-kit, NO `ChartCard` (it requires sortable context).

```tsx
'use client';
import ChartContent from '../../report/[id]/charts/ChartContent';
import { SAMPLE_REPORT } from './sampleReport';

export default function ExampleReport() {
  const r = SAMPLE_REPORT;
  return (
    <section id="example" className="border-y border-line bg-surface-2/40 py-20">
      <div className="theme-light max-w-3xl mx-auto px-6">
        {/* eyebrow + "Here's a report it made." + sub (dark-section headings sit OUTSIDE theme-light;
            put the headings above this wrapper if you want them on the dark band — see design/landing.html) */}
        <div className="card shadow-card-lg rounded-2xl p-6 bg-surface text-ink">
          <p className="font-display text-2xl mb-2">Regional Sales — FY24</p>
          <p className="font-mono text-xs uppercase tracking-wide text-ink-3 mb-5">Generated · sample · {r.charts.length} charts</p>
          <p className="text-ink-2 mb-6">{r.summary}</p>
          <div className="grid gap-6">
            {r.charts.map((c) => (
              <div key={c.chart_id} className="border border-line rounded-xl p-4">
                <div className="min-h-[260px]"><ChartContent spec={c.spec} /></div>
                {c.caption && <p className="font-display italic text-ink-2 text-sm mt-3">{c.caption}</p>}
              </div>
            ))}
          </div>
          {r.data_quality.length > 0 && (
            <p className="mt-6 font-mono text-xs text-ink-3">Data note: {r.data_quality[0]}</p>
          )}
        </div>
      </div>
    </section>
  );
}
```
(Match the section heading treatment to spec §2 / `design/landing.html` — the eyebrow "A real one" + "Here's a report it made." + sub belong on the dark band above the `theme-light` card; adjust wrapper accordingly.)

- [ ] **Step 5: Compose** — add `<ExampleReport/>` to `MarketingLanding` in the section-5 slot.

- [ ] **Step 6: Verify + commit**

```bash
npx tsc --noEmit
git add "src/app/report/[id]/charts/ChartContent.tsx" "src/app/report/[id]/ChartCard.tsx" src/app/components/marketing/
git commit -m "feat(marketing): real read-only example report (extracted ChartContent + synthetic fixture)"
```

---

### Task 5: Pricing + FAQ + Closing CTA + Footer

**Files:**
- Create: `src/app/components/marketing/Pricing.tsx`, `Faq.tsx`, `ClosingCta.tsx`, `MarketingFooter.tsx`
- Modify: `MarketingLanding.tsx`

- [ ] **Step 1: Build the four sections** from `design/landing.html` + `content.ts`:
  - **Pricing** (`id="pricing"`): eyebrow "Pricing", heading "Your first report is free.", sub; the two-column card — left = `PRICING.blurbH`/`blurbP` + primary CTA → `APP_HREF`; right = the `PRICING.rows` list (`k` left, `v` right in `font-mono text-accent`; `muted` rows in `text-ink-3`). Numbers come from `content.ts` (which imports `credits.ts`).
  - **Faq** (`id="faq"`): eyebrow "FAQ", heading "Questions worth asking.", then map `FAQ` to `qa` blocks (question `font-semibold`, answer `text-ink-2`). Static open blocks (no JS accordion needed for v1).
  - **ClosingCta:** `CLOSING.h` (`font-display`), `CLOSING.p`, primary CTA → `APP_HREF`; the teal radial glow; on `bg-surface-2`.
  - **MarketingFooter:** brand + the anchor links + "Sign in" + `© ChartSage`.

- [ ] **Step 2: Compose** — add all four to `MarketingLanding` (final order per spec §2). The landing is now complete.

- [ ] **Step 3: Verify + commit**

```bash
npx tsc --noEmit
git add src/app/components/marketing/
git commit -m "feat(marketing): pricing, FAQ, closing CTA, footer"
```

---

### Task 6: SEO metadata + CTA analytics + redirect audit

**Files:**
- Modify: `src/app/page.tsx` (metadata)
- Modify: the marketing CTA components (analytics)
- Modify: `src/app/anon-limit/page.tsx`, `src/app/reports/page.tsx` (+ verify `welcome`, `login`)

- [ ] **Step 1: SEO metadata** — in `src/app/page.tsx`, export `metadata` (it's a server component, so this is allowed):

```tsx
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'ChartSage — Ten charts and a written summary, from one spreadsheet',
  description: 'Upload a CSV or Excel file. ChartSage picks the ten charts that matter and writes the analysis to match — no SQL, no pivot tables, no BI tool.',
  openGraph: {
    title: 'ChartSage — a report from your spreadsheet, in seconds',
    description: 'Upload a CSV. Get ten charts and a written summary. First report free.',
    type: 'website',
    images: ['/icon.svg'],   // TODO-able later: a dedicated OG image; icon.svg is a valid placeholder
  },
  twitter: { card: 'summary_large_image' },
};
```

- [ ] **Step 2: CTA analytics** — in each component with a primary "Upload a CSV" CTA (Nav, Hero, Pricing, ClosingCta), fire PostHog on click. Import the existing client: `import { posthog } from '../../lib/posthog';` and add `onClick={() => posthog.capture?.('marketing_cta_clicked', { location: 'hero' })}` (location: `nav` | `hero` | `pricing` | `closing`). The CTA is a link to `/app`; the capture is fire-and-forget (don't block navigation).

- [ ] **Step 3: Redirect/CTA audit** — repoint anything that meant "the upload tool" (was `/`) to `/app`:
  - `src/app/anon-limit/page.tsx`: the `← Back to home` link (`href="/"`) — leave as `/` (marketing home is the correct "home"); but if there's a "try again / upload" affordance, point it to `/app`. Confirm by reading the file.
  - `src/app/reports/page.tsx`: any "New report" / "Upload" / empty-state CTA that links to `/` → change to `/app`. (The My Reports empty state should send users to the tool, not the marketing page.) Confirm by reading the file.
  - Verify `welcome/page.tsx` (safeNext default `/reports` — fine) and `login/page.tsx` (next default `/reports`, "Back to home" `/` — fine) need no change.
  - AppHeader brand link stays `/` (redirects logged-in → `/app`; shows marketing to logged-out — correct).

- [ ] **Step 4: Verify + commit**

```bash
npx tsc --noEmit
git add src/app/page.tsx src/app/components/marketing/ src/app/reports/page.tsx src/app/anon-limit/page.tsx
git commit -m "feat(marketing): SEO/OG metadata, CTA analytics, redirect audit to /app"
```

---

### Task 7: Build + visual QA + finish

**Files:** none (verification only)

- [ ] **Step 1: Clean build** — `rm -rf .next && npm run build`. Expected: exit 0; routes include `/` (dynamic — it reads cookies) and `/app`.
- [ ] **Step 2: Visual QA** (via `npm run dev` / preview): logged-out `/` renders the full landing (all 11 sections, dark, light example report with real charts); anchor nav scrolls; every "Upload a CSV" CTA → `/app`; `/app` is the working uploader; (if testable) logged-in `/` redirects to `/app`. Confirm fonts/tokens render and there are no console errors.
- [ ] **Step 3: Behavior check** — `/app` uploads + generates a report (anonymous free path) exactly as the old `/` did; My Reports empty-state CTA lands on `/app`.
- [ ] **Step 4: Finish** — use **superpowers:finishing-a-development-branch** to merge `marketing-site` → `main` (Vercel auto-deploys the frontend). Frontend-only; no backend deploy. Production merge requires explicit user authorization.

---

## Self-Review

**Spec coverage:** §1 routing (conditional `/`, `/app` move, AppHeader hide, CTA audit) → T1 + T6; §2 all 11 sections + verbatim copy → T2 (nav/hero/trust), T3 (how/features/uses), T4 (example), T5 (pricing/faq/closing/footer), with copy in `content.ts`; §3 example embed (real renderer, fixture, read-only, no dnd) → T4 (extract `ChartContent`, synthetic fixture, `ExampleReport`); §4 design system + componentization → all (semantic classes, `components/marketing/`, numbers from `credits.ts`); §5 SEO + analytics → T6; §6 non-goals respected (no backend, single page, no payments/deletion); §7 verification → T7. No gaps.

**Placeholder scan:** Complete code for the routing keystone (T1), the extracted `ChartContent` (T4), `content.ts` (T2), and the conditional page. The `c3` pie chart in the fixture is intentionally left for the implementer to match against `PieChart.tsx` (hand-authoring a chart spec requires reading the one component — line/bar shapes are given complete). Section components (T2/T3/T5) are "port this section of `design/landing.html` to semantic classes + `content.ts`" — concrete because the exact markup/styling/copy already exist in the committed mockup. OG image is a flagged placeholder (`icon.svg`), not a gap.

**Type consistency:** `Report`/`ChartWithCaption`/`ChartLayoutEntry` from `useReportLayout` used in the fixture; `ChartContent({spec})` signature matches its use in `ChartCard` and `ExampleReport`; `content.ts` exports (`HERO`/`TRUST`/`HOW`/`FEATURES`/`USES`/`PRICING`/`FAQ`/`CLOSING`/`APP_HREF`) referenced consistently; credit constants imported from `credits.ts` (`REPORT_COST`/`GENERATE_MORE_COST`/`SIGNUP_GRANT`) — same names as the live module. `getSupabaseServer()` defined in T1, used by `/`.

**Risk note:** `ChartCard` is dnd-coupled (`useSortable`), so the example embed deliberately reuses the extracted `ChartContent` (the chart components) rather than `ChartCard` — avoids needing a DndContext on the marketing page.
