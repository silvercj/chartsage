# ChartSage Visual Redesign ("The Analyst's Instrument") — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Subagents must NOT run `git checkout`, `git switch`, `git reset`, or `git stash`** (a subagent moved HEAD during SP2). Stay on the current branch; commit only the files listed per task.

**Goal:** Restyle the entire ChartSage frontend to the approved "Analyst's Instrument" aesthetic — a dark editorial app shell + light report surface, Fraunces/Geist/Geist-Mono type, teal+ember palette — with zero behavior changes.

**Architecture:** A CSS-variable token layer (dark on `:root`, light on `.theme-light`) surfaced as semantic Tailwind colors (`bg-canvas`, `text-ink`, `bg-accent`…) so the same utilities flip by scope. Fonts via the `geist` package + `next/font` Fraunces. Charts share one ECharts theme module. No component logic, props, routing, auth, or data flow changes.

**Tech Stack:** Next.js 14 App Router, Tailwind 3.4, TypeScript, ECharts 5.5 (`echarts-for-react`), `geist` font package.

**Spec:** `docs/superpowers/specs/2026-05-31-redesign-instrument-design.md`
**Branch:** `redesign-instrument` (already checked out; carries the spec commits).
**Verification model:** This is visual/frontend work with no unit tests for styling. Each task: implement → `npx tsc --noEmit` (clean) → commit. A final task runs `npm run build` + a visual QA pass via the local preview.

---

## File Structure

**New files**
- `src/app/lib/fonts.ts` — central font loader (Fraunces + Geist + Geist Mono), exported for layout + chart theme.
- `src/app/report/[id]/charts/chartTheme.ts` — shared ECharts base option + palette + axis/tooltip helpers + runtime mono-family resolver.

**Modified — foundation**
- `tailwind.config.js` — semantic color tokens + font families + shadows. Remove unused `primary` scale.
- `src/app/globals.css` — dark `:root` tokens, `.theme-light` tokens, grain overlay, `@layer components` (eyebrow/btn/card/pill), reduced-motion.
- `src/app/layout.tsx` — font variables on `<html>`; dark `<main>`; top accent line.
- `package.json` / lockfile — add `geist`.

**Modified — dark shell surfaces**
- `src/app/components/AppHeader.tsx`, `src/app/components/CreditsBadge.tsx`
- `src/app/page.tsx` (home/upload)
- `src/app/login/page.tsx`, `src/app/welcome/page.tsx`, `src/app/anon-limit/page.tsx`
- `src/app/credits/page.tsx`, `src/app/components/OutOfCreditsModal.tsx`, `src/app/components/UpsellModal.tsx`
- `src/app/reports/page.tsx`

**Modified — light report surfaces** (wrapped in `.theme-light`)
- `src/app/report/[id]/page.tsx`, `ReportSummary.tsx`, `ChartCard.tsx`, `DataQualityCallout.tsx`, `Sidebar.tsx`, `SidebarCard.tsx`, `SidebarChartThumbnail.tsx`, `Toolbar.tsx`
- `src/app/report/[id]/charts/*` (BarChart, BoxPlot, Heatmap, HistogramChart, LineChart, PieChart, ScatterChart)
- `src/app/report/[id]/print/page.tsx`

**Token / class contract** (used by every surface task — defined in Task 1):

| Need | Tailwind class |
|---|---|
| Page background | `bg-canvas` |
| Card | `card` (= `bg-surface border border-line rounded-2xl`) + `shadow-card` |
| Inset/section bg | `bg-surface-2` |
| Primary text | `text-ink` · secondary `text-ink-2` · muted/labels `text-ink-3` |
| Hairline border | `border-line` · stronger `border-line-2` |
| Primary button | `btn btn-primary` (teal `bg-accent` + `text-on-accent`, `shadow-glow`) |
| Secondary button | `btn btn-ghost` |
| Eyebrow label | `eyebrow` (mono, uppercase, tracking, `text-accent`) |
| Credits pill | `pill` |
| Display text | `font-display` (Fraunces) · UI `font-sans` (Geist) · data/numerals `font-mono` (Geist Mono) |
| Accent | `text-accent` / `bg-accent` · warm accent `text-ember` / `bg-ember` |

---

### Task 1: Design-token foundation (fonts, tokens, layout)

The keystone. Establishes the token system + fonts everything else uses.

**Files:**
- Modify: `package.json` (add `geist`)
- Create: `src/app/lib/fonts.ts`
- Modify: `tailwind.config.js`
- Modify: `src/app/globals.css`
- Modify: `src/app/layout.tsx`

- [ ] **Step 1: Install the geist font package**

```bash
npm install geist
```
Expected: `geist` added to `package.json` dependencies; lockfile updated.

- [ ] **Step 2: Create the font loader** `src/app/lib/fonts.ts`

```ts
import { GeistSans } from 'geist/font/sans';
import { GeistMono } from 'geist/font/mono';
import { Fraunces } from 'next/font/google';

// Display serif — variable, optical sizing + italic.
export const fraunces = Fraunces({
  subsets: ['latin'],
  axes: ['opsz'],
  style: ['normal', 'italic'],
  variable: '--font-fraunces',
  display: 'swap',
});

// Geist exposes fixed CSS vars: --font-geist-sans / --font-geist-mono
export { GeistSans, GeistMono };
```

- [ ] **Step 3: Replace `tailwind.config.js`** with the token mapping (removes the unused `primary` scale)

```js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        canvas: 'rgb(var(--canvas) / <alpha-value>)',
        surface: {
          DEFAULT: 'rgb(var(--surface) / <alpha-value>)',
          2: 'rgb(var(--surface-2) / <alpha-value>)',
        },
        ink: {
          DEFAULT: 'rgb(var(--ink) / <alpha-value>)',
          2: 'rgb(var(--ink-2) / <alpha-value>)',
          3: 'rgb(var(--ink-3) / <alpha-value>)',
        },
        line: {
          DEFAULT: 'rgb(var(--line) / <alpha-value>)',
          2: 'rgb(var(--line-2) / <alpha-value>)',
        },
        accent: {
          DEFAULT: 'rgb(var(--accent) / <alpha-value>)',
          hover: 'rgb(var(--accent-hover) / <alpha-value>)',
        },
        'on-accent': 'rgb(var(--on-accent) / <alpha-value>)',
        ember: 'rgb(var(--ember) / <alpha-value>)',
      },
      fontFamily: {
        sans: ['var(--font-geist-sans)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-geist-mono)', 'ui-monospace', 'monospace'],
        display: ['var(--font-fraunces)', 'Georgia', 'serif'],
      },
      boxShadow: {
        card: 'var(--shadow-card)',
        'card-lg': 'var(--shadow-card-lg)',
        glow: 'var(--glow)',
      },
    },
  },
  plugins: [],
};
```

- [ ] **Step 4: Replace `src/app/globals.css`** with tokens + base + component classes

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

/* ===== Tokens: dark shell (default) ===== */
:root {
  --canvas: 21 19 15;          /* #15130F */
  --surface: 32 29 23;         /* #201D17 */
  --surface-2: 38 34 24;       /* #262218 */
  --ink: 242 238 228;          /* #F2EEE4 */
  --ink-2: 168 161 146;        /* #A8A192 */
  --ink-3: 110 104 92;         /* #6E685C */
  --line: 49 44 36;            /* #312C24 */
  --line-2: 64 58 47;          /* #403A2F */
  --accent: 45 212 191;        /* #2DD4BF */
  --accent-hover: 94 234 212;  /* #5EEAD4 */
  --on-accent: 4 32 28;        /* #04201C */
  --ember: 232 132 58;         /* #E8843A */
  --shadow-card: 0 2px 4px rgb(0 0 0 / 0.40), 0 18px 40px -16px rgb(0 0 0 / 0.70);
  --shadow-card-lg: 0 2px 4px rgb(0 0 0 / 0.40), 0 44px 90px -34px rgb(0 0 0 / 0.85);
  --glow: 0 0 30px -6px rgb(45 212 191 / 0.50);
  --grain-opacity: 0.045;
}

/* ===== Light report surface ===== */
.theme-light {
  --canvas: 246 243 236;       /* #F6F3EC */
  --surface: 255 255 255;      /* #FFFFFF */
  --surface-2: 239 235 225;    /* #EFEBE1 */
  --ink: 27 26 22;             /* #1B1A16 */
  --ink-2: 92 86 75;           /* #5C564B */
  --ink-3: 154 145 131;        /* #9A9183 */
  --line: 230 224 212;         /* #E6E0D4 */
  --line-2: 216 209 194;       /* #D8D1C2 */
  --accent: 12 92 82;          /* #0C5C52 */
  --accent-hover: 8 70 63;     /* #08463F */
  --on-accent: 255 255 255;    /* #FFFFFF */
  --ember: 188 95 28;          /* #BC5F1C */
  --shadow-card: 0 1px 2px rgb(27 26 22 / 0.04), 0 14px 34px -14px rgb(27 26 22 / 0.18);
  --shadow-card-lg: 0 2px 4px rgb(27 26 22 / 0.04), 0 36px 70px -28px rgb(27 26 22 / 0.28);
  --glow: 0 1px 2px rgb(12 92 82 / 0.40), 0 10px 22px -10px rgb(12 92 82 / 0.55);
  --grain-opacity: 0.035;
}

body {
  background: rgb(var(--canvas));
  color: rgb(var(--ink));
}

/* subtle paper/charcoal grain */
body::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: 9999;
  pointer-events: none;
  opacity: var(--grain-opacity);
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}

@layer components {
  .eyebrow {
    @apply font-mono text-xs uppercase tracking-[0.18em] text-accent;
  }
  .btn {
    @apply inline-flex items-center justify-center gap-2 rounded-xl px-5 py-3 text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed;
  }
  .btn-primary {
    @apply bg-accent text-on-accent shadow-glow hover:bg-accent-hover;
  }
  .btn-ghost {
    @apply border border-line-2 text-ink hover:bg-surface;
  }
  .card {
    @apply bg-surface border border-line rounded-2xl;
  }
  .pill {
    @apply font-mono text-[13px] bg-surface border border-line-2 text-ink rounded-full px-3 py-1.5 inline-flex items-center gap-1.5;
  }
}

/* entrance motion (opt-in via .reveal) */
.reveal { opacity: 0; transform: translateY(14px); animation: rise 0.7s cubic-bezier(0.2, 0.7, 0.2, 1) forwards; }
@keyframes rise { to { opacity: 1; transform: none; } }

@media (prefers-reduced-motion: reduce) {
  .reveal { animation: none; opacity: 1; transform: none; }
}
```

- [ ] **Step 5: Update `src/app/layout.tsx`** — fonts + dark main + top accent line

```tsx
import type { Metadata } from 'next'
import { GeistSans, GeistMono, fraunces } from './lib/fonts'
import PostHogInit from './PostHogInit'
import SessionWatcher from './components/SessionWatcher'
import AppHeader from './components/AppHeader'
import { CreditsProvider } from './lib/useCredits'
import './globals.css'

export const metadata: Metadata = {
  title: 'ChartSage - AI-Powered Data Visualization',
  description: 'Turn any spreadsheet into a beautiful, interactive report with AI-generated insights in seconds.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable} ${fraunces.variable}`}>
      <body className="font-sans antialiased">
        <div className="h-[3px] bg-[linear-gradient(90deg,rgb(var(--accent))_0%,rgb(var(--accent))_62%,rgb(var(--ember))_62%,rgb(var(--ember))_100%)]" />
        <PostHogInit />
        <SessionWatcher />
        <CreditsProvider>
          <AppHeader />
          <main className="min-h-screen bg-canvas text-ink">{children}</main>
        </CreditsProvider>
      </body>
    </html>
  )
}
```

- [ ] **Step 6: Verify types compile**

Run: `npx tsc --noEmit`
Expected: no errors. (App will look unstyled-but-dark until surfaces are updated — that's fine.)

- [ ] **Step 7: Commit**

```bash
git add package.json package-lock.json src/app/lib/fonts.ts tailwind.config.js src/app/globals.css src/app/layout.tsx
git commit -m "feat(redesign): design-token foundation — fonts, dark/light tokens, base classes"
```

---

### Task 2: Shared chart theme (`chartTheme.ts`)

One source of truth for the editorial ECharts look. Charts are client components; ECharts renders to canvas, which **cannot resolve CSS variables** — so the mono font family is resolved at runtime from the computed `--font-geist-mono` value.

**Files:**
- Create: `src/app/report/[id]/charts/chartTheme.ts`

- [ ] **Step 1: Create `chartTheme.ts`**

```ts
// Shared ECharts styling for the light report canvas. Charts keep their own
// data/series logic; this centralizes palette, axes, tooltip, and fonts.

export const CHART_PALETTE = ['#0C5C52', '#5B8C7E', '#C99A3F', '#B5673A', '#3E5C6B', '#9DB7AE'];
export const CHART_TEAL = '#0C5C52';
export const CHART_OCHRE = '#C99A3F';
export const CHART_INK = '#1B1A16';
export const CHART_INK_MUTED = '#9A9183';
export const CHART_LINE = '#E6E0D4';

// ECharts canvas needs a literal font-family string; resolve the Geist Mono
// family that next/font set on --font-geist-mono (falls back gracefully on SSR).
export function monoFamily(): string {
  if (typeof window === 'undefined') return 'ui-monospace, monospace';
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue('--font-geist-mono')
    .trim();
  return v ? `${v}, ui-monospace, monospace` : 'ui-monospace, monospace';
}

// Base option fragment every chart spreads in.
export function chartBase() {
  const mono = monoFamily();
  return {
    color: CHART_PALETTE,
    textStyle: { fontFamily: mono, color: CHART_INK_MUTED },
    grid: { left: 8, right: 18, top: 20, bottom: 8, containLabel: true },
    tooltip: {
      backgroundColor: '#ffffff',
      borderColor: CHART_LINE,
      borderWidth: 1,
      padding: [8, 12],
      textStyle: { color: CHART_INK, fontFamily: mono, fontSize: 12 },
      extraCssText: 'box-shadow:0 8px 24px -8px rgba(27,26,22,.22);border-radius:10px;',
    },
  };
}

// Category axis with stripped chrome + mono labels. `interval`/`rotate` for density.
export function catAxis(extra: Record<string, any> = {}) {
  const { axisLabel, ...rest } = extra;
  return {
    type: 'category',
    axisLine: { show: false },
    axisTick: { show: false },
    splitLine: { show: false },
    axisLabel: { fontFamily: monoFamily(), fontSize: 11, color: CHART_INK_MUTED, ...(axisLabel || {}) },
    ...rest,
  };
}

// Value axis with hairline split-lines + mono labels.
export function valAxis(extra: Record<string, any> = {}) {
  const { axisLabel, ...rest } = extra;
  return {
    type: 'value',
    axisLine: { show: false },
    axisTick: { show: false },
    splitLine: { lineStyle: { color: CHART_LINE, width: 1 } },
    axisLabel: { fontFamily: monoFamily(), fontSize: 11, color: CHART_INK_MUTED, ...(axisLabel || {}) },
    ...rest,
  };
}

// Teal vertical area gradient (object form — no echarts import needed).
export function tealAreaGradient() {
  return {
    type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
    colorStops: [
      { offset: 0, color: 'rgba(12,92,82,0.22)' },
      { offset: 1, color: 'rgba(12,92,82,0)' },
    ],
  };
}
```

- [ ] **Step 2: Verify types compile**

Run: `npx tsc --noEmit`
Expected: no errors (module is not imported yet; this just confirms it's valid TS).

- [ ] **Step 3: Commit**

```bash
git add src/app/report/[id]/charts/chartTheme.ts
git commit -m "feat(redesign): shared editorial ECharts theme (palette, axes, tooltip, mono resolver)"
```

---

### Task 3: App header + credits badge (dark shell)

**Files:**
- Modify: `src/app/components/AppHeader.tsx`
- Modify: `src/app/components/CreditsBadge.tsx`

Keep all logic (the `/report/*/print` hide rule, `useCredits()`, `signOut`, links, email). Restyle only.

- [ ] **Step 1: Restyle `AppHeader.tsx`**

Recipe (apply, preserving structure/handlers):
- Root `<header>`: `border-b border-line bg-canvas/80 backdrop-blur` (remove white/stone classes).
- Brand: keep the inline SVG bar mark inside `inline-flex w-7 h-7 items-center justify-center rounded-lg bg-surface-2 border border-line-2`; wordmark `font-display text-lg font-medium text-ink`.
- Nav links: `text-sm text-ink-2 hover:text-ink transition-colors`.
- Email span: `font-mono text-xs text-ink-3 max-w-[160px] truncate`.
- Sign out button: `text-sm text-ink-2 hover:text-ink`.
- Signed-out "Sign in" link: `btn btn-primary !px-4 !py-2` (or `bg-accent text-on-accent rounded-lg px-4 py-2 text-sm`).
- `<CreditsBadge />` stays.

- [ ] **Step 2: Restyle `CreditsBadge.tsx`**

Recipe: render the `pill` class; `⚡` icon `text-ember`; the number in the pill's mono font. When low (`balance < REPORT_COST`), add `!border-ember/50 !text-ember`. Keep the `<a href="/credits">` wrapper and all balance logic.

- [ ] **Step 3: Verify + commit**

```bash
npx tsc --noEmit
git add src/app/components/AppHeader.tsx src/app/components/CreditsBadge.tsx
git commit -m "feat(redesign): dark app header + credits badge"
```

---

### Task 4: Home / upload page (dark)

**Files:**
- Modify: `src/app/page.tsx`

This is the upload tool restyled with a hero header — NOT a full marketing landing. Preserve ALL logic: dropzone, upload, generate call, the multi-step progress, the cost label on the button (`balance !== null ? 'Generate report · 100' : …`), the 402 OUT_OF_CREDITS → OutOfCreditsModal path.

- [ ] **Step 1: Restyle `page.tsx`**

Recipe:
- Page wrapper: `bg-canvas` (inherited) with generous padding; center column `max-w-3xl mx-auto px-6 py-16`.
- Hero header above the uploader: an `eyebrow` ("CSV → INSIGHT IN ~30 SECONDS"), an `<h1 className="font-display text-4xl sm:text-5xl font-normal tracking-tight text-ink leading-[1.05]">` with a one-word italic teal accent (e.g. `Read the <em className="italic font-medium text-accent">story</em> inside it.`), a `text-ink-2 mt-4 max-w-prose` sub.
- Dropzone: `card` + `border-dashed border-line-2 hover:border-accent` `rounded-2xl p-10 text-center cursor-pointer transition-colors`; icon tile `bg-surface-2`; helper text `text-ink-3 font-mono text-xs`.
- Generate button: `btn btn-primary w-full` (keep the cost-label text + disabled/generating logic).
- Progress steps: restyle to dark — active step `text-accent`, done `text-ink-2`, pending `text-ink-3`; progress bar track `bg-surface-2`, fill `bg-accent`.
- Any error text: `text-ember`.
- Wrap top-level children in `.reveal` for the entrance (optional; skip if it complicates state).

- [ ] **Step 2: Verify + commit**

```bash
npx tsc --noEmit
git add src/app/page.tsx
git commit -m "feat(redesign): dark home/upload page with editorial hero"
```

---

### Task 5: Auth surfaces — login, welcome, anon-limit (dark)

**Files:**
- Modify: `src/app/login/page.tsx`
- Modify: `src/app/welcome/page.tsx`
- Modify: `src/app/anon-limit/page.tsx`

Preserve ALL logic + existing credits copy ("300 free credits", "Sign in or sign up", safeNext, etc.).

- [ ] **Step 1: Restyle the three pages with a shared recipe**

- Page: `min-h-screen bg-canvas flex items-center justify-center px-6`.
- Card: `card shadow-card-lg rounded-2xl p-8 max-w-md w-full`.
- Heading: `font-display text-2xl font-medium text-ink`.
- Sub/body: `text-ink-2 text-sm`.
- Primary action (e.g. Google / "Sign up free"): `btn btn-primary w-full`.
- Magic-link email input: `w-full bg-surface-2 border border-line-2 rounded-lg px-4 py-3 text-ink placeholder:text-ink-3 focus:border-accent outline-none` (keep its `aria-label`).
- Secondary link: `text-ink-2 hover:text-ink`.
- `welcome`: keep the icon tiles, restyle to `bg-surface-2 text-accent`; eyebrow via `eyebrow`.
- `anon-limit`: keep CTA "Sign up free" → `btn btn-primary`.

- [ ] **Step 2: Verify + commit**

```bash
npx tsc --noEmit
git add src/app/login/page.tsx src/app/welcome/page.tsx src/app/anon-limit/page.tsx
git commit -m "feat(redesign): dark login, onboarding, and anon-limit screens"
```

---

### Task 6: Credits page + modals (dark)

**Files:**
- Modify: `src/app/credits/page.tsx`
- Modify: `src/app/components/OutOfCreditsModal.tsx`
- Modify: `src/app/components/UpsellModal.tsx`

Preserve ALL logic: the `useCredits()` gating (`authLoading`/`session`/redirect), history fetch + try/catch, the notify button → `OutOfCreditsModal`, the `upgrade_intent_clicked` + POST `/upgrade-intent` behavior.

- [ ] **Step 1: Restyle `credits/page.tsx`**

- Page: `bg-canvas`; column `max-w-2xl mx-auto px-6 py-12`.
- Title: `font-display text-3xl font-medium text-ink`; sub `text-ink-2`.
- Balance card: `card shadow-card p-6 rounded-2xl`; eyebrow "BALANCE"; the number `font-mono text-4xl font-semibold text-ink` (keep `{balance ?? '—'}`); cost rows `text-ink-2` with mono numbers.
- Notify button: `btn btn-ghost`.
- History: section label `eyebrow`; list `card divide-y divide-line`; each row `px-5 py-3 flex justify-between text-sm`; grant deltas `text-accent`, debits `text-ink-2`; labels `text-ink`.
- Loading/empty text: `text-ink-3`.

- [ ] **Step 2: Restyle the two modals**

Recipe (both): overlay `fixed inset-0 bg-black/60 backdrop-blur-sm`; dialog `card shadow-card-lg rounded-2xl p-6 max-w-md`; title `font-display text-xl text-ink`; body `text-ink-2`; primary `btn btn-primary`; dismiss `btn btn-ghost` or `text-ink-2`. `OutOfCreditsModal`: the out-of-credits accent uses `text-ember`. Keep all handlers/props.

- [ ] **Step 3: Verify + commit**

```bash
npx tsc --noEmit
git add src/app/credits/page.tsx src/app/components/OutOfCreditsModal.tsx src/app/components/UpsellModal.tsx
git commit -m "feat(redesign): dark credits page + out-of-credits/upsell modals"
```

---

### Task 7: My Reports list (dark)

**Files:**
- Modify: `src/app/reports/page.tsx`

Preserve the `useCredits()` gating + fetch logic + empty state + report links.

- [ ] **Step 1: Restyle**

- Page: `bg-canvas`; column `max-w-5xl mx-auto px-6 py-12`.
- Title: `font-display text-3xl font-medium text-ink`; sub `text-ink-2`.
- Report cards/list: `card shadow-card hover:border-line-2 transition-colors rounded-2xl p-5`; report title `font-display text-lg text-ink`; meta (date/charts) `font-mono text-xs text-ink-3`.
- Empty state: `text-ink-3`, with a `btn btn-primary` linking to `/`.
- Loading: `text-ink-3`.

- [ ] **Step 2: Verify + commit**

```bash
npx tsc --noEmit
git add src/app/reports/page.tsx
git commit -m "feat(redesign): dark My Reports list"
```

---

### Task 8: Report surface shell (light) — page, summary, cards, sidebar

**Files:**
- Modify: `src/app/report/[id]/page.tsx`
- Modify: `src/app/report/[id]/ReportSummary.tsx`
- Modify: `src/app/report/[id]/ChartCard.tsx`
- Modify: `src/app/report/[id]/DataQualityCallout.tsx`
- Modify: `src/app/report/[id]/Sidebar.tsx`, `SidebarCard.tsx`, `SidebarChartThumbnail.tsx`

The report reads as a light "document" inside the dark app. Preserve ALL layout/data/dnd-kit/state logic.

- [ ] **Step 1: Scope the report to light** in `page.tsx`

Wrap the report's top-level container so its content uses the light token set:
```tsx
// the outermost element this page renders:
<div className="theme-light bg-canvas text-ink min-h-screen">
  {/* existing report layout unchanged */}
</div>
```
This makes every `bg-surface`/`text-ink`/`border-line`/`bg-accent` inside resolve to the light palette automatically.

- [ ] **Step 2: Restyle the document pieces** (recipes; keep all logic)

- `ReportSummary.tsx`: report title `font-display text-3xl font-medium text-ink`; meta line `font-mono text-xs uppercase tracking-wide text-ink-3` ("generated · {time} · {n} charts"); summary body `text-ink-2 leading-relaxed`.
- `ChartCard.tsx`: `card shadow-card rounded-2xl p-5`; chart title `font-display text-lg text-ink`; caption `font-display italic text-ink-2`; any intent/label `font-mono text-xs text-ink-3`.
- `DataQualityCallout.tsx`: `bg-surface-2 border border-line rounded-xl p-4`; heading `font-mono text-xs uppercase tracking-wide text-ink-3`; items `text-ink-2`; use `text-ember` for warnings.
- `Sidebar.tsx` / `SidebarCard.tsx` / `SidebarChartThumbnail.tsx`: light surfaces `bg-surface border-line`; active item `border-accent text-ink`; labels `font-mono text-xs text-ink-3`. Keep the dnd-kit drag handlers/refs intact.

- [ ] **Step 3: Verify + commit**

```bash
npx tsc --noEmit
git add "src/app/report/[id]/page.tsx" "src/app/report/[id]/ReportSummary.tsx" "src/app/report/[id]/ChartCard.tsx" "src/app/report/[id]/DataQualityCallout.tsx" "src/app/report/[id]/Sidebar.tsx" "src/app/report/[id]/SidebarCard.tsx" "src/app/report/[id]/SidebarChartThumbnail.tsx"
git commit -m "feat(redesign): light report document surface (summary, cards, sidebar)"
```

---

### Task 9: Report toolbar (light, on the document)

**Files:**
- Modify: `src/app/report/[id]/Toolbar.tsx`

Preserve ALL behavior: generate-more (cost label `Generate 5 more · ${GENERATE_MORE_COST}` + the inline spinner when `generating`), export PDF, the 503/402/OUT_OF_CREDITS/UPGRADE_REQUIRED branches, `useCredits().refetch`, the two modals.

- [ ] **Step 1: Restyle**

- Sticky bar: `sticky top-0 z-10 bg-canvas/90 backdrop-blur border-b border-line` (light tokens via the parent `.theme-light`).
- Generate-more button: `btn btn-ghost` (keep spinner markup; the spinner ring `border-line-2 border-t-accent`).
- Export PDF button: `btn btn-primary`.
- Error text: `text-ember`.

- [ ] **Step 2: Verify + commit**

```bash
npx tsc --noEmit
git add "src/app/report/[id]/Toolbar.tsx"
git commit -m "feat(redesign): light report toolbar"
```

---

### Task 10: Chart components — adopt the shared theme (light)

**Files:**
- Modify: `src/app/report/[id]/charts/BarChart.tsx`, `BoxPlot.tsx`, `Heatmap.tsx`, `HistogramChart.tsx`, `LineChart.tsx`, `PieChart.tsx`, `ScatterChart.tsx`

Each chart KEEPS its data/series/guard logic. Replace ad-hoc styling (the rainbow `COLORS`, hard-coded `TEXT_COLOR`/`AXIS_COLOR`, per-chart axis blocks) with `chartTheme` helpers. Preserve the Heatmap dense-grid fix and every existing guard (scatter sampling, line rolling-avg, etc.).

- [ ] **Step 1: Pattern — refactor `LineChart.tsx` to the theme** (reference example)

Replace the module color/axis constants and the inline `textStyle`/`tooltip`/`grid`/`xAxis`/`yAxis` with the shared helpers; keep `rollingAvg`, `hasSeries`, `showSmoothed`, `fmtY`, legend logic:

```tsx
'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';
import { chartBase, catAxis, valAxis, CHART_PALETTE, CHART_INK, tealAreaGradient } from './chartTheme';

// ...rollingAvg unchanged...

export default function LineChart({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const hasSeries = spec.series && Array.isArray(spec.series);
  const xData = hasSeries ? spec.series[0].x : spec.x;
  const xLen = xData?.length ?? 0;
  const showSmoothed = xLen >= 12 && !hasSeries;

  const baseSeries = hasSeries
    ? spec.series.map((s: any, i: number) => ({
        name: s.name, type: 'line', data: s.y, smooth: true, symbol: 'circle', symbolSize: 6,
        showSymbol: false, lineStyle: { width: 2.5 }, itemStyle: { color: CHART_PALETTE[i % CHART_PALETTE.length] },
        emphasis: { showSymbol: true },
      }))
    : [{
        name: spec.y_label || 'value', type: 'line', data: spec.y, smooth: true, symbol: 'circle',
        symbolSize: 6, showSymbol: false, lineStyle: { width: 2.5, color: CHART_PALETTE[0] },
        itemStyle: { color: CHART_PALETTE[0] }, emphasis: { showSymbol: true },
        areaStyle: { color: tealAreaGradient() },
      }];

  if (showSmoothed) {
    const smoothed = rollingAvg(spec.y as number[], 3);
    baseSeries.push({
      name: '3-mo avg', type: 'line', data: smoothed, smooth: true, symbol: 'none',
      lineStyle: { width: 2, color: CHART_INK, type: 'dashed', opacity: 0.5 }, itemStyle: { color: CHART_INK },
    } as any);
  }

  const showLegend = hasSeries || showSmoothed;

  return (
    <ReactECharts
      option={{
        ...chartBase(),
        grid: { left: 8, right: 18, top: 24, bottom: showLegend ? 40 : 8, containLabel: true },
        tooltip: {
          ...chartBase().tooltip, trigger: 'axis',
          formatter: (p: any[]) =>
            p.filter((x) => x.value !== null && x.value !== undefined)
             .map((x) => `<span style="color:${x.color}">●</span> ${x.seriesName ?? ''}: ${fmtY(x.value)}`)
             .join('<br/>'),
        },
        legend: showLegend
          ? { bottom: 0, textStyle: { fontFamily: chartBase().textStyle.fontFamily, fontSize: 11, color: CHART_INK }, icon: 'roundRect', itemWidth: 10, itemHeight: 10, itemGap: 20 }
          : undefined,
        xAxis: catAxis({ data: xData, name: spec.x_label, nameLocation: 'middle', nameGap: 34,
          nameTextStyle: { color: '#9A9183', fontSize: 11 },
          axisLabel: { interval: xLen > 24 ? Math.floor(xLen / 12) : 'auto', rotate: xLen > 8 ? 30 : 0 } }),
        yAxis: valAxis({ name: spec.y_label, nameLocation: 'middle', nameGap: 56,
          nameTextStyle: { color: '#9A9183', fontSize: 11 }, axisLabel: { formatter: fmtY } }),
        series: baseSeries,
      }}
      style={{ width: '100%', height: 320 }}
    />
  );
}
```

- [ ] **Step 2: Apply the same conversion to the other six charts**

For each, import from `./chartTheme` and replace styling only:
- **BarChart / HistogramChart:** `series` `itemStyle: { color: CHART_TEAL, borderRadius: [5,5,0,0] }`, `barWidth: '52%'`, `emphasis.itemStyle.color: '#0A4A42'`; axes via `catAxis`/`valAxis`; `...chartBase()`. Rotate/`interval` x labels when many categories.
- **PieChart:** `color: CHART_PALETTE`; donut (`radius: ['45%','70%']`); `itemStyle: { borderColor: '#fff', borderWidth: 2 }`; labels `fontFamily: monoFamily()`, `color: CHART_INK`; legend mono.
- **ScatterChart:** point `itemStyle: { color: CHART_TEAL, opacity: 0.55 }`; keep sampling guard; `catAxis`/`valAxis`; `...chartBase()`.
- **BoxPlot:** box `itemStyle: { color: 'rgba(12,92,82,0.12)', borderColor: CHART_TEAL, borderWidth: 1.5 }`; `catAxis`/`valAxis`.
- **Heatmap:** keep the existing `dense`/`interval:0` fix; set `visualMap.inRange.color` diverging `['#B5673A','#F6F3EC','#0C5C52']` (symmetric) or `['#F6F3EC','#0C5C52']` (sequential); `axisLabel.fontFamily: monoFamily()`; tooltip via `chartBase().tooltip` shape; cell `itemStyle.borderColor: '#fff'`.

Each chart must still render with messy data: keep existing interval/sampling guards and add `rotate`/`interval:0` on category axes with many/long labels.

- [ ] **Step 3: Verify + commit**

```bash
npx tsc --noEmit
git add "src/app/report/[id]/charts/"
git commit -m "feat(redesign): all charts adopt shared editorial ECharts theme"
```

---

### Task 11: PDF print route (light + brand fonts)

**Files:**
- Modify: `src/app/report/[id]/print/page.tsx`

The header is already hidden here. Ensure the print document uses the light tokens + brand fonts; otherwise keep layout unchanged. (Fonts load globally via `layout.tsx`, so they're available; this just confirms the light scope + display/mono usage.)

- [ ] **Step 1: Restyle**

- Wrap the print content root in `className="theme-light bg-white text-ink"`.
- Report title `font-display`; captions `font-display italic`; any numerals/meta `font-mono`.
- Keep the print/PDF layout, page breaks, and chart rendering exactly as-is.

- [ ] **Step 2: Verify + commit**

```bash
npx tsc --noEmit
git add "src/app/report/[id]/print/page.tsx"
git commit -m "feat(redesign): light, on-brand PDF print route"
```

---

### Task 12: Build + visual QA + finish

**Files:** none (verification only)

- [ ] **Step 1: Clean production build**

```bash
rm -rf .next && npm run build
```
Expected: exit 0, no type/lint errors. (If `.next/types` staleness errors appear, the `rm -rf .next` already handles it.)

- [ ] **Step 2: Visual QA pass**

Run the dev server and check each surface in the browser (or the existing preview harness on `design-concept` is for mockups only — use `npm run dev`):
- Dark shell: home `/`, `/login`, `/welcome`, `/anon-limit`, `/credits`, `/reports`, header, both modals.
- Light report: open a report — summary, chart cards, sidebar, toolbar, all chart types.
- Confirm: fonts load (Fraunces headlines, Geist body, mono numerals), no layout shift, no console errors, the report is light within the dark app.

- [ ] **Step 3: Behavior regression spot-check**

Confirm unchanged: login (magic-link/Google), generate report (+ cost label + 402 modal), generate-more (+ spinner), credits history, sign out. (No code changed here — this is a sanity check that restyling didn't break handlers.)

- [ ] **Step 4: Finish the branch**

Use **superpowers:finishing-a-development-branch** to merge `redesign-instrument` → `main` (project's direct-to-main flow; Vercel auto-deploys the frontend). No backend deploy needed (frontend-only).

---

## Self-Review

**Spec coverage:** §1 tokens → T1; §1 charts/chartTheme → T2+T10; §2 typography → T1 (fonts.ts, tailwind, layout); §3 theming architecture (dark default + `.theme-light` scope) → T1 + T8/T11; §4 per-surface — header/badge T3, home T4, auth trio T5, credits+modals T6, reports T7, report doc T8, toolbar T9, charts T10, print T11; §5 scope (visual-only, no marketing site, charts light) → respected throughout; §6 verification → T12. No gaps.

**Placeholder scan:** Foundation (T1) and chartTheme (T2) have complete code. Surface tasks (T3–T9, T11) are restyle recipes with exact class strings + token references — appropriate for applying a defined design system to existing components (the implementer reads the current file and applies the recipe), not vague "make it nice." T10 gives a complete reference refactor (LineChart) + precise per-chart deltas. No TBD/TODO.

**Type/name consistency:** Tailwind tokens (`canvas`, `surface`/`surface-2`, `ink`/`ink-2`/`ink-3`, `line`/`line-2`, `accent`/`accent-hover`, `on-accent`, `ember`) and component classes (`eyebrow`, `btn`/`btn-primary`/`btn-ghost`, `card`, `pill`) are defined in T1 and referenced consistently after. Font CSS vars (`--font-geist-sans`, `--font-geist-mono`, `--font-fraunces`) match `fonts.ts` ↔ `tailwind.config.js` ↔ `chartTheme.monoFamily()`. `chartBase`/`catAxis`/`valAxis`/`tealAreaGradient`/`monoFamily`/`CHART_*` defined in T2, used in T10.

**Risk note:** The one non-obvious technical point — ECharts canvas cannot read CSS variables — is handled by `monoFamily()` resolving `--font-geist-mono` at runtime (T2), used by all charts (T10).
