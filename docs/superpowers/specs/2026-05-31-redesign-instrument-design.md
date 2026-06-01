# ChartSage Visual Redesign — "The Analyst's Instrument"

**Status:** Approved direction (via light + dark mockups), pending spec review
**Date:** 2026-05-31
**Type:** Visual redesign sub-project (frontend only)

**Goal:** Replace the current default-Tailwind look (system/Inter fonts, flat white cards, even grays) with a distinctive, premium, modern aesthetic across the whole app — without changing any behavior, data flow, auth, or credit logic.

**Approved direction:** *The Analyst's Instrument* — editorial intelligence. A characterful serif display against a clean grotesk, monospace numerals for all data, warm palette, deep teal + ember accents, hairline rules, subtle grain, editorial rhythm and motion.

**Approved theme:** **Hybrid — dark shell + light reports.** The marketing/landing surface and app chrome are dark (sleek, modern first impression); the report reading surface, its charts, and the PDF stay light (data legibility + on-screen/PDF consistency).

Reference mockups (gitignored, local): `design/concept-instrument.html` (light), `design/concept-dark.html` (dark).

---

## 1. Design tokens

Tokens are CSS variables (RGB triplets for Tailwind `<alpha-value>` support), surfaced as semantic Tailwind colors. The **dark** set is the app default (`:root`); the **light** set is scoped to the report surface + PDF via a `.theme-light` wrapper.

### Semantic tokens

| Token | Dark (shell, default) | Light (report + PDF) | Use |
|---|---|---|---|
| `--canvas` | `#15130F` | `#F6F3EC` | page background |
| `--surface` | `#201D17` | `#FFFFFF` | cards |
| `--surface-2` | `#262218` | `#EFEBE1` | inset / section bg |
| `--ink` | `#F2EEE4` | `#1B1A16` | primary text |
| `--ink-2` | `#A8A192` | `#5C564B` | secondary text |
| `--ink-3` | `#6E685C` | `#9A9183` | muted / labels |
| `--line` | `#312C24` | `#E6E0D4` | hairline borders |
| `--line-2` | `#403A2F` | `#D8D1C2` | stronger borders |
| `--accent` | `#2DD4BF` | `#0C5C52` | teal primary |
| `--accent-hover` | `#5EEAD4` | `#08463F` | teal hover |
| `--on-accent` | `#04201C` | `#FFFFFF` | text on teal button |
| `--ember` | `#E8843A` | `#BC5F1C` | warm accent (credits ⚡, out-of-credits, top line) |
| `--ember-tint` | `rgba(232,132,58,.12)` | `#F9ECDD` | ember chip bg |

**Shadows / glow:**
- Light: warm soft shadows — `0 1px 2px rgba(27,26,22,.04), 0 14px 34px -14px rgba(27,26,22,.18)`; large `…-28px rgba(27,26,22,.28)`.
- Dark: deep shadows + teal glow on primary — `--glow: 0 0 30px -6px rgba(45,212,191,.5)`; card top-edge highlight `linear-gradient(90deg,transparent,rgba(255,255,255,.08),transparent)`.

### Charts (light report canvas only) — full editorial treatment
Validated in real ECharts (see `design/charts.html`); this is **not** a library limit. All charts inherit a shared theme (`charts/chartTheme.ts`):
- **Palette:** curated editorial sequence — `#0C5C52` teal · `#5B8C7E` sage · `#C99A3F` ochre · `#B5673A` clay · `#3E5C6B` slate · `#9DB7AE` soft-green.
- **Axes:** stripped chrome — `axisLine` & `axisTick` hidden; hairline `splitLine` (`#E6E0D4`); labels in Geist Mono 11px `#9A9183`.
- **Tooltip:** white, hairline border, soft shadow, rounded; values in Geist Mono.
- **Lines:** `smooth: true`, 2.5px, soft markers (`showSymbol` on emphasis), teal→transparent `areaStyle` gradient where an area read suits.
- **Bars:** rounded tops (`borderRadius`), ~50% width, teal with darker emphasis.
- **Heatmap:** keeps teal `visualMap` + the dense-grid fix; diverging scales clay↔paper↔teal.
- **Robustness (real-world data):** label rotation/truncation for long or many categories, sensible `grid` padding, graceful empty/﻿single-value states — the theme must hold up on messy CSVs, not just clean demos.

### Texture & decoration
- **Grain:** fixed full-screen SVG `feTurbulence` overlay, opacity ~.035 (light) / ~.045 (dark), `pointer-events:none`.
- **Top accent line:** 3px bar, `teal 0–62% / ember 62–100%`, at the very top of the app shell.
- **Hero grid:** faint 64px grid, radial-masked, behind hero copy. Dark adds a soft teal radial glow near the report card.

---

## 2. Typography

Self-hosted (no FOUT), exposed as CSS vars on `<html>`:

- `--font-display`: **Fraunces** — via `next/font/google`; weights 400/500/600 + italic; `font-optical-sizing: auto`. Headlines, card/report/section titles, the wordmark, and italic captions.
- `--font-sans`: **Geist** — via the official `geist` npm package (`geist/font/sans`; Vercel's zero-config self-hosted loader — more reliable than Google Fonts for Geist). All UI: body, labels, buttons, nav. (Replaces Inter.)
- `--font-mono`: **Geist Mono** — via `geist/font/mono`. All numerals & data: credit balances, the ⚡ pill, stat values, deltas, chart axis labels, eyebrows, timestamps, meta.

(Adds one dependency: `geist`. Fraunces stays on `next/font/google`.)

**Type scale:**
| Role | Font | Size / weight | Tracking |
|---|---|---|---|
| Hero H1 | Fraunces | `clamp(2.9rem,5.4vw,4.6rem)` / 400 | -0.025em, lh 1.02 |
| Section / report title | Fraunces | 30px / 500 | -0.02em |
| Card title | Fraunces | 20px / 500 | -0.01em |
| Body | Geist | 15–18px / 350–400 | — |
| Caption | Fraunces italic | 15px / 400 | — |
| Eyebrow / label | Geist Mono | 11–12px / 500, UPPERCASE | 0.14–0.18em |
| Data / numerals | Geist Mono | contextual / 500–600 | -0.02em on big numbers |

Headline accent device: a single italic word in `--accent` (e.g. *"story"*, *"distilled"*) — used sparingly.

---

## 3. Theming architecture

- `globals.css` declares the **dark** token set on `:root` and the **light** set on `.theme-light`.
- `tailwind.config.js` maps semantic names → `rgb(var(--token) / <alpha-value>)` (e.g. `bg-canvas`, `text-ink`, `border-line`, `bg-accent`, `text-on-accent`). The same utility class flips by scope — no per-component theme branching.
- The unused `primary` (sky-blue) scale is removed.
- `layout.tsx`: `<body>` carries the font CSS vars; `<main>` uses `bg-canvas text-ink` (dark by default).
- **Report surface scoping:** `report/[id]/page.tsx` and `report/[id]/print/page.tsx` wrap their document content in a `.theme-light` container so the report (and PDF) render with the light token set regardless of the dark shell. The report's own header/toolbar live inside this light surface (the report reads as a light "document" within the dark app).
- Respect `prefers-reduced-motion`: entrance/draw animations gated off when set.

---

## 4. Per-surface treatment

**Foundation**
- `tailwind.config.js` — token mapping, font families, remove `primary`.
- `globals.css` — dark `:root` + `.theme-light` tokens, grain overlay, base body, reduced-motion.
- `layout.tsx` — Fraunces/Geist/Geist Mono via next/font; dark `<main>`; top accent line.

**Dark shell surfaces**
- `components/AppHeader.tsx` — dark nav; refined brand mark + Fraunces wordmark; mono credits pill (ember ⚡); Reports/Credits links; account email; sign out. Hidden on `/report/*/print` (unchanged).
- `components/CreditsBadge.tsx` — mono pill; ember when low; links `/credits`.
- `page.tsx` (home / upload) — dark hero header (eyebrow + Fraunces headline + sub), the **uploader as the focus** below/beside it, generate button (teal + glow) with cost label, the multi-step progress treatment restyled. *(This is the upload tool restyled — not the full marketing landing, which is a separate downstream project.)*
- `login/page.tsx`, `welcome/page.tsx`, `anon-limit/page.tsx` — dark cards, teal accents, icon tiles; keep existing credits copy.
- `credits/page.tsx` — dark; balance hero in big mono numerals; history list with hairline rows; ember "+grant" / muted debits; notify CTA.
- `reports/page.tsx` (My Reports) — dark; report cards/list with Fraunces titles + mono meta.
- `components/OutOfCreditsModal.tsx`, `components/UpsellModal.tsx` — dark, match shell; ember out-of-credits chip.

**Light report surfaces** (`.theme-light`)
- `report/[id]/page.tsx` — light document wrapper; layout/data flow unchanged.
- `ReportSummary.tsx` — Fraunces report title, mono meta (generated · time · N charts), summary body.
- `Toolbar.tsx` — light, sits on the document; generate-more (cost + spinner, unchanged behavior) + export PDF; teal primary on light.
- `ChartCard.tsx`, `DataQualityCallout.tsx`, `Sidebar*.tsx` — light cards, Fraunces titles, Fraunces-italic captions, mono labels.
- `charts/chartTheme.ts` (**new** shared module) — the base ECharts option/theme described in §1 (palette, stripped axes, hairline split-lines, mono labels, styled tooltip, smooth lines, soft markers, area gradient, rounded bars, real-data robustness). Single source of truth for chart styling.
- `charts/*` (8 components) — consume `chartTheme`; add per-type polish (line area-gradient, bar radius/width, pie donut/label treatment, scatter opacity, box-plot styling). Keep all logic/guards and the heatmap dense-grid fix.

**PDF**
- `report/[id]/print/page.tsx` — `.theme-light`; ensure Fraunces/Geist load in the print route so the export is brand-consistent. Otherwise layout unchanged.

---

## 5. Scope & non-goals

**In scope:** purely visual restyle of all existing surfaces; new token system; font swap; grain/accent/motion details; chart palette refresh.

**Non-goals:**
- No behavior, routing, auth, credit, or data-flow changes. All component props/state/handlers preserved.
- No new pages. The **marketing site / FAQ is a separate downstream project** that will inherit these tokens.
- No dark charts / no dark report variant now (reports stay light).
- No new runtime dependencies beyond the three Google fonts.
- No backend changes.

---

## 6. Verification

- `npx tsc --noEmit` clean; `npm run build` exit 0.
- Visual pass (via local preview) of every surface: dark shell (home, login, welcome, anon-limit, credits, reports, header, modals) and the light report + its charts.
- PDF export still renders correctly and on-brand (light).
- Behavior regression check (manual): magic-link/Google login, anonymous → account claim, generate report (cost + 402 path), generate-more (cost + spinner), credits history, sign out — all unchanged.
- No console errors; fonts load without layout shift.
