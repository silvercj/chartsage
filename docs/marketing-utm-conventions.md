# Marketing UTM conventions & ad register

Canonical UTM taxonomy for **every** ChartSage marketing link — organic and paid —
so PostHog attributes the traffic and the *Product & Model Health* dashboard's
acquisition tiles read correctly. **Tag every outbound marketing link.** An untagged
link lands in `(direct / none)` and is invisible to attribution.

## Taxonomy

| Param | Organic X post | Paid X ad | Meaning |
|---|---|---|---|
| `utm_source` | `x` | `x_ads` | the channel — keep **paid vs organic distinct** |
| `utm_medium` | `social` | `paid` | |
| `utm_campaign` | `<event>` (e.g. `monaco_gp`) | `val_<theme>` (e.g. `val_worldcup`) | the campaign / test |
| `utm_content` | `<angle>` | `<creative>` (e.g. `worldcup`, `direct`) | the specific post angle / ad-creative variant |

- **Organic** (event-data-post skill, reply-1 link):
  `?utm_source=x&utm_medium=social&utm_campaign=<event>&utm_content=<angle>`
- **Paid** (X Ads destination URL):
  `?utm_source=x_ads&utm_medium=paid&utm_campaign=val_<theme>&utm_content=<creative>`

> Source split is deliberate: `x` = organic posts, `x_ads` = paid ads. Dashboard tiles
> that look at "X" use `utm_source IN ('x', 'x_ads')`; ad-only tiles filter `x_ads`.

## Current ad register

| Campaign | Creative (`utm_content`) | X Ads name | Destination URL |
|---|---|---|---|
| `val_worldcup` | `worldcup` | Ad A (timely) | `https://chartsage.app/?utm_source=x_ads&utm_medium=paid&utm_campaign=val_worldcup&utm_content=worldcup` |
| `val_worldcup` | `direct` | Ad B (direct) | `https://chartsage.app/?utm_source=x_ads&utm_medium=paid&utm_campaign=val_worldcup&utm_content=direct` |

*(Append new campaigns/creatives here as you launch them.)*

## Reading it in PostHog (Product & Model Health dashboard)

- **Traffic by UTM source** — `x_ads` (paid) vs `x` (organic) vs `(direct / none)`.
- **X traffic by campaign (organic + ads)** — `utm_source IN ('x','x_ads')`, grouped by campaign.
- **X ads — visitors by creative** — `x_ads` split by `utm_content` (the on-site A/B; complements X Ads' click data).
- **Attribution caveat:** anonymous visitors carry the UTM on their `$pageview`, so source/campaign tiles work. But person-level `$initial_utm_*` is only set for *identified* (signed-up) users (`person_profiles: 'identified_only'`), so post-signup attribution is partial.

## Ad learnings log

*(Append dated notes as campaigns run.)*

- **2026-06-07 — `val_worldcup`, first run.** Delivery only started 5 Jun (eligibility
  halt 1–4 Jun). ~6.4k impressions, **12 link clicks** (CTR 0.19%, ~$0.63/click) over ~2
  days — far too small to judge. Audience was badly mismatched: **97% mobile** (a CSV
  upload tool needs desktop) and **61% age 18–24**. → switched to **desktop + 25–54**
  device/age targeting. Ad B (`direct`) ran ~2× Ad A (`timely`) CTR (8 vs 4 clicks), but
  sample too small to call. Watch **click → report generated**, not signups (first report
  is free; signups lag).
