---
name: event-data-post
description: >-
  Create a timely, data-driven X/Twitter post for the ChartSage brand account,
  built around an event. Use this whenever the user wants a brand social post —
  "let's post something for the Monaco GP", "find a dataset for the World Cup and
  make a tweet", "what could we post for the Oscars?", "ride the euros hype with
  a chartsage post" — AND on open-ended brand-post requests like "create a new
  post for me" or "time for a fresh post", which default to this data-showcase
  workflow (you research the event + dataset). It runs the whole loop: research
  the event → pitch surprising datasets → (the user fetches the raw data) → clean
  it + propose the post + save the CSV to ~/Downloads → (the user runs +
  publishes the report with your custom prompt) → render a clean chart image →
  lay out the thread with ONE researched event hashtag and the link in reply 1.
  Do NOT use it for: plain report generation from the user's own file, posting
  specific pre-decided content (a feature or pricing announcement), chart/layout
  code fixes, standalone data cleaning, or generic social-strategy questions.
---

# Event-data social post

Turn an upcoming event into a ChartSage data-showcase post: a dataset you
**analyse** into a surprising finding about the event → one screenshot-worthy
chart → a short X thread, posted while the event peaks. The interactive report
*is* the demo, and the analysis *is* the post.

This is the execution layer for `docs/marketing-social-playbook.md` (Track A),
optimised per `docs/marketing-x-algorithm.md`. Canonical reference with more
detail: `docs/marketing-event-data-post-playbook.md`.

## You vs. the user — hand off cleanly

Some steps only the user can do: **fetch gated data** (login/large sources you
can't pull) and **run + publish the report under the brand account**. Don't try
to work around those — give them exactly what they need and wait for the result.
Everything else (research, cleaning, the image, the copy) is yours.

## The loop

**1. Spot the hook — _you_.** Usually an event ~3–10 days out with a passionate
audience: race weekends, finals/derbies, elections, awards, launches,
anniversaries, seasonal moments (hurricane-season opening, graduation, tax day).
But it **doesn't have to be sports, or even an event** — any genuinely interesting
*topical* angle works, and if nothing's on the calendar a timeless "huh, really?"
dataset is fine too. Web-search what's coming up if unsure. Pick something where
data can say something non-obvious — the post lives or dies on the *surprise*.

**2. Find real datasets — _you → user_.** Find 1–3 *already-published* datasets — an
actual Kaggle dataset, a NOAA / data.gov file, an Our World in Data / FiveThirtyEight
/ Ergast CSV — and **verify each exists** (open the page, pull the file). You're
*finding* a dataset, **not assembling one**: stitching scraped figures into a table
("team value + titles + drought") is the trap — that's a graphic, not an analysis.
Frame each as a **question the raw rows answer** — a rate to compute, a ranking, a
trend, a correlation (Monaco worked because pole→win % was *computed per circuit*).
Truly-public files you pull yourself (GitHub / NOAA / OWID, sometimes a Kaggle mirror
on Hugging Face); for login-walled Kaggle the user grabs it. Pitch your expected
finding as a *hypothesis* — the real hook is whatever Step 3 surfaces. Let them pick.

**3. Analyse + find the story — _you_.** Actually work the raw data — aggregate,
rank, compute the rate, look across time — and let the finding *emerge*:
- The hook is the **surprising, screenshot-worthy finding the analysis surfaces**
  (often *not* your Step-2 hunch). The ChartSage report you'll brief in Step 5 is
  literally this analysis, rendered — so the dataset has to be worth analysing.
- **Clean** it: re-encode ASCII-safe (accented names like `Nürburgring` store as
  mojibake otherwise), drop junk columns, keep what tells the story, and **name
  columns so ChartSage types them right** — a `pct`/`rate`/`share`/`margin`/
  `ratio` column auto-formats as a percentage (see `_PERCENTAGE_KEYWORDS` in
  `src/api/chart_executor.py`; 0–100 values are fine, the backend normalises).
- **Propose the post**: the one-line hook + the exact numbers, for the user to OK.

**4. Hand back the CSV — _you → user_.** Save the cleaned file to `~/Downloads/`
with a clear name (e.g. `f1_pole_by_circuit_ascii.csv`) so they can upload as-is.

**5. Generate + publish — _user_.** They upload it in the app (logged in → brand
account) **with the custom prompt you give them**. The prompt steers ChartSage to
your angle — e.g. *"Lead with a bar chart of pole_to_win_pct by circuit, ranked
highest first."* (≤280 chars). **Name the hero chart, but don't over-constrain** —
"make ONE chart / minimise others" backfires: chart-selection then picks too few and
the heuristic fallback fills the report with generic distributions. Ask for the hero
**plus a few supporting charts**. They review it, set the hero, **Publish** (Share in
the toolbar — makes the link resolve *and* generates the OG image), and send you the
report URL.

**6. Check the report, then make the chart image — _you_.** **First open the
published report and confirm the hero is the chart your analysis intended** — right
columns, real values, not a flat/generic "<column> — distribution" fallback. If it's
wrong, fix it *before* posting (re-shape the CSV, re-run with a clearer prompt) —
never hand over a broken report. Once it's right, render a clean image (title + chart
only, no UI chrome) to `~/Downloads/` with the bundled script — it handles loading,
stripping the buttons/index/caption, and shooting the hero (first) card:

```bash
~/.venvs/chartsage/bin/python scripts/chart_image.py \
  https://chartsage.app/report/<id> ~/Downloads/<event>_chart.png
```
(`scripts/chart_image.py` is in this skill's directory. The project venv at
`~/.venvs/chartsage` already has Playwright + chromium.)

**7. Lay out the post — _you_.** Per the X algorithm:
- **Main tweet:** hook + the surprising stat + **the chart image** + a soft
  question ("which surprised you?") + **one event hashtag**. **No link.**
- **Reply 1 (self, immediately):** the `chartsage.app/report/…` link with UTM
  `?utm_source=x&utm_medium=social&utm_campaign=<event>&utm_content=<angle>` + a
  one-line product plug.
- **Reply 2 (self, optional):** a bonus stat to keep the thread alive.

**8. Schedule + tend — _user_.** They schedule the main tweet (Buffer) for when
they can be present, drop Reply 1 right after, and reply to every genuine human
reply in the first hour — the author-reply is the single biggest reach signal
(~150× a like). Remind them of this; a scheduled-and-abandoned post underperforms.

**9. Log it — _you_.** Append the post to **`analyses-log.md`** (this skill's dir):
date, hook, event/topic, dataset + source, the report URL, and the hashtag. It's our
running record — skim it first so we never repeat an angle.

## The one hashtag — research it, don't guess

Use a **single event tag**: the tag that event's real audience uses *that week*
(check what's actually trending for it) — `#MonacoGP`, `#SuperBowl`, `#Euro2028`.
Never a generic topic tag (`#F1` adds nothing — the model reads semantics), and
never more than one (3+ trigger a penalty). When in doubt, look up the live tag
rather than inventing one. (`docs/marketing-x-algorithm.md` §0.6 / §4.)

## Things that bite (learned the hard way)

- **It's analysis, not a factoid.** Every post comes from *working a real dataset*
  (compute a rate, rank, trend, correlation — like F1 pole→win % per circuit), not
  from plotting one or two numbers you already knew. If you can't name the analysis
  step, you don't have a post yet — keep digging.
- **Find the dataset, don't build it.** Use a real published file (Kaggle / NOAA /
  data.gov / OWID / FiveThirtyEight). Assembling a table from scraped figures is the
  factoid trap in disguise. Confirm the source exists before pitching — and mind URLs
  (NOAA's HURDAT2 lives on `www.nhc.noaa.gov`, not the ftp host).
- **Verify the report before you post.** Chart-selection (Haiku) can under-pick on
  small/odd tables and silently fall back to generic distribution charts — open the
  published report and confirm the hero is *your* analysis first. (We track this as the
  fallback rate in PostHog's `report_charts_composed`; see `docs/analytics-events.md`.)
- **Don't over-constrain the upload prompt.** "ONE chart / minimise others" makes the
  model under-select → fallback. Name the hero, allow a few supporting charts.
- **ASCII-clean the CSV**, or accented names become mojibake in the stored report.
- **The custom prompt is essential** — without it ChartSage may lead with a
  generic distribution chart instead of your angle.
- **Publish before posting** — an unpublished link won't resolve for clickers,
  and no OG image gets generated.
- **Link in the reply, not the main tweet** — external links cut reach ~30–50%+.
- **Wide ranking charts** render full-width (vertical, all bars, value labels
  where they fit) and collapse to a top-12 card; the collapsed top-N often crops
  better for a landscape social image.

## Worked example — Monaco GP pole-to-win

Event: Monaco GP weekend. Dataset: F1 pole-to-win % by circuit. Finding: pole
converts to a win only ~44% on average; Monaco 46% (≈ average), Barcelona 71%
(the real fortress). Prompt: *"Lead with a bar chart ranking circuits by
pole_to_win_pct, highest first…"*. Hashtag: `#MonacoGP`. Thread: hook + chart +
"which surprised you?" + `#MonacoGP` → reply 1 = report link (UTM) → reply 2 =
the flip side (Rio 10%, Kyalami 20%, Indianapolis 26%).
