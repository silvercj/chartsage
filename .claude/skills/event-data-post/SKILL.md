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

Two steps are the user's: **fetch gated data** (login/large sources you can't pull) and
**publish the final report under the brand account** (Step 6). Everything else is yours —
including **generating and QA-ing a throwaway copy of the report yourself first** (Step 5), so the
user only ever publishes something you've already verified. Never hand over a report you haven't
seen rendered.

## The loop

**1. Spot the hook — _you_.** Usually an event ~3–10 days out with a passionate
audience: race weekends, finals/derbies, elections, awards, launches,
anniversaries, seasonal moments (hurricane-season opening, graduation, tax day).
But it **doesn't have to be sports, or even an event** — any genuinely interesting
*topical* angle works, and if nothing's on the calendar a timeless "huh, really?"
dataset is fine too. Web-search what's coming up if unsure. Pick something where
data can say something non-obvious — the post lives or dies on the *surprise*.

**2. Find a rich dataset — _you (→ user if gated)_.** Reach for **one genuinely rich,
already-published dataset** — many columns/rows or multiple tables (the Fjelstul World
Cup DB, a full Kaggle dataset, a NOAA / data.gov / OWID / FiveThirtyEight file) — *not*
a thin table built to fit a point you already have in mind. **Don't bring a hypothesis.**
You're not hunting for numbers to confirm a hunch; you're hunting for a dataset worth
*exploring*. Verify it exists and pull it (public files yourself — GitHub / NOAA / OWID
/ a Kaggle HF mirror; login-walled Kaggle → the user). Never *assemble* a table from
scraped figures ("team value + titles + drought") — that's the factoid trap: a graphic,
not an analysis.

**3. EDA → find the angles — _you_.** Explore the data *openly* — profile the columns,
aggregate, rank, look across time, correlate — and let the interesting findings
**emerge** (don't force the one you walked in with). Surface **2–4 candidate angles**
with their headline numbers, then:
- **Pick the single most surprising, screenshot-worthy one** to post now — and **bank
  the rest** in `analyses-log.md` (a one-liner each) as future posts. *Note them, don't
  action them.* (Check the log first — don't re-pitch an angle we've already posted.)
- The hook is the **finding the analysis surfaces** (often *not* the one you expected).
  The ChartSage report you'll brief in Step 5 is literally this analysis, rendered.
- **Keep it rich — don't distill to the hook.** Hand ChartSage the *richest* table that still
  contains the hook metric (many columns/rows), not a tiny one cut down to the single narrative.
  A thin table starves chart selection — the model under-picks → the fallback fires → duplicate/
  generic charts → a thin report — and the rendered report **is** the demo. ChartSage *aggregates*
  (mean/sum/count by a group) but won't compute a **derived ratio** (cards-per-game = bookings ÷
  matches), so pre-compute **only that one hook metric** as a column among many and keep the rest.
  (For a *trend* hook, per-period rows; for *ranking/distribution* hooks, near-raw per-row data is
  even better. The "ChartSage can't derive metrics" gap is logged in `docs/FUTURE-IMPROVEMENTS.md`.)
- **Clean** it: re-encode ASCII-safe (accented names like `Nürburgring` store as mojibake
  otherwise) and **name columns so ChartSage types them right** — a `pct`/`rate`/`share`/`margin`/
  `ratio` column auto-formats as a percentage (see `_PERCENTAGE_KEYWORDS` in
  `src/api/chart_executor.py`; 0–100 values are fine, the backend normalises).
- **Propose the post**: the one-line hook + the exact numbers, for the user to OK.

**4. Hand back the CSV — _you → user_.** Save the cleaned file to `~/Downloads/`
with a clear name (e.g. `f1_pole_by_circuit_ascii.csv`) so they can upload as-is.

**5. Self-QA the report yourself — _you_, before anyone publishes.** Don't hand the user a report
to publish blind. Generate a **throwaway** QA report via the API (under the QA anon id — never
published, never on the brand account) and verify it:

```bash
~/.venvs/chartsage/bin/python scripts/qa_generate.py ~/Downloads/<file>.csv "<your custom prompt>"
```
It prints the `session_id` + a chart QA — flagging **EMPTY** / **SELF-GROUP** charts, **FALLBACK**
picks, and **TIME-SCATTER** advisories. Then render it **as owner, no publish needed**, and eyeball
the hero + full page against the **UX checklist** (`docs/report-ux-checklist.md` — axes fitting the
data, no empty/degenerate charts, % formatted right):

```bash
~/.venvs/chartsage/bin/python scripts/qa_render.py https://chartsage.app/report/<id>
```
**If anything's wrong, fix the _generation_ durably** (CLAUDE.md → *Fix the generator, not the
dataset*: reproduce → failing test → fix in `fallback.py` / `chart_executor.py` / the chart
component → deploy), not a one-off CSV hack. Re-QA until clean, then share the rendered hero with
the user. *(Custom prompt: name the hero **plus a couple of supporting charts**; don't say "ONE
chart / minimise" — that under-selects into the fallback. ≤280 chars.)*

> The QA scripts run under a fixed QA anon id at `~/.chartsage/qa-anon-id` (in the backend
> `UNLIMITED_ANON_IDS` allowlist, so it's uncapped) and hit the prod API (`CHARTSAGE_API_URL`).

**6. User publishes the verified report; you render the post image — _user → you_.** Only once the
QA report is clean: the user uploads the **same CSV + prompt** under the brand account, **Publishes**
(Share in the toolbar — resolves the link *and* generates the OG image), and sends the URL. Confirm
their hero matches your QA one (same fixes are deployed, so it will), then render the clean post
image (title, chart + the narrative caption; no UI chrome) to `~/Downloads/`:

```bash
~/.venvs/chartsage/bin/python scripts/chart_image.py \
  https://chartsage.app/report/<id> ~/Downloads/<event>_chart.png
```
(All three scripts — `qa_generate.py`, `qa_render.py`, `chart_image.py` — are in this skill's
`scripts/`. The project venv `~/.venvs/chartsage` already has Playwright + chromium.)

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

**9. Log it — _you_, immediately.** The moment a post goes live, append it to
**`analyses-log.md`** (this skill's dir): date, hook, event/topic, dataset + source,
report URL, hashtag. **Don't batch it** — a stale log means repeated angles (it's how
we nearly re-posted World Cup host advantage). Keep the **Banked angles** list there
current too (Step 3), and **skim the whole log before every pitch**.

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
- **Self-QA before anyone publishes.** Don't hand over a blind report — *you* generate + render a
  throwaway QA copy first (`qa_generate.py` / `qa_render.py`, Step 5). Haiku is non-deterministic on
  small tables (under-picks → fallback; self-groups a line into a spiky tangle; picks scatters for
  time series), so every run differs — verify the *actual* charts, and when one's wrong fix the
  generator, not the CSV. (Fallback rate lives in PostHog's `report_charts_composed`.)
- **Don't over-constrain the upload prompt.** "ONE chart / minimise others" makes the
  model under-select → fallback. Name the hero, allow a few supporting charts.
- **Don't over-distill the data either.** Pre-aggregating down to the hook (a 4-column table)
  starves selection → fallback → duplicate charts → a thin report. A rich table (e.g. the World Cup
  *discipline* post: 7 columns → 10 varied charts, zero fallback) showcases ChartSage *and* dodges
  those bugs. Upload rich; pre-compute only the derived hook metric (Step 3).
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
