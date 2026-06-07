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
  it + propose the post + save the CSV to ~/Downloads → (you generate, self-QA,
  and publish the report via the ChartSage accounts; the user approves it) → render a clean chart image →
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

The user's role is **provide + approve + post**: they **fetch gated data** (login/large sources you
can't pull), **review the report at two gates** (the QA-account stage and the final content-account
report), and **approve + post on X** (one-click-approve the Buffer draft, then tend the replies — no
Twitter API; X stays Buffer-mediated). Everything else is yours: find + enrich the data,
**generate → QA → publish** it on the ChartSage QA and content accounts (see *Automated pipeline*),
render the image, write the thread, and **draft it into the Buffer queue** (see *Buffer scheduling*).
Never put a report in front of them you haven't rendered + QA'd yourself.

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

**4. Save the cleaned CSV — _you_.** Save the cleaned file to `~/Downloads/` with a clear name
(e.g. `f1_pole_by_circuit_ascii.csv`) — it feeds the pipeline (Steps 5–6).

**5. Generate + self-QA + stage for review — _you_.** Generate on the **QA account**, QA it, fix any
generator bugs at the source (don't re-roll — *Automated pipeline* §2), then **publish it and send the
user the link to QA the whole report (Gate 1)**. The scripts, gates, and the diagnose-before-reroll
discipline live in **[Automated pipeline](#automated-pipeline-qa-account--content-account)** below.

**6. Publish the real one + render the image — _you_.** On the user's OK, run the same data + prompt
through the **content account**, QA + publish it (Gate 2 = the user's final OK), then render the hero
(title + chart + narrative, no chrome) to `~/Downloads/`:

```bash
~/.venvs/chartsage/bin/python scripts/chart_image.py \
  https://chartsage.app/report/<id> ~/Downloads/<event>_chart.png
```
(Scripts — `qa_generate.py`, `qa_render.py`, `publish.py`, `chart_image.py` — are in this skill's
`scripts/`; the project venv `~/.venvs/chartsage` has Playwright + chromium.)

**7. Lay out the post — _you_.** Per the X algorithm:
- **Main tweet:** hook + the surprising stat + **the chart image** + a soft
  question ("which surprised you?") + **one event hashtag**. **No link.**
- **Reply 1 (self, immediately):** the `chartsage.app/report/…` link with UTM
  `?utm_source=x&utm_medium=social&utm_campaign=<event>&utm_content=<angle>` + a
  one-line product plug.
- **Reply 2 (self, optional):** a bonus stat to keep the thread alive.

**8. Schedule + tend — _you draft, user approves + tends_.** Pick the slot with
**`buffer_manager.py`** (it reconciles the live queue and flags the gaps), propose it, and on the
user's OK **schedule the thread as a Buffer draft** (`buffer_schedule.py`; image = the report's
`og_image_url`) — see *Buffer scheduling*. The user one-click-approves the draft in Buffer (the
pre-set time is kept), then replies to every genuine human reply in the first hour — the author-reply
is the single biggest reach signal (~150× a like). Remind them; a scheduled-and-abandoned post
underperforms.

**9. Log it — _you_, immediately.** The moment a post goes live, append it to
**`analyses-log.md`** (this skill's dir): date, hook, event/topic, dataset + source,
report URL, hashtag. **Don't batch it** — a stale log means repeated angles (it's how
we nearly re-posted World Cup host advantage). Keep the **Banked angles** list there
current too (Step 3), and **skim the whole log before every pitch**.

## Automated pipeline (QA account → content account)

**Setup (one-time).** Two anon ids, both in the backend `UNLIMITED_ANON_IDS` allowlist and stored
locally: a **QA id** (`~/.chartsage/qa-anon-id`, throwaway testing) and a **content id**
(`~/.chartsage/content-id`, published marketing reports). Scripts resolve the id from `CHARTSAGE_ANON_ID`
env → `~/.chartsage/qa-anon-id`; **target the content account by prefixing `CHARTSAGE_ANON_ID=$(cat
~/.chartsage/content-id)`**. Publishing needs a settings permission rule for `publish.py` (else the
auto-mode classifier hard-blocks it); that rule only removes the per-call prompt — the human still
reviews the content at the gates.

1. **Find + enrich** (Steps 1–4). Gated sources → the user pulls them.
2. **QA account: generate + self-QA — _bounded_.**
   ```bash
   ~/.venvs/chartsage/bin/python scripts/qa_generate.py ~/Downloads/<file>.csv "<custom prompt>"
   ~/.venvs/chartsage/bin/python scripts/qa_render.py https://chartsage.app/report/<id>
   ```
   `qa_generate` prints the chart QA (flags **EMPTY** / **SELF-GROUP** / **FALLBACK** / **TIME-SCATTER**);
   `qa_render` shoots the hero + full page as owner (no publish). Check against the **UX checklist**
   (`docs/report-ux-checklist.md`). **Diagnose every QA fail; don't re-roll it** — a generator bug (dup,
   tangled line, empty chart, mislabel, scatter-for-time-series) → **stop → fix at the source
   (`fallback.py`/`chart_executor.py`/chart component) → push + deploy** (CLAUDE.md *Fix the generator*).
   Re-roll *only* for genuine Haiku variance, **capped at ~2**; persistent flakiness escalates to the
   user. Never spray-and-pray. *(Prompt: name the hero + a couple of supporting charts; ≤280 chars; never
   "ONE chart".)*
3. **QA account: publish → user QAs the whole report (Gate 1).**
   ```bash
   ~/.venvs/chartsage/bin/python scripts/publish.py <id>
   ```
   Send the link; the user reviews the *rendered* report in-browser (catches what the hero image hides).
   Issues → step 2.
4. **Content account: generate → QA → publish.** Same data + prompt under the content id; QA the
   published result. A fail here is a **bug → back to step 2** (fix + redeploy), not a re-roll.
   ```bash
   CHARTSAGE_ANON_ID=$(cat ~/.chartsage/content-id) ~/.venvs/chartsage/bin/python scripts/qa_generate.py ~/Downloads/<file>.csv "<prompt>"
   ~/.venvs/chartsage/bin/python scripts/publish.py <id> --content
   ```
5. **User final OK (Gate 2)** on the live content-account report + the drafted thread.
6. **Render the hero (`chart_image.py`) + lay out the thread (Step 7); the user posts** (X stays manual).

Both publishes (3 + 4) are content going live — the human reviews the *content* at 3 and 5; the
permission rule just removes the per-call publish prompt.

## Buffer scheduling (read the queue, fill the gaps)

Posts reach X through **Buffer** (the brand's `@chartsageapp` channel) via Buffer's **GraphQL API**
(`https://api.buffer.com`, personal key in `~/.chartsage/buffer-token`). We never touch the Twitter
API — Buffer posts to X once a draft is approved. Scripts (this skill's `scripts/`):

- **`buffer_list.py`** — read the live queue: channels, upcoming posts (with times), org limits (`--json` for the agent).
- **`buffer_manager.py`** — the manager: reconciles the queue against the **cadence policy** and prints the calendar with **GAPS** flagged + limit reasoning. Read-only; the *agent* matches each gap to a backlog post.
- **`buffer_schedule.py`** — create/delete a post-thread: `--text` / `--reply`(×N) / `--image <public url>` / `--at <ISO-8601 UTC>` / `--draft` / `--delete <id>` / `--dry-run`. Needs a settings permission rule (like `publish.py`); the human still reviews each plan.
- **`buffer_api.py`** — shared GraphQL helper. (`buffer_probe.py` = schema introspection, dev only.)

**The loop (this *is* Step 8):**
1. `buffer_manager.py` → the gaps (e.g. "Jun 13/15/17/19 empty", runway, limits).
2. *Agent* matches each gap to a **Ready/Backlog** post in `analyses-log.md` — an already-published report first, else build one via the *Automated pipeline*.
3. **Time each post** — anchor to its event so it rides the live hashtag (an F1 post near the race, a WC post around a match); else the default US window (`--slot`, ~15:00 UTC). The manager suggests a default; tune per content.
4. **Propose** the plan (report → slot → thread) for the user's OK.
5. On OK → `buffer_schedule.py --at <slot> --draft --text … --image <og_image_url> --reply … --reply …` → lands as a **Buffer draft** (`status: draft`, time preserved).
6. The **user approves** the draft in Buffer (one click; the pre-set time is kept) and tends replies. Then log it (Step 9) + update the post's status.

**Cadence policy** (the manager's defaults; flags override): **every other day** (`--spacing 2`) until we hold a ~**14-day runway**, then tighten to **daily** (`--spacing 1`); reserve **2/day for event spikes** (a race/match day), spaced US-morning + US-evening so they don't cannibalise. Scale frequency to backlog depth **and** engagement, not just the calendar.

**Image = the report's OG card, for free.** Publishing a report auto-generates a public 1200×630 OG card (hero chart + title + `chartsage.app`) at `og_image_url` (from `/report/<id>/meta`, or `{SUPABASE_URL}/storage/v1/object/public/og-images/<id>.png`). Pass it straight to `--image` — Buffer's API takes only **public URLs** (no file upload yet), so this is the zero-glue path (and lands a free brand impression in the image; beats hosting the portrait `chart_image.py` hero).

**Gotchas:** `saveToDraft:true` → `status: draft` that **keeps the `dueAt`** (an owner key never auto-skips approval — so the gate is reliable); the human approves in Buffer. `buffer_list.py` is the **source of truth** over `analyses-log.md` — reconcile statuses (a "Scheduled" row missing from the queue may already be **sent** — check sent history). Buffer's plan caps (5000 posts / 2000 threads-per-channel) won't bind — the real limits are our cadence + X's per-day cap (`dailyPostingLimits`, wireable).

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
- **Timely post → _current_ data — check the latest date FIRST.** An event post needs a dataset
  that reaches (near) the present; a chart ending years before the event looks stale and kills the
  "today" hook (we nearly shipped an F1 reliability chart ending **2021** for a 2026 race — bad). Frozen
  mirrors bite: the dead Ergast dump (and the `rubenv/ergast-mrd` mirror) stop at ~2022. For F1 use
  **Jolpica** (the live Ergast successor) or a freshly-updated Kaggle Ergast set; for any source,
  verify `max(year)`/latest row before building — if it's stale, get current data (user pulls if gated).
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
