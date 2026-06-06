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

Turn an upcoming event into a ChartSage data-showcase post: a dataset that says
something **surprising** about the event → one screenshot-worthy chart → a short
X thread, posted while the event peaks. The interactive report *is* the demo.

This is the execution layer for `docs/marketing-social-playbook.md` (Track A),
optimised per `docs/marketing-x-algorithm.md`. Canonical reference with more
detail: `docs/marketing-event-data-post-playbook.md`.

## You vs. the user — hand off cleanly

Some steps only the user can do: **fetch gated data** (login/large sources you
can't pull) and **run + publish the report under the brand account**. Don't try
to work around those — give them exactly what they need and wait for the result.
Everything else (research, cleaning, the image, the copy) is yours.

## The loop

**1. Spot the event — _you_.** Find events ~3–10 days out with a passionate
audience and a timely hook: race weekends, finals/derbies, elections, awards,
launches, anniversaries, seasonal moments. Web-search what's coming up if you're
unsure of the calendar. Pick one where data can say something non-obvious — the
post lives or dies on the *surprise*.

**2. Pitch datasets — _you → user_.** Propose 1–3 specific datasets, each with a
*surprising angle*, and where to get them (Kaggle, Ergast/Jolpica F1, data.gov,
Our World in Data, FiveThirtyEight, official stats). You usually can't pull
gated/large/login-walled sources, so ask the user to fetch the raw file and drop
it in `~/Downloads` (or paste a link). Let them pick the one that excites them.

**3. Clean + find the story — _you_.** Profile the data, then:
- Find the **surprising, screenshot-worthy** finding — that's the hook.
- **Clean** it: re-encode ASCII-safe (accented names like `Nürburgring` store as
  mojibake otherwise), drop junk columns, keep what tells the story, and **name
  columns so ChartSage types them right** — a `pct`/`rate`/`share`/`margin`/
  `ratio` column auto-formats as a percentage (see `_PERCENTAGE_KEYWORDS` in
  `src/api/chart_executor.py`; 0–100 values are fine, the backend normalises).
- **Propose the post**: the one-line hook + the exact numbers, for the user to OK.

**4. Hand back the CSV — _you → user_.** Save the cleaned file to `~/Downloads/`
with a clear name (e.g. `f1_pole_by_circuit_ascii.csv`) so they can upload as-is.

**5. Generate + publish — _user_.** They upload it in the app (logged in → brand
account) **with the custom prompt you give them**. The prompt is what steers
ChartSage to the chart/angle you want — e.g. *"Lead with a bar chart ranking
circuits by pole_to_win_pct, highest first; minimise generic count charts."*
(keep it ≤280 chars). They review it, set the hero, **Publish** (Share in the
toolbar — this makes the link resolve *and* generates the OG image), and send you
the report URL.

**6. Make the chart image — _you_.** Render a clean image (title + chart only, no
UI chrome) to `~/Downloads/` with the bundled script — it handles loading,
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

## The one hashtag — research it, don't guess

Use a **single event tag**: the tag that event's real audience uses *that week*
(check what's actually trending for it) — `#MonacoGP`, `#SuperBowl`, `#Euro2028`.
Never a generic topic tag (`#F1` adds nothing — the model reads semantics), and
never more than one (3+ trigger a penalty). When in doubt, look up the live tag
rather than inventing one. (`docs/marketing-x-algorithm.md` §0.6 / §4.)

## Things that bite (learned the hard way)

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
