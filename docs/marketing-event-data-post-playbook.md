# Event-Data Post Playbook — timely data showcases

*The repeatable recipe for turning an upcoming event into a ChartSage data-showcase post. Execution layer for [marketing-social-playbook.md](marketing-social-playbook.md) Track A, optimised per [marketing-x-algorithm.md](marketing-x-algorithm.md). Created 2026-06-05; first run = the Monaco-GP pole-to-win post.*

## The idea

Ride a live event's built-in audience: find a dataset that says something **surprising** about the event, turn it into one screenshot-worthy ChartSage chart, and post it while the event peaks. The interactive report *is* the product demo.

## The loop (who does what)

1. **Spot the event — _Claude_.** Scan for events ~3–10 days out with a passionate audience and a timely hook: F1 race weekends, major finals/derbies, elections, awards, launches, anniversaries, seasonal moments. Pick one where data can say something non-obvious.

2. **Pitch datasets — _Claude → you_.** Claude proposes 1–3 specific datasets, each with a *surprising angle*, and where to get them (Kaggle, Ergast/Jolpica F1, data.gov, Our World in Data, FiveThirtyEight, official stats). Claude often **can't pull gated/large/login-walled sources**, so **you fetch the raw file** and hand it over (drop it in `~/Downloads`, or paste a link).

3. **Clean + find the story — _Claude_.** Claude profiles the data and:
   - finds the **surprising, screenshot-worthy** finding (the hook);
   - **cleans** it — ASCII-safe encoding (kills mojibake like `Nürburgring` → `Nurburgring`), drop junk columns, keep the ones that tell the story, and **name columns so ChartSage infers types** (a `pct`/`rate`/`share`/`margin`/`ratio` column auto-formats as a percentage — see `_PERCENTAGE_KEYWORDS` in `src/api/chart_executor.py`; 0–100 values are fine, the backend normalises to fractions);
   - **proposes the post**: the one-line hook + the exact numbers.

4. **Hand back the clean CSV — _Claude → you_.** Claude **saves the cleaned dataset to `~/Downloads/`** with a clear name (e.g. `f1_pole_by_circuit_ascii.csv`) so you can upload it as-is.

5. **Generate + publish — _you_.** In the app (logged in, so it's under the brand account), upload the CSV **with the custom prompt Claude gives you** — the prompt steers ChartSage to the chart/angle we want (e.g. *"Lead with a bar chart ranking circuits by pole_to_win_pct, highest first; minimise generic count charts."*; ≤280 chars). Review it, set the hero how you want it (collapse/expand the ranking), **Publish** (Share in the toolbar — this makes the link resolve **and** generates the OG image), then **send Claude the report URL**.

6. **Make the image — _Claude_.** Claude renders a **clean chart image** (title + chart only, no UI chrome) to `~/Downloads/` for the main tweet.

7. **Lay out the post — _Claude_.** Per the X-algorithm playbook:
   - **Main tweet:** hook + the surprising stat + **the chart image** + a soft question ("which surprised you?") + **one well-researched event hashtag** — the event's *actual* tag (e.g. `#MonacoGP`), never a generic topic tag. **No link in the main tweet.**
   - **Reply 1 (self, immediately):** the `chartsage.app/report/…` link with UTM (`?utm_source=x&utm_medium=social&utm_campaign=<event>&utm_content=<angle>`) + a one-line product plug.
   - **Reply 2 (self, optional):** a bonus stat to keep the thread alive.

8. **Schedule + tend — _you_.** Schedule the main tweet (Buffer) for when you can be present, drop Reply 1 right after, and **reply to every genuine human reply in the first hour** — your author-reply is the single biggest reach signal (~150× a like).

## The single hashtag

One **event** tag, researched — the tag the event's real audience uses *that week* (check what's actually trending for it): `#MonacoGP`, `#SuperBowl`, `#Euro2028`, etc. Never a generic topic tag (`#F1` adds nothing — the model reads semantics), never more than one (3+ trigger a penalty). See [marketing-x-algorithm.md](marketing-x-algorithm.md) §0.6 / §4.

## Gotchas (learned the hard way)

- **ASCII-clean the CSV**, or accented names store as mojibake in the report.
- **The custom prompt is essential** — without it ChartSage may lead with a generic distribution chart instead of your angle.
- **Publish before posting** — an unpublished report's link won't resolve for clickers, and no OG image is generated.
- **Link goes in the reply, not the main tweet** (external links cut reach ~30–50%+).
- **Wide ranking charts** render full-width (vertical, all bars, value labels where they fit) and collapse to a top-12 card; for the landscape OG/social image the collapsed top-N often crops better than the full ranking.

## Worked example — Monaco GP pole-to-win (2026-06-05)

- **Event:** Monaco GP weekend.
- **Dataset:** F1 pole-to-win conversion rate by circuit → `~/Downloads/f1_pole_by_circuit_ascii.csv`.
- **Finding:** pole converts to a win only ~44% of the time on average; Monaco **46%** (≈ average), Barcelona **71%** (the real fortress).
- **Prompt:** *"Lead with a bar chart ranking circuits by pole_to_win_pct, highest first…"*
- **Hashtag:** `#MonacoGP`.
- **Post:** hook + chart image + "which surprised you?" + `#MonacoGP` → Reply 1 = the report link (UTM) → Reply 2 = the flip side (Rio 10%, Kyalami 20%, Indianapolis 26%).
