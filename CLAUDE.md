# CLAUDE.md

Guidance for Claude Code sessions working on ChartSage.

## Project docs
- **Architecture & how to add a chart kind:** [ChartSage.md](ChartSage.md)
- **Report UX checklist (what to eyeball in a generated report):** [docs/report-ux-checklist.md](docs/report-ux-checklist.md)
- **Design specs:** `docs/superpowers/specs/`
- **Run / deploy:** [README.md](README.md) — frontend → Vercel on `git push origin main`. Backend → Cloud Run, **manual** (no auto-trigger): for **code-only** changes run **`scripts/deploy-backend.sh [<sha>]`** — it builds the image and swaps it (`run services update --image`), preserving all env/secrets, then checks `/health`. ⚠️ Only use `gcloud builds submit --config cloudbuild.yaml` for **env/secret changes**, and then pass **every** `--substitutions` — its defaults are placeholders that clobber `SUPABASE_URL` + Stripe envs.

## Fix the generator, not the dataset
When a report or chart looks wrong, fix it at the **point of generation** so the fix holds for
*every* dataset — not a one-off patch for the CSV in front of you. Every chart bug we've hit was a
**general** flaw, never a property of that one file: a flat fallback bar, a year rendered as a
histogram, a line tangled by self-grouping, a "3-mo avg" label on yearly data — each lived in chart
selection (`src/api/fallback.py`), an executor (`src/api/chart_executor.py`), or a chart component
(`src/app/report/[id]/charts/`). The discipline:
1. **Reproduce** it locally (load the CSV → `profile_dataframe` → the failing path) so you understand the *class* of input that triggers it.
2. **Write a failing test, then fix it at the source** (TDD) — generalise from "this CSV" to "any dataset shaped like this".
3. **Log it in [docs/report-ux-checklist.md](docs/report-ux-checklist.md)** so the next report (and the next session) catches the whole class.

Reshaping the input CSV is a last resort — at most a column-naming hint (e.g. don't name a goal-count
column `margin`, which auto-formats as a %), never the actual fix. **Before a marketing post, generate
the report yourself via the API and QA the *rendered* hero (see the `event-data-post` skill) — never
hand the user a report to publish that you haven't verified.**

## Analytics & events — treat as first-class
Events/analytics are how we see what the product **and the models** are doing — activation, credit economics, and model-output quality (e.g. the chart **fallback rate**). **Consider analytics on every change:** when you add or change a feature, ask "what event proves this works / is used?" and emit or update one.
- **Every event — name, trigger, props, lifecycle: [docs/analytics-events.md](docs/analytics-events.md).**
- **Keep that dictionary in sync in the same change** (add / rename / re-prop / remove an event → update it, dated). Backend events go through `PostHogServer` (`src/api/posthog_server.py`); analytics must never break a product flow.

## Marketing / social posting
- [docs/marketing-strategy.md](docs/marketing-strategy.md) — strategy (organic / product-led growth).
- [docs/marketing-social-playbook.md](docs/marketing-social-playbook.md) — X + Reddit execution (Tracks A/B/C).
- [docs/marketing-x-algorithm.md](docs/marketing-x-algorithm.md) — X algorithm reference (link-in-reply, hashtags 0–1 event-only, author-reply is the top signal).
- [docs/marketing-utm-conventions.md](docs/marketing-utm-conventions.md) — canonical UTM taxonomy (organic `x`/`social` vs paid `x_ads`/`paid`), the live ad register, and an ad-learnings log. **Tag every marketing link** so PostHog attributes it.
- **Posting status & history → [.claude/skills/event-data-post/analyses-log.md](.claude/skills/event-data-post/analyses-log.md)** — the source of truth for every post (`Posted` / `Ready` / `Backlog` / `Paused`). **Keep it current:** update a post's status the moment it changes, and log new posts + banked angles as they arise.
- **[docs/marketing-event-data-post-playbook.md](docs/marketing-event-data-post-playbook.md) — the repeatable event → dataset → report → post workflow.** Claude researches an upcoming event and pitches surprising datasets; **you** fetch the raw data; Claude cleans it (ASCII-safe), proposes the post angle, and saves the CSV to `~/Downloads`; **you** run + publish the report in the app with Claude's custom prompt and share the link; Claude renders the clean chart image to `~/Downloads` and lays out the thread (chart image + **one** event hashtag in the main tweet, **link in reply 1**).
