# CLAUDE.md

Guidance for Claude Code sessions working on ChartSage.

## Project docs
- **Architecture & how to add a chart kind:** [ChartSage.md](ChartSage.md)
- **Design specs:** `docs/superpowers/specs/`
- **Run / deploy:** [README.md](README.md) — frontend → Vercel on `git push origin main`; backend → Cloud Run, **manual**, via `gcloud builds submit --config cloudbuild.yaml --substitutions=...` (no auto-trigger).

## Marketing / social posting
- [docs/marketing-strategy.md](docs/marketing-strategy.md) — strategy (organic / product-led growth).
- [docs/marketing-social-playbook.md](docs/marketing-social-playbook.md) — X + Reddit execution (Tracks A/B/C).
- [docs/marketing-x-algorithm.md](docs/marketing-x-algorithm.md) — X algorithm reference (link-in-reply, hashtags 0–1 event-only, author-reply is the top signal).
- **[docs/marketing-event-data-post-playbook.md](docs/marketing-event-data-post-playbook.md) — the repeatable event → dataset → report → post workflow.** Claude researches an upcoming event and pitches surprising datasets; **you** fetch the raw data; Claude cleans it (ASCII-safe), proposes the post angle, and saves the CSV to `~/Downloads`; **you** run + publish the report in the app with Claude's custom prompt and share the link; Claude renders the clean chart image to `~/Downloads` and lays out the thread (chart image + **one** event hashtag in the main tweet, **link in reply 1**).
