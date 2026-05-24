# ChartSage

Drop a CSV or Excel file. Get a narrated data report with charts in under 10 seconds.

## What it does

ChartSage profiles your data, asks Claude to pick the 5-7 charts that tell the most useful story, renders them with ECharts, and wraps the result in a written executive summary plus a data-quality callout when something looks off in your data.

## Tech Stack

- **Frontend:** Next.js 14, React, TypeScript, Tailwind CSS, ECharts
- **Backend:** FastAPI, Python 3.11+, pandas, Pydantic v2
- **AI:** Claude via Anthropic SDK (Haiku 4.5 default; switchable)
- **Storage:** Redis (24-hour session TTL)

## Getting started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Redis running on localhost:6379 (`brew install redis && brew services start redis`)
- An Anthropic API key

### Setup

```bash
# Backend
cp .env.example .env
# edit .env to add ANTHROPIC_API_KEY
pip install -r requirements.txt

# Frontend
npm install
```

### Run

In one terminal:

```bash
make dev          # FastAPI on :8000
```

In another:

```bash
npm run dev       # Next.js on :3000
```

Open `http://localhost:3000`, drop a CSV, see a report.

## Switching models

Default is `haiku-4-5` (~$0.01 per report). Switch by setting one env var:

```bash
CLAUDE_MODEL=sonnet-4-6 make dev          # ~$0.035/report
CLAUDE_MODEL=opus-4-7 make dev            # ~$0.04/report
```

Per-pass overrides (cheap selection, smarter narrative):

```bash
CLAUDE_MODEL_SELECTION=haiku-4-5
CLAUDE_MODEL_NARRATIVE=sonnet-4-6
```

## Tests

```bash
make test         # unit + integration (~4s, no API calls)
make test-e2e     # real Claude smoke tests (~60s, ~$0.06)
```

## Architecture

See [docs/superpowers/specs/2026-05-23-chartsage-rebuild-design.md](docs/superpowers/specs/2026-05-23-chartsage-rebuild-design.md).

## License

MIT.
