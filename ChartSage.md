# ChartSage — Internal Documentation

## Architecture

CSV/Excel → DataFrame → profile → Claude (selection via parallel tool use) → executors → Claude (narrative via forced tool use) → Report → Redis (24h) → frontend renders.

Full design: [docs/superpowers/specs/2026-05-23-chartsage-rebuild-design.md](docs/superpowers/specs/2026-05-23-chartsage-rebuild-design.md).

## Adding a new chart kind

1. Add an executor to `src/api/chart_executor.py` and register it in `TOOL_EXECUTORS`.
2. Add the matching tool definition to `CHART_TOOLS` in `src/api/chart_tools.py`.
3. Add an executor test file under `tests/unit/`.
4. Add a frontend renderer at `src/app/report/[id]/charts/<NewKind>.tsx` and wire it into `ChartCard.tsx`.

## Logs

- Per-run logs under `src/api/logs/chartsage_run_<timestamp>_<runid>.log`.
- Last 50 runs kept; older auto-deleted.
- Trailer block at end of each log shows the run summary (model, tokens, charts, elapsed).
