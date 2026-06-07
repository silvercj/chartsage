# Analyses posted — running log

Our record of every event-data post produced with the `event-data-post` skill
(Step 9). **Skim this before pitching** so we don't repeat an angle, and **append a
row after each post**: date, hook, event/topic, dataset (+ source), report URL,
hashtag, status.

| Date | Hook | Event / topic | Dataset (source) | Report | Hashtag | Status |
|---|---|---|---|---|---|---|
| 2026-06-04 | 3 hosts in 2026 — host nations win **61%** of WC games vs **43%** on neutral ground (964 matches since 1930) | FIFA World Cup 2026 | World Cup match results — host vs neutral win rate | _published (UTM campaign ~`worldcup`; link not captured)_ | `#WorldCup2026` | **Posted** |
| 2026-06-05 | "Pole = win"? Monaco pole-sitter wins just **46%** (≈ the ~44% avg); Barcelona **71%** is the real fortress; Rio 10% | Monaco GP | F1 pole→win % by circuit (Ergast/Jolpica, computed per circuit) | chartsage.app/report/2f2bf9e3a61d4f98892309da2fa72200 — UTM `f1-monaco`/`pole-fortress` | `#MonacoGP` | **Posted** |
| 2026-06-06 | Recorded major Atlantic hurricanes/yr by decade — 2020s lead every decade (~4.3/yr) | Atlantic hurricane season | NOAA HURDAT2 1851–2025 | _drafted; report hit the chart-fallback bug (now fixed) — not yet re-run/posted_ | `#HurricaneSeason` | Drafted (paused) |
| 2026-06-07 | The 1954 World Cup averaged **5.4 goals/game**; the last five ~2.5 — a 70-yr slide that flatlined in the '90s | FIFA World Cup 2026 | Fjelstul World Cup DB — goals/game per men's tournament, 1930–2022 | chartsage.app/report/0755fecf5ab2440c8d8f9ab840d658db — UTM `worldcup`/`goals-per-game` | `#WorldCup2026` | Posted |
| 2026-06-07 | The shootout hierarchy: Argentina **6–1** (most of anyone); Germany & Croatia perfect; England's curse = 1 in 4 | FIFA World Cup 2026 | Fjelstul World Cup DB — penalty-shootout W/L by nation, since 1982 | chartsage.app/report/878a439141bf425eadde0dc87e0f4fc5 — UTM `worldcup`/`shootouts` | `#WorldCup2026` | Posted |

<!-- Append new posts above this line. Keep newest at the bottom of the table. -->

## Banked angles (found via EDA, not yet posted)

Surfaced while building a post but not used — future candidates. Skim before pitching.

- **Draws have ~tripled** — ~9% of men's WC matches were draws in the early tournaments vs ~23% now (peak 33% in 1982). The game got cagier (pairs with the goals decline). _Fjelstul matches._
- **Bookings per game ~doubled** — ~2.0 → ~3.9 across the modern era. _Fjelstul bookings; cards only since 1970, so treat pre-1970 with care._
- ~~The shootout era~~ → **posted 2026-06-07** (shootout win rate by nation).
- **Biggest hammerings** — Hungary 9–0 (1954) & 10–1 (1982), several 8–0s; with 48 teams = more mismatches in 2026, more blowouts coming? _Fjelstul matches._
- **Penalties ≈ 8% of all WC goals; own goals ≈ 2%.** _Fjelstul goals._
