# Analyses posted — running log

Our record of every event-data post produced with the `event-data-post` skill
(Step 9). **Skim this before pitching** so we don't repeat an angle, and **append a
row after each post**: date, hook, event/topic, dataset (+ source), report URL,
hashtag, status.

**Status:** `Posted` = live on X · `Scheduled`/`Drafted` = queued in Buffer (auto-posts / saved draft) · `Ready` = built, not yet queued · `Backlog` = angle found, not built · `Paused` = started, blocked.

| Date | Hook | Event / topic | Dataset (source) | Report | Hashtag | Status |
|---|---|---|---|---|---|---|
| 2026-06-04 | 3 hosts in 2026 — host nations win **61%** of WC games vs **43%** on neutral ground (964 matches since 1930) | FIFA World Cup 2026 | World Cup match results — host vs neutral win rate | _published (UTM campaign ~`worldcup`; link not captured)_ | `#WorldCup2026` | **Posted** |
| 2026-06-05 | "Pole = win"? Monaco pole-sitter wins just **46%** (≈ the ~44% avg); Barcelona **71%** is the real fortress; Rio 10% | Monaco GP | F1 pole→win % by circuit (Ergast/Jolpica, computed per circuit) | chartsage.app/report/2f2bf9e3a61d4f98892309da2fa72200 — UTM `f1-monaco`/`pole-fortress` | `#MonacoGP` | **Posted** |
| 2026-06-06 | Recorded major Atlantic hurricanes/yr by decade — 2020s lead every decade (~4.3/yr) | Atlantic hurricane season | NOAA HURDAT2 1851–2025 | _drafted; report hit the chart-fallback bug (now fixed) — not yet re-run/posted_ | `#HurricaneSeason` | Drafted (paused) |
| 2026-06-07 | The 1954 World Cup averaged **5.4 goals/game**; the last five ~2.5 — a 70-yr slide that flatlined in the '90s | FIFA World Cup 2026 | Fjelstul World Cup DB — goals/game per men's tournament, 1930–2022 | chartsage.app/report/0755fecf5ab2440c8d8f9ab840d658db — UTM `worldcup`/`goals-per-game` | `#WorldCup2026` | **Posted** (X, 2026-06-07 11:00; confirmed via Buffer sent history) |
| 2026-06-07 | The shootout hierarchy: Argentina **6–1** (most of anyone); Germany & Croatia perfect; England's curse = 1 in 4 | FIFA World Cup 2026 | Fjelstul World Cup DB — penalty-shootout W/L by nation, since 1982 | chartsage.app/report/03f80df2faf4402c9a24e182541dfb35 — UTM `worldcup`/`shootouts` | `#WorldCup2026` | **Scheduled** (Buffer; Mon 08 Jun 11:45) |
| 2026-06-07 | The WC rout is dying — 4+ goal blowouts fell from ~31% (1954) to ~5% today; avg margin 2.3 → 1.4. 48 teams in 2026 — comeback? | FIFA World Cup 2026 | Fjelstul World Cup DB — competitive balance per men's tournament (blowout/draw/one-goal rates, margin, goals, size), 1930–2022 (rich 8-col → 7-chart report) | chartsage.app/report/7c710bd8661c423c9d26e150062f50d2 — UTM `worldcup`/`blowouts` | `#WorldCup2026` | **Scheduled** (Buffer; 1st full pipeline run on the content account) |
| 2026-06-07 | The WC got mean, then mellowed — cards/game tripled from 1.6 (1970) to a **5.2 peak in 2006**, cooled to ~3.5; reds 0 → 0.44 → 0.06 | FIFA World Cup 2026 | Fjelstul World Cup DB — **discipline by men's tournament** (cards/yellows/reds/goals per game + size), 1970–2022 (rich 7-col table → 10-chart report) | chartsage.app/report/3f9e7154b6734c1fa5221eacd9f29445 — UTM `worldcup`/`cards` | `#WorldCup2026` | **Scheduled** |

| 2026-06-07 | Finishing used to be the hard part — ~half the F1 grid DNF'd for decades (peak **57% in 1966**); now ~1 in 10. Verstappen's lap-1 exit a throwback | Monaco GP 2026 (Verstappen DNF) | **Ergast 1950–2021 + live Jolpica 2022–2025** — DNF/reliability per season (rich 6-col → 10-chart report) | chartsage.app/report/b3cef7aeae2642eabb5b8a61ea599680 — UTM `f1-monaco`/`dnf-history` | `#MonacoGP` | **Posted** (1st full QA→content pipeline run) |

<!-- Append new posts above this line. Keep newest at the bottom of the table. -->

## Banked angles (found via EDA, not yet posted)

Surfaced while building a post but not used — future candidates. Skim before pitching.

- **Draws have ~tripled** — ~9% of men's WC matches were draws in the early tournaments vs ~23% now (peak 33% in 1982). The game got cagier (pairs with the goals decline). _Fjelstul matches._
- ~~Bookings per game ~doubled~~ → **built 2026-06-07 (Ready, see table)** — cards/game 1970–2022 (tripled to a 2006 peak, then cooled).
- ~~The shootout era~~ → **built 2026-06-07 (Ready, see table)** — shootout win rate by nation.
- ~~Biggest hammerings~~ → **built 2026-06-07 (In progress, see table)** — blowout-rate decline + 48-team angle.
- **Penalties ≈ 8% of all WC goals; own goals ≈ 2%.** _Fjelstul goals._
