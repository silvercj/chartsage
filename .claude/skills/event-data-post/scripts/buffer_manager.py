#!/usr/bin/env python3
"""Schedule manager (READ-ONLY): reconcile the Buffer queue against our cadence policy
and surface the GAPS to fill.

Division of labour: this script does the deterministic schedule math — pull the live
queue, apply the cadence (spacing + horizon), flag empty slots, report limits. The
*agent* then matches each gap to a backlog post (analyses-log.md: Ready / Backlog /
banked angles), tunes the time per content (event-anchored — near a live race/match so
it rides the trending hashtag; else the default US window), proposes the plan for the
user's OK, and schedules with buffer_schedule.py (--draft, for one-click approval).

Cadence policy (defaults; override with flags):
  - every-other-day (--spacing 2.0) until we hold a ~2-week runway, then tighten to
    daily (--spacing 1.0); reserve 2/day for event spikes (pass two posts on a day).
  - default suggested slot time --slot 15:00 UTC (~US late-morning); agent tunes per post.

    buffer_manager.py                 # calendar + gaps over the next 14 days
    buffer_manager.py --days 21 --spacing 1.0
    buffer_manager.py --json          # machine-readable (for the agent)
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buffer_list import collect, parse_dt  # noqa: E402


def fmt(dt, tz_local=True):
    return (dt.astimezone() if tz_local else dt).strftime("%a %d %b %H:%M %Z")


def label(node):
    return (node.get("text") or "").replace("\n", " ").strip()[:48]


def analyse(data, now, days, spacing_days, slot_hhmm):
    horizon = now + timedelta(days=days)
    spacing = timedelta(days=spacing_days)
    hh, mm = map(int, slot_hhmm.split(":"))

    posts, total, cap_posts, cap_threads = [], 0, None, None
    for org in data["organizations"]:
        lim = org.get("limits") or {}
        cap_posts, cap_threads = lim.get("scheduledPosts"), lim.get("scheduledThreadsPerChannel")
        for n in org["posts"]:
            total += 1
            dt = parse_dt(n.get("dueAt"))
            if dt:
                posts.append((dt, n))
    posts.sort(key=lambda x: x[0])
    upcoming = [(dt, n) for dt, n in posts if now - timedelta(hours=2) <= dt <= horizon]
    post_times = [dt for dt, _ in upcoming]
    last = upcoming[-1][0] if upcoming else now
    runway_days = max(0.0, (last - now).total_seconds() / 86400)

    # desired grid: slot-time today, then every `spacing`, through horizon; tol = spacing/2
    # tiles the timeline with no overlap/dead-zone, so each post fills exactly one slot.
    desired, c = [], now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    while c < now:
        c += spacing
    while c <= horizon:
        desired.append(c)
        c += spacing
    tol = spacing.total_seconds() / 2
    remaining, gaps = list(post_times), []
    for d in desired:
        hit = next((t for t in remaining if abs((t - d).total_seconds()) <= tol), None)
        if hit:
            remaining.remove(hit)
        else:
            gaps.append(d)

    return {
        "now": now, "horizon": horizon, "spacing_days": spacing_days,
        "upcoming": upcoming, "runway_days": runway_days,
        "gaps": gaps, "total_scheduled": total,
        "cap_posts": cap_posts, "cap_threads": cap_threads,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=14, help="horizon (default 14)")
    ap.add_argument("--spacing", type=float, default=2.0, help="target days between posts (2=every other day)")
    ap.add_argument("--slot", default="15:00", help="default UTC HH:MM for suggested gap slots")
    ap.add_argument("--ramp", type=int, default=14, help="runway (days) at which to recommend tightening cadence")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    a = analyse(collect(), now, args.days, args.spacing, args.slot)

    if args.json:
        out = dict(a)
        out["now"] = a["now"].isoformat()
        out["horizon"] = a["horizon"].isoformat()
        out["upcoming"] = [{"dueAt": dt.isoformat(), "status": n.get("status"), "text": label(n)}
                           for dt, n in a["upcoming"]]
        out["gaps"] = [g.isoformat() for g in a["gaps"]]
        print(json.dumps(out, indent=2))
        return

    print(f"SCHEDULE MANAGER — @chartsageapp    now: {fmt(now)}")
    print(f"cadence: a post every {args.spacing:g} days    horizon: {args.days} days "
          f"(through {fmt(a['horizon'])})")
    print(f"limits: {a['total_scheduled']}/{a['cap_posts']} scheduled posts · "
          f"threads/channel cap {a['cap_threads']} · X daily cap: wireable (dailyPostingLimits)")

    print(f"\nupcoming ({len(a['upcoming'])}):")
    for dt, n in a["upcoming"]:
        print(f"  {fmt(dt)}   [{n.get('status')}]  {label(n)}")
    print(f"runway: {a['runway_days']:.1f} days" + (f"  (last post {fmt(a['upcoming'][-1][0])})" if a["upcoming"] else ""))

    print(f"\nGAPS to fill ({len(a['gaps'])}) — agent: match each to a Ready/Backlog post in analyses-log.md:")
    for g in a["gaps"]:
        print(f"  {fmt(g)}   (default slot — tune to any live event hashtag that day)")

    print("\nrecommendation:", end=" ")
    if a["runway_days"] >= args.ramp:
        print(f"runway {a['runway_days']:.0f}d >= {args.ramp}d — consider tightening to daily (--spacing 1.0); "
              "reserve 2/day for event spikes.")
    else:
        print(f"runway {a['runway_days']:.0f}d < {args.ramp}d target — hold every-other-day; fill the "
              f"{len(a['gaps'])} gaps above from the backlog. Tighten to daily once the runway reaches {args.ramp}d.")


if __name__ == "__main__":
    main()
