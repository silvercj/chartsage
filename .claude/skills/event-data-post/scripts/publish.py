#!/usr/bin/env python3
"""Publish (or unpublish) a ChartSage report you own, by its id — for the event-data flow:
stage a QA-account report so the user can review the whole thing in-browser (Gate 1), or
publish the content-account report for the post (step 4). Publishing resolves the share link
+ generates the OG image.

    publish.py <report-id>                 # publish
    publish.py <report-id> --unpublish     # take it back down

Owner anon id: CHARTSAGE_ANON_ID env, else CHARTSAGE_QA_ANON_ID, else ~/.chartsage/qa-anon-id.
For the content account:  CHARTSAGE_ANON_ID=$(cat ~/.chartsage/content-id) publish.py <id>
"""
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = os.environ.get(
    "CHARTSAGE_API_URL",
    "https://chartsage-backend-112026133429.us-central1.run.app",
).rstrip("/")


def anon_id() -> str:
    v = os.environ.get("CHARTSAGE_ANON_ID") or os.environ.get("CHARTSAGE_QA_ANON_ID")
    if v:
        return v.strip()
    f = Path.home() / ".chartsage" / "qa-anon-id"
    if f.is_file():
        return f.read_text().strip()
    sys.exit("no anon id (set CHARTSAGE_ANON_ID or put one in ~/.chartsage/qa-anon-id)")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: publish.py <report-id> [--unpublish]")
    rid = sys.argv[1]
    action = "unpublish" if "--unpublish" in sys.argv[2:] else "publish"
    req = urllib.request.Request(f"{API}/report/{rid}/{action}", data=b"", method="POST")
    req.add_header("X-Anon-Id", anon_id())
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read().decode()
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} on {action} {rid}: {e.read().decode()[:300]}")
    print(f"{action}ed: https://chartsage.app/report/{rid}")
    if body.strip() and body.strip() not in ("{}", "null"):
        print(" ", body[:200])


if __name__ == "__main__":
    main()
