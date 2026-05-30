#!/usr/bin/env python3
"""Query PostHog analytics via HogQL.

Reads POSTHOG_PERSONAL_API_KEY and POSTHOG_PROJECT_ID from the environment
(see .env). Usage:

    python scripts/ph_query.py "select event, count() from events group by event order by 2 desc"

With no argument, runs a default "events in the last 7 days" summary.
The personal API key is secret — never commit it; it lives only in .env.
"""
import json
import os
import sys
import urllib.request
import urllib.error

HOST = os.environ.get("POSTHOG_QUERY_HOST", "https://us.posthog.com")
KEY = os.environ.get("POSTHOG_PERSONAL_API_KEY")
PROJECT = os.environ.get("POSTHOG_PROJECT_ID")

DEFAULT_QUERY = (
    "select event, count() as n, max(timestamp) as last_seen "
    "from events where timestamp > now() - interval 7 day "
    "group by event order by n desc"
)


def run_hogql(query: str) -> dict:
    if not KEY or not PROJECT:
        sys.exit("POSTHOG_PERSONAL_API_KEY and POSTHOG_PROJECT_ID must be set (source .env)")
    url = f"{HOST}/api/projects/{PROJECT}/query/"
    payload = json.dumps({"query": {"kind": "HogQLQuery", "query": query}}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        sys.exit(f"HTTP {e.code}: {body}")


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    result = run_hogql(query)
    cols = result.get("columns", [])
    rows = result.get("results", [])
    if cols:
        print(" | ".join(str(c) for c in cols))
        print("-" * 60)
    for row in rows:
        print(" | ".join(str(v) for v in row))
    if not rows:
        print("(no rows)")


if __name__ == "__main__":
    main()
