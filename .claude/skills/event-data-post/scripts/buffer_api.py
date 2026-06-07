#!/usr/bin/env python3
"""Shared Buffer GraphQL helper for the event-data-post scheduling scripts.

Buffer's new GraphQL API (https://api.buffer.com) authenticated with a *personal*
API key (Settings -> API at publish.buffer.com/settings/api). The key acts on your
whole account (all organizations + channels). Token resolves from BUFFER_TOKEN env,
else ~/.chartsage/buffer-token.

Usage:
    from buffer_api import gql
    data = gql("query { ... }", {"var": 1})
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = os.environ.get("BUFFER_API_URL", "https://api.buffer.com").rstrip("/")


def token() -> str:
    v = os.environ.get("BUFFER_TOKEN")
    if v:
        return v.strip()
    f = Path.home() / ".chartsage" / "buffer-token"
    if f.is_file():
        return f.read_text().strip()
    sys.exit("no Buffer token (set BUFFER_TOKEN or put one in ~/.chartsage/buffer-token)")


def gql(query: str, variables: dict | None = None, *, raise_on_error: bool = False) -> dict:
    """POST a GraphQL query/mutation. Returns the `data` object.

    GraphQL-level errors are printed to stderr; pass raise_on_error=True to abort.
    """
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(API, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {token()}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} from Buffer API: {e.read().decode()[:600]}")
    except urllib.error.URLError as e:
        sys.exit(f"network error reaching {API}: {e}")
    if payload.get("errors"):
        msg = json.dumps(payload["errors"], indent=2)[:1200]
        if raise_on_error:
            sys.exit(f"GraphQL errors:\n{msg}")
        sys.stderr.write(f"GraphQL errors:\n{msg}\n")
    return payload.get("data") or {}
