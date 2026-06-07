#!/usr/bin/env python3
"""List the Buffer account, channels, and upcoming post queue (READ-ONLY).

Proves API access and shows the scheduling picture the manager needs for gap-filling:
what's already queued (per channel, with times), and the org's scheduling limits.

    buffer_list.py            # everything: orgs, channels, upcoming posts, limits
    buffer_list.py --json     # raw JSON (for the manager to consume)
"""
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buffer_api import gql  # noqa: E402

ACCOUNT = """
query {
  account {
    id email timezone
    organizations { id name channelCount
      limits { scheduledPosts scheduledThreadsPerChannel } }
  }
}
"""

CHANNELS = """
query($org: OrganizationId!) {
  channels(input: { organizationId: $org }) {
    id displayName name service type isDisconnected isLocked isQueuePaused timezone
  }
}
"""

POSTS = """
query($org: OrganizationId!, $channels: [ChannelId!]) {
  posts(first: 100, input: {
    organizationId: $org,
    filter: { status: [scheduled, needs_approval, draft, sending], channelIds: $channels },
    sort: { field: dueAt, direction: asc }
  }) {
    edges { node {
      id text status dueAt channelId channelService isCustomScheduled
      externalLink metadata { ... on TwitterPostMetadata { thread { text } } }
    } }
    pageInfo { hasNextPage endCursor }
  }
}
"""


def parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def fmt(dt):
    if not dt:
        return "(no time set)"
    return dt.astimezone().strftime("%a %d %b %Y %H:%M %Z")


def collect():
    acct = (gql(ACCOUNT).get("account")) or {}
    orgs = acct.get("organizations") or []
    out = {"account": {k: acct.get(k) for k in ("id", "email", "timezone")}, "organizations": []}
    for org in orgs:
        oid = org["id"]
        channels = gql(CHANNELS, {"org": oid}).get("channels") or []
        cids = [c["id"] for c in channels]
        edges = (gql(POSTS, {"org": oid, "channels": cids}).get("posts") or {}).get("edges") or []
        out["organizations"].append({
            "id": oid, "name": org.get("name"), "channelCount": org.get("channelCount"),
            "limits": org.get("limits") or {},
            "channels": channels,
            "posts": [e["node"] for e in edges],
        })
    return out


def main():
    data = collect()
    if "--json" in sys.argv[1:]:
        print(json.dumps(data, indent=2))
        return
    acct = data["account"]
    print(f"account: {acct.get('email')}   tz={acct.get('timezone')}   orgs={len(data['organizations'])}")
    for org in data["organizations"]:
        lim = org["limits"]
        print(f"\n# {org['name']}  ({org['id']})   channels={org['channelCount']}")
        print(f"  limits: scheduledPosts={lim.get('scheduledPosts')}  "
              f"scheduledThreadsPerChannel={lim.get('scheduledThreadsPerChannel')}")
        chmap = {c["id"]: c for c in org["channels"]}
        for c in org["channels"]:
            flags = " ".join(f for f, on in (
                ("DISCONNECTED", c.get("isDisconnected")), ("LOCKED", c.get("isLocked")),
                ("PAUSED", c.get("isQueuePaused"))) if on)
            print(f"  - {c.get('service'):12} {c.get('displayName') or c.get('name')}  ({c['id']}) {flags}")
        posts = org["posts"]
        print(f"  upcoming posts: {len(posts)}")
        # per-channel scheduled count (for gap awareness vs limits)
        by_ch = {}
        for n in posts:
            by_ch.setdefault(n.get("channelId"), []).append(n)
        for cid, ns in by_ch.items():
            c = chmap.get(cid, {})
            print(f"    {c.get('service','?')} {c.get('displayName') or c.get('name') or cid}: {len(ns)} queued")
        for n in posts:
            thread = (((n.get("metadata") or {}).get("thread")) or [])
            tn = f"  +{len(thread)} replies" if thread else ""
            text = (n.get("text") or "").replace("\n", " ")[:64]
            print(f"      [{n.get('status'):14}] {fmt(parse_dt(n.get('dueAt')))}  "
                  f"{n.get('channelService','')}  {text}{tn}")


if __name__ == "__main__":
    main()
