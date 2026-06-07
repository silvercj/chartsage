#!/usr/bin/env python3
"""Create / delete a Buffer post-thread on the @chartsageapp X channel (the createPost half).

The event-data-post threads map cleanly onto one createPost call: main tweet + image,
then reply 1 (the report link) + reply 2 as a Twitter thread.

    # scheduled DRAFT for your one-click approval (the manager's default hand-off):
    buffer_schedule.py --at 2026-06-10T15:00:00Z --draft \
        --text "MAIN hook ... #WorldCup2026" \
        --image https://host/hero.png \
        --reply "Full breakdown: chartsage.app/report/...?utm_..." \
        --reply "Bonus stat ..."

    # straight into the scheduled queue (skip the draft/approval stage):
    buffer_schedule.py --at 2026-06-10T15:00:00Z --text "..." ...

    # next free slot instead of an explicit time:
    buffer_schedule.py --queue --draft --text "..." ...

    # inspect the exact mutation input without sending:
    buffer_schedule.py --dry-run --at ... --text "..." ...

    # remove a post by id:
    buffer_schedule.py --delete <postId>

Channel auto-resolves to the single connected twitter channel (override with --channel).
Image MUST be a PUBLIC url — Buffer's GraphQL API does not accept file uploads yet.
dueAt is ISO-8601 UTC, e.g. 2026-06-10T15:00:00Z.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buffer_api import gql  # noqa: E402

CREATE = """
mutation($input: CreatePostInput!) {
  createPost(input: $input) {
    __typename
    ... on PostActionSuccess { post {
      id status dueAt channelId text
      metadata { ... on TwitterPostMetadata { thread { text } } }
    } }
    ... on MutationError { message }
  }
}
"""

DELETE = """
mutation($input: DeletePostInput!) {
  deletePost(input: $input) {
    __typename
    ... on DeletePostSuccess { id }
    ... on VoidMutationError { message }
  }
}
"""


def resolve_channel(want=None):
    acct = gql("query { account { organizations { id } } }").get("account") or {}
    chans = []
    for o in acct.get("organizations") or []:
        cs = gql(
            "query($o: OrganizationId!) { channels(input: {organizationId: $o}) "
            "{ id service displayName isDisconnected } }",
            {"o": o["id"]},
        ).get("channels") or []
        chans += cs
    if want:
        if any(c["id"] == want for c in chans):
            return want
        sys.exit(f"channel {want} not found on this account")
    tw = [c for c in chans if c.get("service") == "twitter" and not c.get("isDisconnected")]
    if len(tw) == 1:
        return tw[0]["id"]
    if not tw:
        sys.exit("no connected twitter channel on this account")
    sys.exit("multiple twitter channels; pass --channel <id>: "
             + ", ".join(f"{c['displayName']}={c['id']}" for c in tw))


def build_input(args, channel_id):
    inp = {"text": args.text, "channelId": channel_id}
    if args.image:
        inp["assets"] = [{"image": {"url": args.image}}]
    if args.reply:
        inp["metadata"] = {"twitter": {"thread": [{"text": r, "assets": []} for r in args.reply]}}
    if args.at:
        inp["schedulingType"] = "automatic"
        inp["mode"] = "customScheduled"
        inp["dueAt"] = args.at
    elif args.queue:
        inp["schedulingType"] = "automatic"
        inp["mode"] = "addToQueue"
    if args.draft:
        inp["saveToDraft"] = True
    return inp


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", help="main tweet text")
    p.add_argument("--reply", action="append", help="a thread reply (repeatable, in order)")
    p.add_argument("--image", help="PUBLIC image url for the main tweet")
    p.add_argument("--at", help="schedule time, ISO-8601 UTC (e.g. 2026-06-10T15:00:00Z)")
    p.add_argument("--queue", action="store_true", help="add to next free slot instead of --at")
    p.add_argument("--draft", action="store_true", help="saveToDraft: lands for approval, keeps --at time")
    p.add_argument("--channel", help="channel id (default: the single twitter channel)")
    p.add_argument("--delete", metavar="POST_ID", help="delete a post by id and exit")
    p.add_argument("--dry-run", action="store_true", help="print the mutation input, don't send")
    args = p.parse_args()

    if args.delete:
        res = gql(DELETE, {"input": {"id": args.delete}}, raise_on_error=True).get("deletePost") or {}
        print(json.dumps(res, indent=2))
        return

    if not args.text:
        p.error("--text is required (unless --delete)")
    if not (args.at or args.queue or args.draft):
        p.error("specify one of --at <time>, --queue, or --draft (refusing to share immediately)")

    channel_id = resolve_channel(args.channel)
    inp = build_input(args, channel_id)
    if args.dry_run:
        print("DRY RUN — createPost input:")
        print(json.dumps(inp, indent=2))
        return
    res = gql(CREATE, {"input": inp}, raise_on_error=True).get("createPost") or {}
    if res.get("__typename") == "MutationError":
        sys.exit(f"createPost failed: {res.get('message')}")
    post = res.get("post") or {}
    print(json.dumps(post, indent=2))


if __name__ == "__main__":
    main()
