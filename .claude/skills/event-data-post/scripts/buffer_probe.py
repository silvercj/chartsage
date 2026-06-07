#!/usr/bin/env python3
"""Throwaway: introspect Buffer's GraphQL schema to learn real root fields + key types."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from buffer_api import gql  # noqa: E402

ROOTS = """
query {
  __schema {
    queryType { fields { name args { name } } }
    mutationType { fields { name args { name } } }
  }
}
"""

d = gql(ROOTS)
schema = d.get("__schema") or {}
print("=== ROOT QUERY FIELDS ===")
for f in (schema.get("queryType") or {}).get("fields", []) or []:
    args = ", ".join(a["name"] for a in f.get("args", []))
    print(f"  {f['name']}({args})")
print("=== ROOT MUTATION FIELDS ===")
for f in (schema.get("mutationType") or {}).get("fields", []) or []:
    args = ", ".join(a["name"] for a in f.get("args", []))
    print(f"  {f['name']}({args})")


def dump_type(name: str):
    q = """
    query($n: String!) {
      __type(name: $n) {
        name kind
        fields { name type { name kind ofType { name kind ofType { name kind } } } }
        inputFields { name type { name kind ofType { name kind ofType { name kind } } } }
        enumValues { name }
      }
    }
    """
    t = gql(q, {"n": name}).get("__type")
    if not t:
        print(f"\n=== TYPE {name}: (not found) ===")
        return

    def tn(ty):
        if not ty:
            return "?"
        return ty.get("name") or tn(ty.get("ofType"))

    print(f"\n=== TYPE {name} ({t.get('kind')}) ===")
    for f in t.get("fields") or []:
        print(f"  .{f['name']}: {tn(f['type'])}")
    for f in t.get("inputFields") or []:
        print(f"  input {f['name']}: {tn(f['type'])}")
    for e in t.get("enumValues") or []:
        print(f"  enum {e['name']}")


for name in sys.argv[1:]:
    dump_type(name)
