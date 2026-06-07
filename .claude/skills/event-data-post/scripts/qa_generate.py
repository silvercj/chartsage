#!/usr/bin/env python3
"""Generate a THROWAWAY QA report via the ChartSage API and dump a chart QA, so we
verify a report's charts *before* the user publishes the real one on the brand account.
Runs under the QA anon id — never published, never on the brand account.

    qa_generate.py <csv-path> "<custom prompt>"
    qa_generate.py <csv-path> --prompt-file <path>

Config (env, with sane defaults):
    CHARTSAGE_API_URL     backend base URL (default: prod Cloud Run)
    CHARTSAGE_QA_ANON_ID  QA anon id (default: ~/.chartsage/qa-anon-id, else a random
                          one that's capped at 1 report). Add this id to the backend
                          UNLIMITED_ANON_IDS allowlist to lift the free-report cap.
"""
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

API = os.environ.get(
    "CHARTSAGE_API_URL",
    "https://chartsage-backend-112026133429.us-central1.run.app",
).rstrip("/")

# A scatter whose x is one of these is a time series drawn as a scatter — usually wants a
# line/bar (a recurring Haiku pick; see docs/report-ux-checklist.md).
_TEMPORAL = {"year", "years", "date", "decade", "season", "period", "quarter", "month", "week"}


def qa_anon_id() -> str:
    v = os.environ.get("CHARTSAGE_QA_ANON_ID")
    if v:
        return v.strip()
    f = Path.home() / ".chartsage" / "qa-anon-id"
    if f.is_file():
        return f.read_text().strip()
    rid = str(uuid.uuid4())
    print(f"! no QA anon id found; using a fresh random id {rid} (capped at 1 report). "
          f"Put a fixed id in ~/.chartsage/qa-anon-id (and the backend allowlist) to uncap.",
          file=sys.stderr)
    return rid


def _multipart(fields: dict, files: dict):
    boundary = "----chartsageqa" + uuid.uuid4().hex
    parts = []
    for name, val in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{val}\r\n'.encode())
    for name, path in files.items():
        fn = os.path.basename(path)
        ctype = mimetypes.guess_type(fn)[0] or "application/octet-stream"
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{fn}"\r\n'
                     f'Content-Type: {ctype}\r\n\r\n'.encode())
        parts.append(Path(path).read_bytes())
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


def _req(method: str, path: str, anon: str, data=None, ctype=None):
    req = urllib.request.Request(f"{API}{path}", data=data, method=method)
    req.add_header("X-Anon-Id", anon)
    if ctype:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=240) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} on {method} {path}: {e.read().decode()[:400]}")


def qa_dump(rep: dict) -> list:
    charts = rep.get("charts") or []
    print(f"\nsummary: {(rep.get('summary') or '')[:140]}")
    km = rep.get("key_metrics") or []
    print("key_metrics:", [(m.get("label"), m.get("value")) for m in km])
    print(f"\n{len(charts)} charts:")
    issues, advisories = [], []
    for i, c in enumerate(charts):
        s = c.get("spec", c)
        fb = "FB" if "fallback" in (s.get("intent") or "").lower() else "model"
        series = s.get("series")
        if series:
            pts = sum(len(ser.get("y", ser.get("data", [])) or []) for ser in series)
            data, empty = f"series({len(series)}ser,{pts}pt)", pts == 0
        else:
            ys = s.get("y") or []
            data, empty = f"x={len(s.get('x') or [])},y={len(ys)}", not ys
        src = s.get("source_columns") or []
        flags, notes = [], []
        if empty:
            flags.append("EMPTY")
        if len(src) != len(set(src)):
            flags.append("SELF-GROUP")
        if s.get("kind") == "scatter" and src:
            xcol = "".join(ch for ch in str(src[0]).lower() if ch.isalpha())
            if xcol in _TEMPORAL:
                notes.append("TIME-SCATTER(line?)")
        if flags:
            issues.append(f"[{i}] {s.get('title')!r}: {','.join(flags)}")
        if notes:
            advisories.append(f"[{i}] {s.get('title')!r}: {','.join(notes)}")
        tail = f"  <!! {' '.join(flags + notes)}>" if (flags or notes) else ""
        print(f"  [{i}] {s.get('kind'):10} {fb:5} {s.get('title')!r}")
        print(f"        {data} | ydisp={s.get('y_display_type')} | src={src}{tail}")
    print("\nQA:", "CLEAN — no empty/self-group charts." if not issues
          else "ISSUES -> " + " | ".join(issues))
    if advisories:
        print("advisories (eyeball in the render):", " | ".join(advisories))
    return issues


def main():
    if len(sys.argv) < 3:
        sys.exit('usage: qa_generate.py <csv> "<prompt>"  |  <csv> --prompt-file <path>')
    csv = sys.argv[1]
    prompt = Path(sys.argv[3]).read_text() if sys.argv[2] == "--prompt-file" else sys.argv[2]
    anon = qa_anon_id()

    body, boundary = _multipart({"custom_prompt": prompt}, {"file": csv})
    res = _req("POST", "/generate-report", anon, body, f"multipart/form-data; boundary={boundary}")
    rid = res.get("session_id") or res.get("id")
    print(f"session_id: {rid}")

    qa_dump(_req("GET", f"/report/{rid}", anon))
    here = os.path.dirname(os.path.abspath(__file__))
    print(f"\nrender it (as owner, no publish):\n  ~/.venvs/chartsage/bin/python {here}/qa_render.py "
          f"https://chartsage.app/report/{rid}")


if __name__ == "__main__":
    main()
