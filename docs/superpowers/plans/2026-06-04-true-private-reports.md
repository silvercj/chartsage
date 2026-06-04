# True-Private Reports — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce report visibility — reports are private by default and viewable/exportable only by their owner (authed `user_id` OR anon `anon_id`); explicit publish makes them public + indexed. Fixes the "made private but still viewable in incognito" bug.

**Architecture:** No schema change — reuse the existing `is_public` boolean (false = private, true = public). Add a single fail-closed access helper (`_resolve_report_access`) and route every report read/export/manage endpoint through it. A signed, short-lived **render token** lets the owner-authorized PDF/PPTX export (server-side Playwright, which carries no user auth) render a private report without exposing it. Frontend shows a "private" state on 404.

**Tech Stack:** FastAPI (Cloud Run), Next.js 14 (Vercel), Supabase, pytest + FakeDB, Playwright.

**Branch:** `true-private-reports` (already created off `origin/main`). venv: `venv/bin/python`. Deploy needs `CLOUDSDK_PYTHON=/opt/homebrew/opt/python@3.12/bin/python3.12`.

**Key files (current state):**
- `src/api/main.py`: `_require_report_owner@202` (authed-only — REPLACE), `_public_urls@213`, `_report_title_desc@225`; `GET /report@458` (no identity dep), `publish@469`/`unpublish@490` (get_identity + _require_report_owner), `report_meta@503` (get_identity_optional); mutations `patch_report_layout@527`, `generate_more@632`, `add_chart@763`, `deepen_report@892` (get_identity, NO ownership check today); exports `export_pdf@1063`/`pptx@1114`/`xlsx@1142`/`zip@1183`/`md@1208`/`html@1233` (get_identity, NO visibility check today).
- `src/api/deps.py`: `Identity` (has `.user_id`, `.anon_id`, `.is_authenticated`), `get_identity`, `get_identity_optional`.
- `src/api/db.py` + `tests/helpers/fake_db.py`: `get_report`, `set_report_visibility`, `save_report(report_id, anon_id, user_id, ...)`, rows carry `is_public`/`user_id`/`anon_id`.
- `tests/helpers/fake_auth.py`: `auth_identity(user_id)`, `anon_identity(anon_id)`.
- `src/api/pdf_export.py`: `render_report_pdf(session_id)`, `render_og_image(session_id)`, `render_chart_images(session_id)` — all `page.goto(f"{_FRONTEND_BASE}/report/{id}/print|og")`.
- Frontend: `src/app/report/[id]/ReportClient.tsx:58` (apiFetch GET /report; 404 handling @60), `print/page.tsx:34` (plain `fetch` of GET /report), `Toolbar.tsx:34` (apiFetch /meta → `owned`/`is_public`), `ShareModal.tsx`.

**Test command:** `venv/bin/python -m pytest tests/integration/test_report_privacy.py -v` (per task). NOTE: importing `main` is slow on this machine (iCloud-cold stripe import) — allow generous time; do not treat a slow first import as a hang.

---

## Task 1: Access-control helpers + enforce `GET /report`

**Files:**
- Modify: `src/api/main.py` (replace `_require_report_owner@202`; change `get_report@458`)
- Test: `tests/integration/test_report_privacy.py` (new)

- [ ] **Step 1: Write the failing test**

```python
"""True-private reports: visibility + ownership enforcement."""
import uuid
import pytest
from fastapi.testclient import TestClient

import main
from deps import get_db, get_identity, get_identity_optional
from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_auth import auth_identity, anon_identity

OWNER = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"
ANON  = "33333333-3333-3333-3333-333333333333"
ANON2 = "44444444-4444-4444-4444-444444444444"

def _client(db, identity=None):
    app = main.app
    app.dependency_overrides[get_db] = lambda: db
    if identity is not None:
        app.dependency_overrides[get_identity] = lambda: identity
        app.dependency_overrides[get_identity_optional] = lambda: identity
    else:
        app.dependency_overrides.pop(get_identity, None)
        app.dependency_overrides[get_identity_optional] = lambda: __import__("deps").Identity()
    return TestClient(app)

def _seed(db, *, user_id=None, anon_id=None, is_public=False):
    rid = uuid.uuid4().hex
    db.save_report(rid, uuid.UUID(anon_id) if anon_id else None,
                   uuid.UUID(user_id) if user_id else None,
                   {"summary": "s", "charts": [], "layout": [], "data_quality": [],
                    "key_metrics": [], "metadata": {}, "generated_at": "x"},
                   None, "Secret title")
    if is_public:
        db.set_report_visibility(rid, True)
    return rid

@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    main.app.dependency_overrides.clear()

def test_owner_views_private_report():
    db = FakeDB(); rid = _seed(db, user_id=OWNER)
    r = _client(db, auth_identity(OWNER)).get(f"/report/{rid}")
    assert r.status_code == 200 and r.json()["summary"] == "s"

def test_non_owner_private_report_404():
    db = FakeDB(); rid = _seed(db, user_id=OWNER)
    r = _client(db, auth_identity(OTHER)).get(f"/report/{rid}")
    assert r.status_code == 404

def test_anon_owner_views_own_private_report():
    db = FakeDB(); rid = _seed(db, anon_id=ANON)
    r = _client(db, anon_identity(ANON)).get(f"/report/{rid}")
    assert r.status_code == 200

def test_different_anon_cannot_view_private_report():
    db = FakeDB(); rid = _seed(db, anon_id=ANON)
    r = _client(db, anon_identity(ANON2)).get(f"/report/{rid}")
    assert r.status_code == 404

def test_public_report_viewable_by_anyone():
    db = FakeDB(); rid = _seed(db, user_id=OWNER, is_public=True)
    r = _client(db, None).get(f"/report/{rid}")   # no identity at all
    assert r.status_code == 200

def test_missing_report_404():
    db = FakeDB()
    r = _client(db, auth_identity(OWNER)).get(f"/report/{uuid.uuid4().hex}")
    assert r.status_code == 404
```

- [ ] **Step 2: Run — expect FAIL** (`test_non_owner_private_report_404` etc. fail because GET /report has no check)

`venv/bin/python -m pytest tests/integration/test_report_privacy.py -v`

- [ ] **Step 3: Implement** — replace `_require_report_owner` (main.py ~202) with:

```python
def _is_owner(row: dict, identity: Identity) -> bool:
    if identity.user_id and row.get("user_id") == str(identity.user_id):
        return True
    if identity.anon_id and row.get("anon_id") == str(identity.anon_id):
        return True
    return False


def _resolve_report_access(db, identity: Identity, session_id: str, *, require_owner: bool = False):
    """Load a report and enforce visibility/ownership. Returns (row, is_owner).
    - missing -> 404; private & not owner -> 404 (hide existence);
      require_owner & not owner on a public report -> 403."""
    row = db.get_report(session_id)
    if not row:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Report not found."})
    is_owner = _is_owner(row, identity)
    if not row.get("is_public") and not is_owner:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Report not found."})
    if require_owner and not is_owner:
        raise HTTPException(status_code=403, detail={"code": "NOT_OWNER", "message": "You don't own this report."})
    return row, is_owner
```

Change `get_report@458` to take optional identity + enforce:

```python
@app.get("/report/{session_id}")
async def get_report(
    session_id: str,
    request: Request,
    identity: Identity = Depends(get_identity_optional),
    db: SupabaseDB = Depends(get_db),
):
    row, _ = _resolve_report_access(db, identity, session_id)  # render-token bypass added in Task 5
    return JSONResponse(content=row["report_json"])
```

(`Request` import already present; it's used for the render token in Task 5.)

- [ ] **Step 4: Run — expect PASS.** `venv/bin/python -m pytest tests/integration/test_report_privacy.py -v`

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(privacy): access helper + enforce GET /report visibility"`

---

## Task 2: Anon-aware publish/unpublish + `/meta` privacy

**Files:** Modify `src/api/main.py` (`publish@469`, `unpublish@490`, `report_meta@503`); append tests.

- [ ] **Step 1: Append failing tests**

```python
def test_anon_owner_can_publish():
    db = FakeDB(); rid = _seed(db, anon_id=ANON)
    r = _client(db, anon_identity(ANON)).post(f"/report/{rid}/publish")
    assert r.status_code == 200
    assert db.get_report(rid)["is_public"] is True

def test_non_owner_cannot_publish_private_404():
    db = FakeDB(); rid = _seed(db, user_id=OWNER)
    r = _client(db, auth_identity(OTHER)).post(f"/report/{rid}/publish")
    assert r.status_code == 404

def test_meta_hides_title_of_private_report_from_non_owner():
    db = FakeDB(); rid = _seed(db, user_id=OWNER)
    r = _client(db, None).get(f"/report/{rid}/meta")  # unauth (server-side generateMetadata)
    assert r.status_code == 200
    body = r.json()
    assert body["is_public"] is False and body["owned"] is False
    assert body["title"] != "Secret title"   # must NOT leak the private title

def test_meta_owner_sees_full_for_private():
    db = FakeDB(); rid = _seed(db, user_id=OWNER)
    r = _client(db, auth_identity(OWNER)).get(f"/report/{rid}/meta")
    assert r.json()["owned"] is True and r.json()["title"] == "Secret title"
```

- [ ] **Step 2: Run — expect FAIL** (anon publish 401/404; meta leaks title).

- [ ] **Step 3: Implement.** In `publish_report` and `unpublish_report`, replace `_require_report_owner(db, identity, session_id)` with `_resolve_report_access(db, identity, session_id, require_owner=True)`. Leave their bodies otherwise unchanged. (publish keeps `Depends(get_identity)`; an anon caller sends `X-Anon-Id` so `get_identity` returns an anon `Identity`.)

Rewrite `report_meta` body (after loading) so private non-owners get minimal data:

```python
@app.get("/report/{session_id}/meta")
async def report_meta(
    session_id: str,
    identity: Identity = Depends(get_identity_optional),
    db: SupabaseDB = Depends(get_db),
):
    row = db.get_report(session_id)
    if not row:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Report not found."})
    owned = _is_owner(row, identity)
    is_public = bool(row.get("is_public"))
    if not is_public and not owned:
        return {"is_public": False, "owned": False,
                "title": "Private report", "description": "", "og_image_url": None}
    title, desc = _report_title_desc(row.get("report_json") or {})
    return {"is_public": is_public, "title": title, "description": desc,
            "og_image_url": _public_urls(session_id, row)["og_image_url"], "owned": owned}
```

- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5: Commit** — `git commit -am "feat(privacy): anon-aware publish + meta hides private titles"`

---

## Task 3: Mutations require ownership

**Files:** Modify `src/api/main.py` (`patch_report_layout@527`, `generate_more@632`, `add_chart@763`, `deepen_report@892`); append tests.

- [ ] **Step 1: Append failing tests** (one per endpoint; non-owner must not mutate):

```python
@pytest.mark.parametrize("path,method", [
    ("/layout", "patch"), ("/generate-more", "post"),
    ("/add-chart", "post"), ("/deepen", "post"),
])
def test_mutations_blocked_for_non_owner(path, method):
    db = FakeDB(); rid = _seed(db, user_id=OWNER)            # private, owned by OWNER
    client = _client(db, auth_identity(OTHER))
    r = getattr(client, method)(f"/report/{rid}{path}", json={} if method != "patch" else {"layout": []})
    assert r.status_code == 404   # private + not owner -> hidden
```

- [ ] **Step 2: Run — expect FAIL** (mutations currently allow any identity).

- [ ] **Step 3: Implement.** At the top of each of the four handler bodies (right after the signature, before any work / credit spend), insert:

```python
    _resolve_report_access(db, identity, session_id, require_owner=True)
```

Use the handler's existing `session_id`, `db`, `identity` params (all four already inject `db` and `identity`). Place it **before** any `db.spend_credits` call so a non-owner is rejected before being charged.

- [ ] **Step 4: Run — expect PASS** (+ re-run existing mutation tests to confirm owner-path still works: `venv/bin/python -m pytest tests/integration -k "layout or generate_more or add_chart or deepen" -v`).
- [ ] **Step 5: Commit** — `git commit -am "feat(privacy): mutations require report ownership"`

---

## Task 4: Exports enforce visibility

**Files:** Modify `src/api/main.py` (6 export endpoints @1063–1233); append tests.

- [ ] **Step 1: Append failing tests**

```python
import pdf_export, report_export   # noqa

@pytest.mark.parametrize("ext", ["pdf", "pptx", "xlsx", "zip", "md", "html"])
def test_export_private_blocked_for_non_owner(ext, monkeypatch):
    db = FakeDB(); rid = _seed(db, user_id=OWNER)
    r = _client(db, auth_identity(OTHER)).get(f"/report/{rid}/export.{ext}")
    assert r.status_code == 404

def test_export_md_owner_ok():
    db = FakeDB(); rid = _seed(db, user_id=OWNER)
    r = _client(db, auth_identity(OWNER)).get(f"/report/{rid}/export.md")
    assert r.status_code == 200
```

- [ ] **Step 2: Run — expect FAIL** (exports have no visibility check).

- [ ] **Step 3: Implement.** Each export handler currently loads the report itself (e.g. `row = db.get_report(session_id)` or similar). Replace that load with the access helper at the top of each of the six bodies:

```python
    row, _ = _resolve_report_access(db, identity, session_id)
```

…and use `row["report_json"]` where they previously used the fetched row. (Exports follow the **view** rule — `require_owner=False` — so a public report stays exportable by anyone; a private one is owner-only.) Keep `Depends(get_identity)` as-is OR switch to `get_identity_optional` if a test exposes a hard 400 for anon on a public export — prefer `get_identity_optional` for the GET exports for consistency with `GET /report`.

- [ ] **Step 4: Run — expect PASS.** (`export.md`/`html` need no Playwright; `pdf`/`pptx` render — Task 5 handles the private render path. For this task, the non-owner 404 fires before any render, so the parametrized test passes without Playwright.)
- [ ] **Step 5: Commit** — `git commit -am "feat(privacy): exports enforce report visibility"`

---

## Task 5: Render token (so the owner can export a PRIVATE report)

**Files:** Create `src/api/render_token.py`; modify `src/api/main.py` (GET /report accepts token; export handlers mint + pass), `src/api/pdf_export.py` (accept token); `src/app/report/[id]/print/page.tsx` + `og/page.tsx` (forward token). Test: `tests/unit/test_render_token.py`.

**Why:** server-side Playwright renders `/print` & `/og`, which fetch `GET /report` with no user auth → a private report would 404 mid-render. A signed, 120s, report-scoped token authorizes that one render without exposing the report.

- [ ] **Step 1: Write failing test** `tests/unit/test_render_token.py`

```python
import time
from render_token import make_render_token, verify_render_token

def test_roundtrip_valid():
    t = make_render_token("abc")
    assert verify_render_token(t, "abc") is True

def test_wrong_report_id_rejected():
    assert verify_render_token(make_render_token("abc"), "xyz") is False

def test_tamper_rejected():
    assert verify_render_token(make_render_token("abc") + "x", "abc") is False

def test_expired_rejected():
    assert verify_render_token(make_render_token("abc", ttl=-1), "abc") is False

def test_garbage_rejected():
    assert verify_render_token("not.a.token", "abc") is False
```

- [ ] **Step 2: Run — expect FAIL** (module missing).

- [ ] **Step 3: Implement** `src/api/render_token.py`:

```python
"""Short-lived, report-scoped HMAC token authorizing one server-side render
(Playwright print/og) to read a PRIVATE report via GET /report without user auth.
Safe if leaked: scoped to a single report id and expires in ~2 minutes."""
import hashlib
import hmac
import os
import time

_TTL_DEFAULT = 120

def _secret() -> bytes:
    # Reuse the app's existing secret; fall back to the Supabase service key so the
    # token is unforgeable in every environment. Never logged.
    s = os.environ.get("RENDER_TOKEN_SECRET") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or ""
    return s.encode()

def make_render_token(report_id: str, ttl: int = _TTL_DEFAULT) -> str:
    exp = str(int(time.time()) + ttl)
    msg = f"{report_id}.{exp}".encode()
    sig = hmac.new(_secret(), msg, hashlib.sha256).hexdigest()
    return f"{exp}.{sig}"

def verify_render_token(token: str, report_id: str) -> bool:
    try:
        exp_s, sig = token.split(".", 1)
        if int(exp_s) < int(time.time()):
            return False
        expected = hmac.new(_secret(), f"{report_id}.{exp_s}".encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, expected)
    except (ValueError, AttributeError):
        return False
```

- [ ] **Step 4: Run — expect PASS.** `venv/bin/python -m pytest tests/unit/test_render_token.py -v`

- [ ] **Step 5: Wire the token into the access path.** In `GET /report` (Task 1), accept a valid render token BEFORE the access check:

```python
    rt = request.query_params.get("rt") or request.headers.get("x-render-token")
    if rt:
        from render_token import verify_render_token
        if verify_render_token(rt, session_id):
            row = db.get_report(session_id)
            if row:
                return JSONResponse(content=row["report_json"])
    row, _ = _resolve_report_access(db, identity, session_id)
    return JSONResponse(content=row["report_json"])
```

- [ ] **Step 6: Mint + pass the token in PDF/PPTX exports.** `pdf_export.render_report_pdf`, `render_og_image`, `render_chart_images` take an optional `render_token: str | None = None` and append it: `url = f"{_FRONTEND_BASE}/report/{session_id}/print"` → if token, `url += f"?rt={render_token}"`. In the export handlers that call these (after `_resolve_report_access`), do `rt = make_render_token(session_id)` (import at top: `from render_token import make_render_token`) and pass `render_token=rt`.

- [ ] **Step 7: Forward the token in the Next render routes.** `print/page.tsx:34` and `og/page.tsx` read it from the URL and pass it through:

```tsx
// print/page.tsx — change the fetch (line ~34)
const rt = typeof window !== 'undefined' ? new URLSearchParams(window.location.search).get('rt') : null;
fetch(`${process.env.NEXT_PUBLIC_API_URL}/report/${params.id}${rt ? `?rt=${encodeURIComponent(rt)}` : ''}`)
```

Apply the same change to the og route's report fetch.

- [ ] **Step 8: Add a backend test** that GET /report serves a private report with a valid token, and rejects an invalid one:

```python
def test_get_report_with_render_token():
    from render_token import make_render_token
    db = FakeDB(); rid = _seed(db, user_id=OWNER)
    c = _client(db, None)
    assert c.get(f"/report/{rid}?rt={make_render_token(rid)}").status_code == 200
    assert c.get(f"/report/{rid}?rt=bad.token").status_code == 404
```

- [ ] **Step 9: Run** `venv/bin/python -m pytest tests/unit/test_render_token.py tests/integration/test_report_privacy.py -v` — expect PASS. `tsc`: `cd <frontend>` and confirm no type error in the two routes (Vercel build is the real gate).

- [ ] **Step 10: Commit** — `git commit -am "feat(privacy): signed render token for owner exports of private reports"`

---

## Task 6: Frontend private-state + Share-modal copy

**Files:** Modify `src/app/report/[id]/ReportClient.tsx` (404/403 → private state), `src/app/report/[id]/ShareModal.tsx` (copy). `cloudbuild`-independent; verified by `tsc`/Vercel.

- [ ] **Step 1: ReportClient — distinguish private from expired.** Replace the `.then` 404 branch (line ~60) so 404/403 render a private state:

```tsx
.then(async (r) => {
  if (r.status === 404 || r.status === 403) {
    const e = new Error('This report is private, or it doesn’t exist.') as Error & { code?: string };
    e.code = 'PRIVATE';
    throw e;
  }
  if (!r.ok) throw new Error('Failed to load report');
  return r.json();
})
```

In `ErrorView`, when the error code is `PRIVATE`, show the private copy + a sign-in nudge:

```tsx
function ErrorView({ message, code }: { message: string; code?: string }) {
  const isPrivate = code === 'PRIVATE';
  return (
    <div className="theme-light flex flex-col items-center justify-center min-h-screen bg-canvas text-ink">
      <div className="text-center max-w-md">
        <h2 className="font-display text-2xl font-medium text-ink">
          {isPrivate ? 'This report is private' : 'Could not load report'}
        </h2>
        <p className="mt-2 text-ink-2">{message}</p>
        {isPrivate && <p className="mt-2 text-ink-3 text-sm">If it’s yours, sign in on the device that created it.</p>}
        <div className="mt-6 flex gap-3 justify-center">
          {isPrivate && <a href="/login" className="btn btn-secondary">Sign in</a>}
          <a href="/app" className="btn btn-primary">{isPrivate ? 'New report' : 'Back to upload'}</a>
        </div>
      </div>
    </div>
  );
}
```

Track the code in state: `const [errCode, setErrCode] = useState<string | undefined>();` set in `.catch((e) => { setError(e.message); setErrCode((e as any).code); })`, and render `<ErrorView message={error} code={errCode} />`.

- [ ] **Step 2: ShareModal copy.** Add a one-line note that publishing makes the report **public and search-engine indexable**, and that reports are **private by default**. (Find the modal's body copy and add a short `<p className="text-ink-3 text-sm">` to that effect; keep existing controls.)

- [ ] **Step 3: tsc** — confirm no type errors (`npx tsc --noEmit` if it runs locally; otherwise rely on the Vercel build in Task 7).

- [ ] **Step 4: Commit** — `git commit -am "feat(privacy): private-state report page + Share-modal copy"`

---

## Task 7: Full verification + deploy (USER-AUTHORIZED)

**Files:** none (gate + deploy). **Pause for explicit user authorization before deploying.**

- [ ] **Step 1:** Full backend suite — `venv/bin/python -m pytest -q` (expect green; allow for the slow stripe import on this machine).
- [ ] **Step 2:** Merge `true-private-reports` → `main` (controller; fast-forward push) — **user-authorized**.
- [ ] **Step 3:** Deploy backend (Cloud Run) — **user-authorized**:
  `CLOUDSDK_PYTHON=/opt/homebrew/opt/python@3.12/bin/python3.12 gcloud builds submit --config cloudbuild.yaml --project=chartsage-497909 --substitutions=_TAG=<sha>,_SUPABASE_URL=https://xxwtbegkgozufftuhbil.supabase.co,_STRIPE_PRICE_STARTER=price_1TdyM8Dx0m4Hh32Hp8BG6wej,_STRIPE_PRICE_STANDARD=price_1TdyMQDx0m4Hh32HRTJCaAs0,_STRIPE_PRICE_PRO=price_1TdyMfDx0m4Hh32HdeVMa9rO` (the 3 live price IDs + SUPABASE_URL MUST be passed). Then watch the Vercel build go Ready.
- [ ] **Step 4: Smoke (the bug repro).** Generate a report → copy its link → open in **incognito** → expect the **private** state (404). Sign in as the owner → it loads. Publish → the link works logged-out + is indexable. Make private → the link 404s again. Export a private report as the owner → PDF downloads (render token path).
- [ ] **Step 5:** Update `docs/FUTURE-IMPROVEMENTS.md` if any follow-ups surface (e.g., the deferred "unlisted" state).

---

## Self-review notes (spec coverage)
- Private-by-default enforced (Tasks 1, 3, 4) ✔ · anon-aware ownership + anon publish (Tasks 1, 2) ✔ · 404 for private non-owners (Task 1 helper) ✔ · `/meta` no title leak (Task 2) ✔ · exports gated + private render path (Tasks 4, 5) ✔ · frontend private state (Task 6) ✔ · no schema change (reuse `is_public`) ✔ · migration effect = existing unpublished become private (inherent; called out in Task 7 smoke) ✔.
