# Publish / Shareable Reports (SP5-A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Let a signed-in report owner publish a report to a public, search-indexable, embeddable page with a per-report social preview image; everything else stays unlisted + `noindex`.

**Architecture:** A `visibility` flag on `reports` + owner-gated publish/unpublish endpoints; a soft-auth `/meta` endpoint feeds Next's server-side `generateMetadata` (so crawlers/unfurlers see correct `<head>` + robots index/noindex) while the chart body stays client-rendered; a chrome-less `/embed` route doubles as the Playwright capture target for the OG image (stored in a public Supabase bucket); `sitemap.ts` lists published reports.

**Tech Stack:** FastAPI (Cloud Run) + Supabase (Postgres + Storage) + Playwright; Next.js 14 App Router (Vercel) + PostHog. Backend TDD; frontend implement→tsc→commit.

**Conventions / setup:**
- Interpreter `venv/bin/python`; run tests `venv/bin/python -m pytest …`.
- HTTPException detail = `{"code": "...", "message": "..."}`. Identity via `Depends(get_identity)` (`deps.Identity`, `.is_authenticated`, `.user_id`, `.distinct_id`). Test auth: `tests/helpers/fake_auth.py` `auth_identity(uid)` / `anon_identity(aid)` + the `_Holder` override pattern (`tests/integration/test_credits_endpoints.py`).
- **Branch:** controller creates `publish-reports` off main **before Task 1** (carries spec `a195d41`). **Subagents must NOT run `git checkout`/`switch`/`reset`/`stash`** — only `git add` + `git commit`.
- **The `reports` table already exists in prod (SP1) with no migration file** — `publish.sql` only `ALTER`s it. Reports row shape: `id, anon_id, user_id, report_json, csv_storage_key, title, created_at, updated_at`.
- **Build gate caution:** local `next build` is broken (Node 23). Local `tsc --noEmit` is the dev check, **but Vercel's `next build` type-checks server components / route handlers more strictly** (it caught a discriminated-union error tsc missed). This plan adds a server component (`generateMetadata`) — keep its types simple/explicit, and **Task 11 must watch the Vercel deploy reach `● Ready` (`vercel inspect <newest-url>`) before declaring the frontend shipped.** Production deploy + the Supabase/bucket provisioning require explicit user authorization.

**File structure:**
- `docs/migrations/publish.sql` *(new)* — ALTER reports.
- `src/api/db.py` *(modify)* — `set_report_visibility`, `list_public_reports`.
- `tests/helpers/fake_db.py` *(modify)* — visibility on rows + the two methods.
- `src/api/deps.py` *(modify)* — `get_identity_optional`.
- `src/api/storage.py` *(modify)* — `upload_public_image` (public `og-images` bucket).
- `src/api/pdf_export.py` *(modify)* — `render_og_image` (1200×630 screenshot of `/embed`).
- `src/api/main.py` *(modify)* — `_require_report_owner`, `_report_title_desc`, `_public_urls`; `publish`/`unpublish`/`meta`/`reports/public` endpoints; OG wiring.
- `tests/integration/test_publish.py` *(new)*; `tests/unit/test_fake_db_publish.py` *(new)*; `tests/unit/test_identity_optional.py` *(new)*.
- `src/app/report/[id]/embed/page.tsx` *(new)* — chrome-less, public-only.
- `src/app/report/[id]/ReportClient.tsx` *(new)* — current client logic, moved verbatim.
- `src/app/report/[id]/page.tsx` *(modify→server component)* — `generateMetadata` + renders `ReportClient`.
- `src/app/report/[id]/ShareModal.tsx` *(new)*; `src/app/report/[id]/Toolbar.tsx` *(modify)* — Share/Publish.
- `src/app/components/AppHeader.tsx` *(modify)* — hide on `/embed` (+ `/print`).
- `src/app/sitemap.ts`, `src/app/robots.ts`, `next.config.js` *(new)*; `src/app/layout.tsx` *(modify — metadataBase)*.

---

### Task 1: Migration — `publish.sql`

**Files:** Create `docs/migrations/publish.sql`

- [ ] **Step 1: Write the migration** (no automated test — verified at the live smoke):

```sql
-- docs/migrations/publish.sql
-- SP5-A publish/shareable reports. Run once in Supabase (the `reports` table
-- already exists from SP1). Opt-in public visibility + OG image key.
alter table reports add column if not exists is_public    boolean not null default false;
alter table reports add column if not exists og_image_key text;
alter table reports add column if not exists published_at  timestamptz;
create index if not exists reports_public_idx
  on reports (is_public, updated_at desc) where is_public;
```

- [ ] **Step 2: Commit**
```bash
git add docs/migrations/publish.sql
git commit -m "feat(publish): reports visibility migration (is_public, og_image_key)"
```

---

### Task 2: DB layer — visibility methods + FakeDB

**Files:** Modify `src/api/db.py`, `tests/helpers/fake_db.py`; Test `tests/unit/test_fake_db_publish.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_fake_db_publish.py
from uuid import uuid4
from tests.helpers.fake_db import FakeDB


def _save(db, user_id=None):
    rid = str(uuid4())
    db.save_report(rid, anon_id=None, user_id=user_id, report_json={"title": "T", "charts": []},
                   csv_storage_key=None, title="T")
    return rid


def test_set_visibility_and_list_public():
    db = FakeDB()
    r1 = _save(db); r2 = _save(db)
    assert db.get_report(r1)["is_public"] is False          # default
    assert db.set_report_visibility(r1, True, og_image_key="r1.png") is True
    assert db.get_report(r1)["is_public"] is True
    assert db.get_report(r1)["og_image_key"] == "r1.png"
    ids = {r["id"] for r in db.list_public_reports()}
    assert r1 in ids and r2 not in ids
    assert db.set_report_visibility(r1, False) is True
    assert db.get_report(r1)["is_public"] is False
    assert db.set_report_visibility("missing", True) is False
```

- [ ] **Step 2: Run → fail** — `venv/bin/python -m pytest tests/unit/test_fake_db_publish.py -v` → `AttributeError: … 'set_report_visibility'`.

- [ ] **Step 3: Implement on FakeDB** — in `tests/helpers/fake_db.py`, in `save_report`'s row dict add `"is_public": False, "og_image_key": None`. Add (in the reports section):
```python
    def set_report_visibility(self, report_id, is_public, og_image_key=None, published_at=None) -> bool:
        row = self._rows.get(report_id)
        if not row:
            return False
        row["is_public"] = is_public
        if og_image_key is not None:
            row["og_image_key"] = og_image_key
        return True

    def list_public_reports(self, limit: int = 5000) -> list[dict]:
        rows = [r for r in self._rows.values() if r.get("is_public")]
        rows.sort(key=lambda r: r.get("_seq", 0), reverse=True)
        return [{"id": r["id"], "updated_at": r.get("_seq")} for r in rows[:limit]]
```

- [ ] **Step 4: Implement on SupabaseDB** — in `src/api/db.py` after `update_layout`:
```python
    def set_report_visibility(self, report_id: str, is_public: bool,
                              og_image_key: str | None = None,
                              published_at: str | None = None) -> bool:
        patch: dict = {"is_public": is_public}
        if og_image_key is not None:
            patch["og_image_key"] = og_image_key
        if published_at is not None:
            patch["published_at"] = published_at
        res = self.client.table("reports").update(patch).eq("id", report_id).execute()
        return len(res.data) > 0

    def list_public_reports(self, limit: int = 5000) -> list[dict]:
        res = (self.client.table("reports").select("id, updated_at")
               .eq("is_public", True).order("updated_at", desc=True).limit(limit).execute())
        return res.data or []
```

- [ ] **Step 5: Run → pass.** `venv/bin/python -m pytest tests/unit/test_fake_db_publish.py -v` (1 passed).
- [ ] **Step 6: Commit**
```bash
git add src/api/db.py tests/helpers/fake_db.py tests/unit/test_fake_db_publish.py
git commit -m "feat(publish): report visibility db methods + FakeDB"
```

---

### Task 3: `get_identity_optional` (soft auth)

**Files:** Modify `src/api/deps.py`; Test `tests/unit/test_identity_optional.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_identity_optional.py
from deps import get_identity_optional


def test_no_headers_is_anonymous_not_error():
    ident = get_identity_optional(authorization=None, x_anon_id=None)
    assert ident.is_authenticated is False
    assert ident.user_id is None

def test_garbage_anon_is_anonymous():
    ident = get_identity_optional(authorization=None, x_anon_id="not-a-uuid")
    assert ident.is_authenticated is False
```

- [ ] **Step 2: Run → fail** (`ImportError: cannot import name 'get_identity_optional'`).

- [ ] **Step 3: Implement** — in `src/api/deps.py` (after `get_identity`):
```python
def get_identity_optional(
    authorization: str | None = Header(None),
    x_anon_id: str | None = Header(None),
) -> Identity:
    """Like get_identity but NEVER raises — for public endpoints that may be
    called with no auth (e.g. Next's server-side generateMetadata). Valid Bearer
    -> the user; anything else -> an anonymous Identity."""
    if authorization and authorization.lower().startswith("bearer "):
        user_id = verify_token(authorization[7:].strip())
        if user_id is not None:
            return Identity(user_id=user_id)
    if x_anon_id:
        try:
            return Identity(anon_id=UUID(x_anon_id))
        except (ValueError, AttributeError):
            pass
    return Identity()
```

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit**
```bash
git add src/api/deps.py tests/unit/test_identity_optional.py
git commit -m "feat(publish): get_identity_optional soft-auth dependency"
```

---

### Task 4: Publish / unpublish endpoints (owner-gated)

**Files:** Modify `src/api/main.py`; Test `tests/integration/test_publish.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/integration/test_publish.py
import pytest
from uuid import uuid4
from fastapi.testclient import TestClient
from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_posthog import FakePostHog
from tests.helpers.fake_auth import auth_identity, anon_identity


class _Holder:
    def __init__(self): self.current = None
    def __call__(self): return self.current


@pytest.fixture
def ctx(monkeypatch):
    monkeypatch.setenv("FRONTEND_BASE_URL", "https://chartsage.app")
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co")
    db, ph = FakeDB(), FakePostHog()
    holder = _Holder()
    from main import app, get_db, get_posthog, get_identity
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_posthog] = lambda: ph
    app.dependency_overrides[get_identity] = holder
    yield TestClient(app), db, ph, holder
    app.dependency_overrides.clear()


def _save(db, user_id):
    rid = str(uuid4())
    db.save_report(rid, anon_id=None, user_id=user_id, report_json={"title": "Sales", "charts": []},
                   csv_storage_key=None, title="Sales")
    return rid


def test_publish_owner_ok(ctx):
    tc, db, ph, holder = ctx
    user = str(uuid4()); rid = _save(db, user); holder.current = auth_identity(user)
    r = tc.post(f"/report/{rid}/publish")
    assert r.status_code == 200
    body = r.json()
    assert body["public_url"].endswith(f"/report/{rid}")
    assert body["embed_url"].endswith(f"/report/{rid}/embed")
    assert db.get_report(rid)["is_public"] is True
    assert len(ph.find("report_published")) == 1


def test_publish_non_owner_403(ctx):
    tc, db, ph, holder = ctx
    rid = _save(db, str(uuid4())); holder.current = auth_identity(str(uuid4()))  # different user
    r = tc.post(f"/report/{rid}/publish")
    assert r.status_code == 403 and r.json()["detail"]["code"] == "NOT_OWNER"
    assert db.get_report(rid)["is_public"] is False


def test_publish_anon_401(ctx):
    tc, db, ph, holder = ctx
    rid = _save(db, str(uuid4())); holder.current = anon_identity(str(uuid4()))
    assert tc.post(f"/report/{rid}/publish").status_code == 401


def test_publish_missing_404(ctx):
    tc, db, ph, holder = ctx
    holder.current = auth_identity(str(uuid4()))
    assert tc.post(f"/report/{uuid4()}/publish").status_code == 404


def test_unpublish_owner_flips(ctx):
    tc, db, ph, holder = ctx
    user = str(uuid4()); rid = _save(db, user); holder.current = auth_identity(user)
    db.set_report_visibility(rid, True)
    r = tc.post(f"/report/{rid}/unpublish")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert db.get_report(rid)["is_public"] is False
    assert len(ph.find("report_unpublished")) == 1
```

- [ ] **Step 2: Run → fail** (404, endpoints missing).

- [ ] **Step 3: Implement** — in `src/api/main.py`, add helpers (near the other module helpers) + endpoints (with the other `/report/{session_id}` routes). Confirm `os`, `Identity`, `SupabaseDB`, `PostHogServer`, `Depends`, `HTTPException` are imported (they are):
```python
def _require_report_owner(db, identity, session_id: str) -> dict:
    row = db.get_report(session_id)
    if not row:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Report not found."})
    if not identity.is_authenticated:
        raise HTTPException(status_code=401, detail={"code": "AUTH_REQUIRED", "message": "Sign in to manage this report."})
    if row.get("user_id") != str(identity.user_id):
        raise HTTPException(status_code=403, detail={"code": "NOT_OWNER", "message": "You don't own this report."})
    return row


def _public_urls(session_id: str, row: dict) -> dict:
    og = None
    key = row.get("og_image_key")
    if key:
        og = f"{os.environ.get('SUPABASE_URL', '')}/storage/v1/object/public/og-images/{key}"
    return {
        "public_url": f"{_FRONTEND_BASE}/report/{session_id}",
        "embed_url": f"{_FRONTEND_BASE}/report/{session_id}/embed",
        "og_image_url": og,
    }


@app.post("/report/{session_id}/publish")
async def publish_report(
    session_id: str,
    identity: Identity = Depends(get_identity),
    db: SupabaseDB = Depends(get_db),
    posthog: PostHogServer = Depends(get_posthog),
):
    _require_report_owner(db, identity, session_id)
    db.set_report_visibility(session_id, True, published_at="now()")
    posthog.capture(identity.distinct_id, "report_published", {"reportId": session_id})
    return _public_urls(session_id, db.get_report(session_id))


@app.post("/report/{session_id}/unpublish")
async def unpublish_report(
    session_id: str,
    identity: Identity = Depends(get_identity),
    db: SupabaseDB = Depends(get_db),
    posthog: PostHogServer = Depends(get_posthog),
):
    _require_report_owner(db, identity, session_id)
    db.set_report_visibility(session_id, False)
    posthog.capture(identity.distinct_id, "report_unpublished", {"reportId": session_id})
    return {"ok": True}
```
(`_FRONTEND_BASE` already exists in main.py from the payments work. The OG image is wired in Task 7 — publish here just sets visibility.)

- [ ] **Step 4: Run → pass** (`tests/integration/test_publish.py` 5 passed).
- [ ] **Step 5: Commit**
```bash
git add src/api/main.py tests/integration/test_publish.py
git commit -m "feat(publish): POST /report/{id}/publish + /unpublish (owner-gated)"
```

---

### Task 5: `/meta` + `/reports/public` endpoints

**Files:** Modify `src/api/main.py`; extend `tests/integration/test_publish.py`

- [ ] **Step 1: Add failing tests** (append to `test_publish.py`):

```python
def test_meta_owner_vs_anon(ctx):
    tc, db, ph, holder = ctx
    user = str(uuid4()); rid = _save(db, user)
    holder.current = auth_identity(user)
    m = tc.get(f"/report/{rid}/meta").json()
    assert m["is_public"] is False and m["owned"] is True and m["title"] == "Sales"
    holder.current = anon_identity(str(uuid4()))
    m2 = tc.get(f"/report/{rid}/meta").json()
    assert m2["owned"] is False


def test_meta_works_without_auth_headers(ctx):
    # generateMetadata calls this server-side with NO identity override; simulate by
    # removing the override so the real get_identity_optional runs with no headers.
    tc, db, ph, holder = ctx
    rid = _save(db, str(uuid4()))
    from main import app, get_identity
    app.dependency_overrides.pop(get_identity, None)   # use real soft-auth
    r = tc.get(f"/report/{rid}/meta")                  # no Authorization / X-Anon-Id
    assert r.status_code == 200 and r.json()["owned"] is False


def test_reports_public_lists_only_public(ctx):
    tc, db, ph, holder = ctx
    pub = _save(db, str(uuid4())); _priv = _save(db, str(uuid4()))
    db.set_report_visibility(pub, True)
    ids = {r["id"] for r in tc.get("/reports/public").json()}
    assert pub in ids and _priv not in ids
```
*(Note: `/meta` uses `get_identity_optional`, which is NOT overridden by the `get_identity` holder — so `test_meta_owner_vs_anon` needs `/meta` to read the holder's identity. To make owner/anon testable, `/meta` should depend on `get_identity_optional`; override THAT in the fixture too. Update the `ctx` fixture to also set `app.dependency_overrides[get_identity_optional] = holder` — import it from `main`.)*

- [ ] **Step 2: Update the fixture** — in `ctx`, import `get_identity_optional` from `main` and add `app.dependency_overrides[get_identity_optional] = holder`. (For `test_meta_works_without_auth_headers`, that test pops BOTH overrides — adjust it to also `app.dependency_overrides.pop(get_identity_optional, None)`.)

- [ ] **Step 3: Run → fail** (404 / missing).

- [ ] **Step 4: Implement** — in `src/api/main.py`, ensure `from deps import get_identity_optional` is imported, then add:
```python
def _report_title_desc(report_json: dict) -> tuple[str, str]:
    title = (report_json.get("title") or "ChartSage report").strip()[:120]
    narrative = report_json.get("narrative")
    summary = narrative.get("summary") if isinstance(narrative, dict) else (narrative if isinstance(narrative, str) else None)
    desc = (summary or "An AI-generated report — charts and insights from a spreadsheet.").strip()[:200]
    return title, desc


@app.get("/report/{session_id}/meta")
async def report_meta(
    session_id: str,
    identity: Identity = Depends(get_identity_optional),
    db: SupabaseDB = Depends(get_db),
):
    row = db.get_report(session_id)
    if not row:
        raise HTTPException(status_code=404, detail={"code": "NOT_FOUND", "message": "Report not found."})
    title, desc = _report_title_desc(row.get("report_json") or {})
    owned = bool(identity.is_authenticated and row.get("user_id") == str(identity.user_id))
    return {
        "is_public": bool(row.get("is_public")),
        "title": title, "description": desc,
        "og_image_url": _public_urls(session_id, row)["og_image_url"],
        "owned": owned,
    }


@app.get("/reports/public")
async def reports_public(db: SupabaseDB = Depends(get_db)):
    return db.list_public_reports()
```

- [ ] **Step 5: Run → pass; then full suite** — `venv/bin/python -m pytest tests/integration/test_publish.py -v` then `venv/bin/python -m pytest -p no:warnings -o console_output_style=classic 2>&1 | grep -E "passed|failed" | tail -1`.
- [ ] **Step 6: Commit**
```bash
git add src/api/main.py tests/integration/test_publish.py
git commit -m "feat(publish): GET /report/{id}/meta (soft-auth) + GET /reports/public"
```

---

### Task 6: Embed view (chrome-less, public-only)

**Files:** Create `src/app/report/[id]/embed/page.tsx`; Modify `src/app/components/AppHeader.tsx`

- [ ] **Step 1: Build the embed page.** A `'use client'` page that fetches `/report/{id}/meta` then `/report/{id}` (via `apiFetch`), and:
  - if `meta.is_public` is false → render a small centered "This report isn't public." placeholder (NO data fetched/rendered).
  - if public → render the report's **charts only** (reuse the existing chart components from `ReportView` — `KpiTiles`, `ChartCard`, `ReportSummary` via dynamic import, `ssr:false`), in a simple vertical stack, **no Toolbar / Sidebar / drag-drop / header**, padded for a clean 1200×630-friendly top section.
  - Export route metadata so it's not indexed standalone but passes value when embedded:
```tsx
// at top of embed/page.tsx (it's a client page, so set robots via a <head> or a metadata export from a tiny server wrapper).
// Simplest: make embed/page.tsx a SERVER component that exports:
export const metadata = {
  robots: { index: false, follow: false, 'max-image-preview': 'large' },
  other: { robots: 'noindex, indexifembedded' },
};
// ...and render a <EmbedClient id={params.id} /> ('use client') child holding the fetch+charts.
```
  (Mirror the Task 8 server/client split: `embed/page.tsx` server component exports `metadata` + renders `EmbedClient`; create `src/app/report/[id]/embed/EmbedClient.tsx` `'use client'` with the fetch + chart rendering.)

- [ ] **Step 2: Hide global chrome on embed (+ print).** In `src/app/components/AppHeader.tsx` (client; uses `usePathname`), return `null` when the path matches `/report/*/embed` (and `/print` if not already hidden):
```tsx
const pathname = usePathname();
if (pathname && (pathname.endsWith('/embed') || pathname.endsWith('/print'))) return null;
```
(If `AppHeader` isn't already a client component using `usePathname`, add `'use client'` + `import { usePathname } from 'next/navigation'`.)

- [ ] **Step 3: Type-check** — `npx tsc --noEmit -p tsconfig.json` (clean).
- [ ] **Step 4: Commit**
```bash
git add "src/app/report/[id]/embed/page.tsx" "src/app/report/[id]/embed/EmbedClient.tsx" src/app/components/AppHeader.tsx
git commit -m "feat(publish): chrome-less public-only /report/[id]/embed view"
```

---

### Task 7: OG image generation (Playwright) + wire into publish

**Files:** Modify `src/api/storage.py`, `src/api/pdf_export.py`, `src/api/main.py`; extend `tests/integration/test_publish.py`

- [ ] **Step 1: Add failing test** (append to `test_publish.py`) — publish should attempt OG generation + store the key; a render failure must NOT fail the publish:

```python
def test_publish_generates_og(ctx, monkeypatch):
    tc, db, ph, holder = ctx
    user = str(uuid4()); rid = _save(db, user); holder.current = auth_identity(user)
    calls = {}
    async def fake_render(session_id):
        calls["rendered"] = session_id
        return b"\x89PNG..."
    import pdf_export
    monkeypatch.setattr(pdf_export, "render_og_image", fake_render)
    # storage is overridden to a fake that records the upload
    from main import app, get_storage
    class FakeStorage:
        def upload_public_image(self, key, png): calls["key"] = key; return key
    app.dependency_overrides[get_storage] = lambda: FakeStorage()
    r = tc.post(f"/report/{rid}/publish")
    assert r.status_code == 200
    assert calls["rendered"] == rid and calls["key"] == f"{rid}.png"
    assert db.get_report(rid)["og_image_key"] == f"{rid}.png"


def test_publish_survives_og_failure(ctx, monkeypatch):
    tc, db, ph, holder = ctx
    user = str(uuid4()); rid = _save(db, user); holder.current = auth_identity(user)
    import pdf_export
    async def boom(session_id): raise RuntimeError("render failed")
    monkeypatch.setattr(pdf_export, "render_og_image", boom)
    r = tc.post(f"/report/{rid}/publish")
    assert r.status_code == 200 and db.get_report(rid)["is_public"] is True   # published anyway
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement storage** — `src/api/storage.py`, add near the top `OG_BUCKET = "og-images"` and the method:
```python
    def upload_public_image(self, key: str, png_bytes: bytes) -> str:
        """Upload to the public og-images bucket; returns the storage key."""
        try:
            self.client.storage.from_(OG_BUCKET).upload(
                path=key, file=png_bytes,
                file_options={"content-type": "image/png", "upsert": "true"},
            )
        except Exception as e:
            raise StorageError(f"og upload failed: {e}") from e
        return key
```

- [ ] **Step 4: Implement the renderer** — `src/api/pdf_export.py`, add (mirroring `render_report_pdf`'s use of `_ensure_browser` + `_FRONTEND_BASE`):
```python
async def render_og_image(session_id: str) -> bytes:
    await _ensure_browser()
    page = await _browser.new_page(viewport={"width": 1200, "height": 630})
    try:
        await page.goto(f"{_FRONTEND_BASE}/report/{session_id}/embed",
                        wait_until="networkidle", timeout=30_000)
        return await page.screenshot(type="png",
                                     clip={"x": 0, "y": 0, "width": 1200, "height": 630})
    finally:
        await page.close()
```
(Match the existing `_ensure_browser`/`_browser` usage exactly — read `pdf_export.py` to confirm whether `_ensure_browser` returns the browser or sets the `_browser` global, and follow that pattern.)

- [ ] **Step 5: Wire into publish** — in `src/api/main.py`, change `publish_report` to add `storage: SupabaseStorage = Depends(get_storage)` and insert the OG block after `set_report_visibility(..., True, ...)` and before the return:
```python
    db.set_report_visibility(session_id, True, published_at="now()")
    try:
        from pdf_export import render_og_image
        png = await render_og_image(session_id)
        og_key = storage.upload_public_image(f"{session_id}.png", png)
        db.set_report_visibility(session_id, True, og_image_key=og_key)
    except Exception:
        logging.exception("OG image generation failed; publishing without a custom preview")
    posthog.capture(identity.distinct_id, "report_published", {"reportId": session_id})
    return _public_urls(session_id, db.get_report(session_id))
```
(Import `from pdf_export import render_og_image` is done lazily inside the try so the monkeypatch on the module works and an import/runtime error degrades gracefully. Ensure `SupabaseStorage`/`get_storage` are imported in main.py — they are, used by other endpoints.)

- [ ] **Step 6: Run → pass; full suite green.**
- [ ] **Step 7: Commit**
```bash
git add src/api/storage.py src/api/pdf_export.py src/api/main.py tests/integration/test_publish.py
git commit -m "feat(publish): per-report OG image via Playwright embed capture"
```

---

### Task 8: Report page → server metadata + client split

**Files:** Create `src/app/report/[id]/ReportClient.tsx`; rewrite `src/app/report/[id]/page.tsx`; modify `src/app/layout.tsx`

- [ ] **Step 1: Move the client logic verbatim.** Create `src/app/report/[id]/ReportClient.tsx` containing the **current entire contents of `page.tsx` unchanged** (keep the `'use client'` directive, all imports, `Loading`/`ErrorView`/`ReportView`), with one change: rename the default-exported component from `ReportPage` to **`ReportClient`** and keep its `{ params }: { params: { id: string } }` prop. (It already reads `params.id`.)

- [ ] **Step 2: Rewrite `page.tsx` as a server component** (remove `'use client'`):
```tsx
import type { Metadata } from 'next';
import ReportClient from './ReportClient';

const BASE = process.env.NEXT_PUBLIC_API_URL!;
const SITE = process.env.NEXT_PUBLIC_SITE_URL || 'https://chartsage.app';

export async function generateMetadata({ params }: { params: { id: string } }): Promise<Metadata> {
  try {
    const res = await fetch(`${BASE}/report/${params.id}/meta`, { cache: 'no-store' });
    if (!res.ok) return { robots: { index: false, follow: false } };
    const m = await res.json();
    if (!m.is_public) return { robots: { index: false, follow: false } };
    const url = `${SITE}/report/${params.id}`;
    return {
      title: m.title,
      description: m.description,
      alternates: { canonical: url },
      robots: { index: true, follow: true },
      openGraph: {
        title: m.title, description: m.description, url, type: 'article',
        images: m.og_image_url ? [{ url: m.og_image_url, width: 1200, height: 630 }] : undefined,
      },
      twitter: {
        card: 'summary_large_image', title: m.title, description: m.description,
        images: m.og_image_url ? [m.og_image_url] : undefined,
      },
    };
  } catch {
    return { robots: { index: false, follow: false } };
  }
}

export default function Page({ params }: { params: { id: string } }) {
  return <ReportClient params={params} />;
}
```
Add `NEXT_PUBLIC_SITE_URL=https://chartsage.app` to the frontend env (Vercel) — or it defaults to `https://chartsage.app`.

- [ ] **Step 3: Set `metadataBase`** in `src/app/layout.tsx` so relative OG URLs resolve:
```tsx
export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL || 'https://chartsage.app'),
  title: 'ChartSage - AI-Powered Data Visualization',
  description: 'Turn any spreadsheet into a beautiful, interactive report with AI-generated insights in seconds.',
};
```

- [ ] **Step 4: Type-check** — `npx tsc --noEmit -p tsconfig.json` (clean). **Be precise with types in `generateMetadata` (server-side, strictly checked by Vercel's build).**
- [ ] **Step 5: Commit**
```bash
git add "src/app/report/[id]/page.tsx" "src/app/report/[id]/ReportClient.tsx" src/app/layout.tsx
git commit -m "feat(publish): server-rendered report metadata (index/OG for public, noindex else)"
```

---

### Task 9: Share / Publish UI (owner-only)

**Files:** Create `src/app/report/[id]/ShareModal.tsx`; modify `src/app/report/[id]/Toolbar.tsx`

- [ ] **Step 1: Build `ShareModal.tsx`** (`'use client'`, props `{ open, onClose, sessionId, initialIsPublic }`):
  - If not yet public: a confirmation body — *"Publishing makes this report and its charts public and indexable by search engines. Your uploaded file is never shared. You can make it private again anytime."* + a **Publish** button → `apiFetch('/report/${sessionId}/publish', {method:'POST'})` → on ok, fire `posthog.capture?.('report_published', {reportId})`, flip to the "published" state with the returned `public_url`.
  - If public: show the **public link** (read-only input + Copy), an **embed snippet** (`<iframe src="{embed_url}" width="100%" height="600" style="border:0" loading="lazy"></iframe>`, read-only + Copy), and a **"Make private"** button → `apiFetch('/report/${sessionId}/unpublish', {method:'POST'})` → fire `report_unpublished` → flip back. Use semantic tokens (`card`, `btn`, `btn-primary`, `btn-ghost`, `text-ink/ink-2`, `border-line`).

- [ ] **Step 2: Wire into `Toolbar.tsx`** — owner detection + the Share button:
  - Add state `const [owned, setOwned] = useState(false); const [isPublic, setIsPublic] = useState(false); const [showShare, setShowShare] = useState(false);`
  - `useEffect`: `apiFetch('/report/${sessionId}/meta').then(r=>r.ok&&r.json()).then(m=>{ if(m){ setOwned(!!m.owned); setIsPublic(!!m.is_public);} }).catch(()=>{})` (client-side, sends the Bearer → `owned` true for the owner).
  - Render a **"Share"** `btn btn-ghost` (only when `owned`) before the Export dropdown → `onClick={()=>setShowShare(true)}`.
  - Render `<ShareModal open={showShare} onClose={()=>setShowShare(false)} sessionId={sessionId} initialIsPublic={isPublic} />` alongside the other modals.

- [ ] **Step 3: Type-check** — `npx tsc --noEmit -p tsconfig.json`.
- [ ] **Step 4: Commit**
```bash
git add "src/app/report/[id]/ShareModal.tsx" "src/app/report/[id]/Toolbar.tsx"
git commit -m "feat(publish): owner Share/Publish UI + embed snippet in the report toolbar"
```

---

### Task 10: Sitemap + robots + embed iframe headers

**Files:** Create `src/app/sitemap.ts`, `src/app/robots.ts`, `next.config.js`

- [ ] **Step 1: `src/app/sitemap.ts`**
```ts
import type { MetadataRoute } from 'next';
const SITE = process.env.NEXT_PUBLIC_SITE_URL || 'https://chartsage.app';
const API = process.env.NEXT_PUBLIC_API_URL!;

export const revalidate = 3600;

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const staticRoutes = ['', '/app', '/terms', '/privacy', '/contact', '/pricing']
    .map((p) => ({ url: `${SITE}${p}`, changeFrequency: 'weekly' as const, priority: p === '' ? 1 : 0.6 }));
  let reports: MetadataRoute.Sitemap = [];
  try {
    const res = await fetch(`${API}/reports/public`, { next: { revalidate: 3600 } });
    if (res.ok) {
      reports = (await res.json()).map((r: { id: string }) => ({
        url: `${SITE}/report/${r.id}`, changeFrequency: 'monthly' as const, priority: 0.5,
      }));
    }
  } catch { /* sitemap still serves the static routes */ }
  return [...staticRoutes, ...reports];
}
```
(Drop any static route that doesn't exist; keep only real pages.)

- [ ] **Step 2: `src/app/robots.ts`**
```ts
import type { MetadataRoute } from 'next';
const SITE = process.env.NEXT_PUBLIC_SITE_URL || 'https://chartsage.app';
export default function robots(): MetadataRoute.Robots {
  return { rules: { userAgent: '*', allow: '/' }, sitemap: `${SITE}/sitemap.xml` };
}
```

- [ ] **Step 3: `next.config.js`** — explicitly allow framing the embed route (CSP frame-ancestors), leave everything else default:
```js
/** @type {import('next').NextConfig} */
const nextConfig = {
  async headers() {
    return [{
      source: '/report/:id/embed',
      headers: [{ key: 'Content-Security-Policy', value: 'frame-ancestors *' }],
    }];
  },
};
module.exports = nextConfig;
```

- [ ] **Step 4: Type-check** — `npx tsc --noEmit -p tsconfig.json`.
- [ ] **Step 5: Commit**
```bash
git add src/app/sitemap.ts src/app/robots.ts next.config.js
git commit -m "feat(publish): sitemap (public reports) + robots + embed frame headers"
```

---

### Task 11: Build + QA + deploy + smoke — **user-authorized**

- [ ] **Step 1: Full backend suite** — `venv/bin/python -m pytest -p no:warnings -o console_output_style=classic 2>&1 | grep -E "passed|failed|error" | tail -1` (all green).
- [ ] **Step 2: Type-check** — `npx tsc --noEmit -p tsconfig.json` (clean). *(Local `next build` is broken on Node 23 — Vercel is the build gate; Step 6 watches it.)*
- [ ] **Step 3: User provisioning (STOP — user actions):**
  1. Run `docs/migrations/publish.sql` in Supabase → SQL editor.
  2. Create a **public** Storage bucket named **`og-images`** (Supabase → Storage → New bucket → Public).
  3. (Optional) set `NEXT_PUBLIC_SITE_URL=https://chartsage.app` in Vercel env (defaults to that if unset).
- [ ] **Step 4: Deploy backend (authorized):**
```bash
cd /Users/chrissilver/Documents/ChartSage
CLOUDSDK_PYTHON=/opt/homebrew/opt/python@3.12/bin/python3.12 gcloud builds submit \
  --config cloudbuild.yaml \
  --substitutions=_SUPABASE_URL=https://xxwtbegkgozufftuhbil.supabase.co,_TAG=$(git rev-parse --short HEAD) \
  --project=chartsage-497909
```
(Do NOT source `FRONTEND_BASE_URL`.)
- [ ] **Step 5: Merge frontend → main (authorized):** controller runs `git checkout main && git merge --ff-only publish-reports && git push origin main`.
- [ ] **Step 6: WATCH the Vercel build** — `vercel inspect <newest-deployment-url>` until `status ● Ready` (server-component build risk). If it errors, fetch logs (`vercel inspect <url> --logs`), fix, redeploy. Do not declare shipped until Ready.
- [ ] **Step 7: Live smoke:** sign in → open one of your reports → **Share → Publish** → confirm: (a) `GET …/report/<id>/meta` returns `is_public:true` + an `og_image_url`; (b) the public page's `<head>` has `robots: index` + the OpenGraph image (view-source / a social-preview debugger); (c) `https://chartsage.app/sitemap.xml` lists the report; (d) the `<iframe>` embed snippet renders on a scratch HTML page; (e) a non-published report shows `noindex`; (f) **Make private** reverts it. `report_published` fires in PostHog.

---

## Self-Review

**Spec coverage:** visibility column → T1; db methods → T2; `get_identity_optional` → T3; publish/unpublish owner-gating → T4; meta + reports/public → T5; embed view → T6; OG image (Playwright + storage) → T7; server `generateMetadata` index/noindex + client split → T8; Share/Publish UI + embed snippet → T9; sitemap/robots/embed headers → T10; deploy + bucket + migration + smoke → T11. Privacy guarantee (noindex default, opt-in publish, raw CSV untouched) is enforced by T8 (noindex unless public) + T4 (owner-gated publish) + the fact no endpoint exposes the CSV. All spec sections covered.

**Placeholder scan:** No TBD/"handle errors"/"similar to". Each step has concrete code or a precise mechanical instruction (the verbatim ReportClient move). Backend tasks are full TDD with real test bodies.

**Type/name consistency:** `set_report_visibility(report_id, is_public, og_image_key=None, published_at=None)` and `list_public_reports(limit)` identical across `db.py`, `FakeDB`, and callers. `get_identity_optional` used by `/meta` (T5) is defined in T3 and overridden in the T5 fixture. `_public_urls`/`_require_report_owner`/`_report_title_desc` defined in T4–T5 and reused. `og_image_key` = `"{session_id}.png"` consistent between T7 storage upload and the URL builder. `render_og_image(session_id)` defined in T7 and monkeypatched in its test. Frontend: `meta.owned`/`is_public`/`og_image_url`/`public_url`/`embed_url` field names match the T5 endpoint response across T8/T9/T6.
