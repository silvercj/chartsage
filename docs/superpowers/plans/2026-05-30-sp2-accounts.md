# SP2 — Accounts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real accounts (Google + magic link via Supabase Auth) so logged-in users get unlimited reports + Generate-more, anonymous users keep the 1-free-report gate and hit an upsell on Generate-more, the anon's first report migrates into their new account, and a My Reports page lets users revisit saved reports.

**Architecture:** The FastAPI backend gains an identity layer — a `get_identity()` dependency that verifies a Supabase JWT (Bearer token) against the cached public JWKS and otherwise falls back to the existing `X-Anon-Id` header. Gating collapses to *authenticated → unlimited*, *anonymous → capped*. Two new endpoints (`/claim-anon-reports`, `/my-reports`) handle migration and the dashboard. The Next.js frontend adds a Supabase browser client, a server-side auth-callback route that exchanges the code and claims anon reports, plus login / welcome / reports pages, an upsell modal, and a top-right auth nav. RLS is enabled on `reports` as defense-in-depth (the backend keeps using the service-role key, which bypasses RLS).

**Tech Stack:** FastAPI + PyJWT (JWKS verification) on Cloud Run · Supabase Auth (asymmetric ES256 JWTs) · Next.js 14 App Router + `@supabase/ssr` + `@supabase/supabase-js` on Vercel · PostHog.

**Spec:** `docs/superpowers/specs/2026-05-30-sp2-accounts-design.md`

---

## Key facts grounded in the current codebase

- `src/api/deps.py` currently exposes `get_anon_id` only. We **add** `Identity` + `get_identity` and **keep** `get_anon_id` (its unit tests in `tests/unit/test_deps.py` must stay green; it becomes unused by `main.py` and can be removed in a later cleanup).
- `get_identity` must return the **literal error codes `MISSING_ANON_ID` / `INVALID_ANON_ID`** for the no-identity / bad-anon cases, because `tests/integration/test_api_errors.py::test_missing_anon_id_returns_400` asserts `"MISSING_ANON_ID" in resp.text`.
- The two anonymous Generate-more tests in `tests/integration/test_api_layout.py` (`test_generate_more_appends_charts`, `test_generate_more_unknown_session`) will break under the new anon→402 rule. They are **updated** in Task 5 to authenticate. This is intended behaviour change, not a regression.
- PyJWT 2.13.0 and cryptography 44.0.2 are **already installed** (pulled in by `supabase-auth`/`gotrue`). `jwt.PyJWKClient` is available. We pin `PyJWT[crypto]` explicitly because SP2 uses it directly.
- Report `id` is `uuid4().hex` (unguessable). `get_report` and `patch_layout` have **no identity check today and keep none** — reports stay link-accessible; RLS guards direct browser DB access; My Reports is the curated owned list. Do **not** add ownership checks to `get_report`/`patch_layout` (it would break the anon-creates-then-views and the no-auth Playwright print routes).
- Frontend has **no test harness** (manual smoke per project norm). Backend tasks are full TDD; frontend tasks are implement-then-`tsc --noEmit`-then-manual-smoke, each ending in a commit.
- New `src/app/components/` directory is introduced (existing components live under `src/app/report/[id]/`).

## File structure

### New files
```
src/api/auth.py                          # verify_token() — JWKS-based Supabase JWT verification
tests/unit/test_auth.py                  # verify_token unit tests (RSA test keypair)
tests/unit/test_get_identity.py          # get_identity branching unit tests
tests/helpers/fake_auth.py               # auth_identity()/anon_identity() builders for tests
tests/integration/test_auth_gating.py    # authed-unlimited, anon generate-more 402, authed generate-more 200
tests/integration/test_claim_and_reports.py  # claim migration + my-reports isolation
src/app/lib/supabase.ts                  # browser Supabase client + getAccessToken()
src/app/auth/callback/route.ts           # code exchange + anon-report claim + redirect
src/app/login/page.tsx                   # Google + magic-link login
src/app/welcome/page.tsx                 # onboarding (client-side first-visit gate)
src/app/reports/page.tsx                 # My Reports dashboard
src/app/components/UpsellModal.tsx        # generate-more upsell for anonymous users
src/app/components/AuthNav.tsx            # top-right sign-in / my-reports / sign-out
src/app/components/SessionWatcher.tsx     # PostHog identify on login + reset on logout
```

### Modified files
```
requirements.txt                         # pin PyJWT[crypto]
src/api/deps.py                           # add Identity + get_identity (keep get_anon_id)
src/api/db.py                             # add claim_anon_reports + list_user_reports
tests/helpers/fake_db.py                  # mirror claim_anon_reports + list_user_reports (+ insertion seq)
src/api/main.py                           # identity-aware gating; 402 on anon generate-more; +2 endpoints
tests/integration/test_api_layout.py      # authenticate the 2 generate-more tests
package.json                              # @supabase/ssr + @supabase/supabase-js
src/app/lib/api.ts                        # inject Authorization: Bearer when a session exists
src/app/report/[id]/Toolbar.tsx           # 402 → UpsellModal
src/app/anon-limit/page.tsx               # real /login link
src/app/layout.tsx                        # mount SessionWatcher + AuthNav
.env / Vercel                             # NEXT_PUBLIC_SUPABASE_URL (provisioning, Task 16)
```

---

## Task 1: Pin PyJWT in requirements

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the pin** under the "Infra / integrations" group in `requirements.txt`, right after the `supabase==2.30.1` line:

```
PyJWT[crypto]==2.13.0  # Supabase JWT (JWKS) verification; already present via supabase-auth
```

- [ ] **Step 2: Verify it resolves with no conflict**

Run: `./venv/bin/pip install --dry-run -r requirements.txt`
Expected: completes without an error; PyJWT/cryptography report as already satisfied (no downgrade of supabase/gotrue).

- [ ] **Step 3: Confirm the import works**

Run: `./venv/bin/python -c "from jwt import PyJWKClient; import jwt; print(jwt.__version__)"`
Expected: prints `2.13.0` and no ImportError.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build: pin PyJWT[crypto] for Supabase JWT verification"
```

---

## Task 2: Backend JWT verification (`auth.py`)

**Files:**
- Create: `src/api/auth.py`
- Test: `tests/unit/test_auth.py`

- [ ] **Step 1: Write the failing test** — `tests/unit/test_auth.py`:

```python
import time
from uuid import UUID, uuid4

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from auth import verify_token


def _keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    pub = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv, pub


def _token(priv, *, sub, aud="authenticated", exp_offset=3600):
    payload = {"sub": sub, "aud": aud, "exp": int(time.time()) + exp_offset}
    return jwt.encode(payload, priv, algorithm="RS256")


def test_valid_token_returns_user_uuid():
    priv, pub = _keypair()
    uid = str(uuid4())
    assert verify_token(_token(priv, sub=uid), _public_key=pub) == UUID(uid)


def test_expired_token_returns_none():
    priv, pub = _keypair()
    assert verify_token(_token(priv, sub=str(uuid4()), exp_offset=-10), _public_key=pub) is None


def test_wrong_audience_returns_none():
    priv, pub = _keypair()
    assert verify_token(_token(priv, sub=str(uuid4()), aud="anon"), _public_key=pub) is None


def test_tampered_token_returns_none():
    priv, pub = _keypair()
    tok = _token(priv, sub=str(uuid4()))
    tampered = tok[:-3] + ("aaa" if not tok.endswith("aaa") else "bbb")
    assert verify_token(tampered, _public_key=pub) is None


def test_wrong_key_returns_none():
    priv, _ = _keypair()
    _, other_pub = _keypair()
    assert verify_token(_token(priv, sub=str(uuid4())), _public_key=other_pub) is None


def test_non_uuid_sub_returns_none():
    priv, pub = _keypair()
    assert verify_token(_token(priv, sub="not-a-uuid"), _public_key=pub) is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./venv/bin/python -m pytest tests/unit/test_auth.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'auth'`.

- [ ] **Step 3: Implement** — create `src/api/auth.py`:

```python
"""Supabase JWT verification via cached JWKS.

Authenticated browser requests carry `Authorization: Bearer <supabase access token>`.
We verify the token's signature against Supabase's public JWKS (asymmetric ES256/RS256),
plus audience + expiry, entirely locally — no per-request call to Supabase.
"""
import os
from uuid import UUID

import jwt
from jwt import PyJWKClient

_SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
_AUDIENCE = "authenticated"
_ALGORITHMS = ["ES256", "RS256"]

_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        if not _SUPABASE_URL:
            raise RuntimeError("SUPABASE_URL must be set for JWT verification")
        url = f"{_SUPABASE_URL}/auth/v1/.well-known/jwks.json"
        _jwks_client = PyJWKClient(url, cache_keys=True)
    return _jwks_client


def verify_token(token: str, *, _public_key=None) -> UUID | None:
    """Return the user UUID (the `sub` claim) if the token is valid, else None.

    `_public_key` is a test injection point: pass a PEM public key to verify
    against directly instead of fetching the live JWKS.
    """
    try:
        if _public_key is not None:
            key = _public_key
            algorithms = ["RS256", "ES256"]
        else:
            key = _get_jwks_client().get_signing_key_from_jwt(token).key
            algorithms = _ALGORITHMS
        claims = jwt.decode(
            token,
            key,
            algorithms=algorithms,
            audience=_AUDIENCE,
            options={"verify_exp": True},
        )
    except Exception:
        return None
    sub = claims.get("sub")
    if not sub:
        return None
    try:
        return UUID(str(sub))
    except (ValueError, AttributeError):
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/unit/test_auth.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/auth.py tests/unit/test_auth.py
git commit -m "feat(api): Supabase JWT verification via cached JWKS"
```

---

## Task 3: Identity dependency (`get_identity`)

**Files:**
- Modify: `src/api/deps.py`
- Test: `tests/unit/test_get_identity.py`

- [ ] **Step 1: Write the failing test** — `tests/unit/test_get_identity.py`:

```python
from uuid import uuid4

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from deps import Identity, get_identity


def _client():
    app = FastAPI()

    @app.get("/whoami")
    def whoami(identity: Identity = Depends(get_identity)):
        return {
            "user_id": str(identity.user_id) if identity.user_id else None,
            "anon_id": str(identity.anon_id) if identity.anon_id else None,
            "auth": identity.is_authenticated,
        }

    return TestClient(app)


def test_anon_only_is_unauthenticated():
    anon = str(uuid4())
    res = _client().get("/whoami", headers={"X-Anon-Id": anon})
    assert res.status_code == 200
    assert res.json() == {"user_id": None, "anon_id": anon, "auth": False}


def test_no_headers_returns_400_missing_anon():
    res = _client().get("/whoami")
    assert res.status_code == 400
    assert "MISSING_ANON_ID" in res.text


def test_malformed_anon_returns_400_invalid_anon():
    res = _client().get("/whoami", headers={"X-Anon-Id": "not-a-uuid"})
    assert res.status_code == 400
    assert "INVALID_ANON_ID" in res.text


def test_valid_bearer_is_authenticated(monkeypatch):
    uid = uuid4()
    monkeypatch.setattr("deps.verify_token", lambda t: uid)
    res = _client().get("/whoami", headers={"Authorization": "Bearer good"})
    assert res.status_code == 200
    body = res.json()
    assert body["user_id"] == str(uid)
    assert body["auth"] is True


def test_invalid_bearer_returns_401_no_downgrade(monkeypatch):
    # A present-but-invalid token is a hard 401 even if X-Anon-Id is also sent.
    monkeypatch.setattr("deps.verify_token", lambda t: None)
    res = _client().get(
        "/whoami",
        headers={"Authorization": "Bearer bad", "X-Anon-Id": str(uuid4())},
    )
    assert res.status_code == 401
    assert "INVALID_TOKEN" in res.text
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./venv/bin/python -m pytest tests/unit/test_get_identity.py -v`
Expected: FAIL — `ImportError: cannot import name 'Identity' from 'deps'`.

- [ ] **Step 3: Implement** — append to `src/api/deps.py` (keep the existing `get_anon_id` exactly as-is; add the imports at the top of the file):

Add to the imports at the top:
```python
from dataclasses import dataclass
from auth import verify_token
```

Append below `get_anon_id`:
```python
@dataclass
class Identity:
    """Who is calling. Authenticated when a valid Supabase JWT was presented."""
    user_id: UUID | None = None
    anon_id: UUID | None = None

    @property
    def is_authenticated(self) -> bool:
        return self.user_id is not None

    @property
    def distinct_id(self) -> str:
        """Stable analytics id: the user id when authenticated, else the anon id."""
        return str(self.user_id) if self.user_id else str(self.anon_id)


def get_identity(
    authorization: str | None = Header(None),
    x_anon_id: str | None = Header(None),
) -> Identity:
    """Resolve the caller's identity.

    A valid Bearer token wins (authenticated). Otherwise fall back to the
    anonymous X-Anon-Id header. A Bearer token that is present but invalid is a
    hard 401 (the client should refresh + retry) — we never silently downgrade
    to anonymous. Error codes match the legacy get_anon_id codes so existing
    tests stay green.
    """
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        user_id = verify_token(token)
        if user_id is not None:
            return Identity(user_id=user_id)
        raise HTTPException(
            status_code=401,
            detail={"code": "INVALID_TOKEN",
                    "message": "Your session is invalid or expired. Please sign in again."},
        )
    if x_anon_id:
        try:
            return Identity(anon_id=UUID(x_anon_id))
        except (ValueError, AttributeError):
            raise HTTPException(
                status_code=400,
                detail={"code": "INVALID_ANON_ID",
                        "message": "X-Anon-Id is not a valid UUID."},
            )
    raise HTTPException(
        status_code=400,
        detail={"code": "MISSING_ANON_ID",
                "message": "Authentication or X-Anon-Id header is required."},
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/unit/test_get_identity.py tests/unit/test_deps.py -v`
Expected: all passed (new get_identity tests + existing get_anon_id tests both green).

- [ ] **Step 5: Commit**

```bash
git add src/api/deps.py tests/unit/test_get_identity.py
git commit -m "feat(api): get_identity dependency (Bearer JWT or anon fallback)"
```

---

## Task 4: DB helpers — claim + list (real + fake)

**Files:**
- Modify: `src/api/db.py`
- Modify: `tests/helpers/fake_db.py`

These are exercised end-to-end by the integration tests in Task 6 (the real `SupabaseDB` methods are validated in the live smoke, Task 16 — same pattern as the rest of the thin DB layer).

- [ ] **Step 1: Add real implementations** — append these two methods inside `class SupabaseDB` in `src/api/db.py` (after `count_anon_reports`):

```python
    def claim_anon_reports(self, anon_id: UUID, user_id: UUID) -> int:
        """Reassign an anon's unclaimed reports to a user. Idempotent."""
        res = (self.client.table("reports")
               .update({"user_id": str(user_id), "anon_id": None, "updated_at": "now()"})
               .eq("anon_id", str(anon_id))
               .is_("user_id", "null")
               .execute())
        return len(res.data or [])

    def list_user_reports(self, user_id: UUID) -> list[dict]:
        """Compact summaries for the My Reports page, newest first."""
        res = (self.client.table("reports")
               .select("id, title, report_json, created_at")
               .eq("user_id", str(user_id))
               .order("created_at", desc=True)
               .execute())
        return [_summarize_report_row(r) for r in (res.data or [])]
```

Add this module-level helper near the bottom of `src/api/db.py` (after the class):
```python
def _summarize_report_row(row: dict) -> dict:
    charts = (row.get("report_json") or {}).get("charts", [])
    kinds: list[str] = []
    for c in charts:
        kind = (c.get("spec") or {}).get("kind")
        if kind and kind not in kinds:
            kinds.append(kind)
    return {
        "id": row["id"],
        "title": row.get("title") or "Untitled report",
        "chartCount": len(charts),
        "kinds": kinds,
        "createdAt": row.get("created_at"),
    }
```

> Note: `list_user_reports` returns `kinds` in addition to the spec's `{id, title, chartCount, createdAt}` — the My Reports UI renders kind badges, so the backend extracts the deduped chart kinds server-side (keeps the browser payload small; full `report_json` never leaves the server). Pulling `report_json` per report is acceptable at MVP scale; a denormalized `chart_count`/`kinds` column is a future optimization, out of scope.

- [ ] **Step 2: Mirror in FakeDB** — edit `tests/helpers/fake_db.py`.

First, give rows an insertion sequence so `list_user_reports` can order deterministically. Change `__init__` and `save_report`:

```python
    def __init__(self):
        self._rows: dict[str, dict] = {}   # report_id -> row dict
        self._seq = 0

    def save_report(
        self,
        report_id: str,
        anon_id: Optional[UUID],
        user_id: Optional[UUID],
        report_json: dict,
        csv_storage_key: Optional[str],
        title: str,
    ) -> None:
        self._seq += 1
        self._rows[report_id] = {
            "id": report_id,
            "anon_id": str(anon_id) if anon_id else None,
            "user_id": str(user_id) if user_id else None,
            "report_json": deepcopy(report_json),
            "csv_storage_key": csv_storage_key,
            "title": title,
            "_seq": self._seq,
        }
```

Then append these two methods to `class FakeDB` (after `count_anon_reports`):

```python
    def claim_anon_reports(self, anon_id: UUID, user_id: UUID) -> int:
        n = 0
        for r in self._rows.values():
            if r["anon_id"] == str(anon_id) and r["user_id"] is None:
                r["user_id"] = str(user_id)
                r["anon_id"] = None
                n += 1
        return n

    def list_user_reports(self, user_id: UUID) -> list[dict]:
        rows = [r for r in self._rows.values() if r["user_id"] == str(user_id)]
        rows.sort(key=lambda r: r.get("_seq", 0), reverse=True)
        out = []
        for r in rows:
            charts = (r["report_json"] or {}).get("charts", [])
            kinds: list[str] = []
            for c in charts:
                kind = (c.get("spec") or {}).get("kind")
                if kind and kind not in kinds:
                    kinds.append(kind)
            out.append({
                "id": r["id"],
                "title": r["title"] or "Untitled report",
                "chartCount": len(charts),
                "kinds": kinds,
                "createdAt": r.get("_seq"),
            })
        return out
```

- [ ] **Step 3: Sanity-check FakeDB parity in a REPL** (no committed test here; behaviour is covered by Task 6 integration tests):

Run:
```bash
./venv/bin/python -c "
import sys; sys.path.insert(0, 'tests'); sys.path.insert(0, 'src/api')
from uuid import uuid4
from helpers.fake_db import FakeDB
db = FakeDB(); a=str(uuid4()); u=str(uuid4())
db.save_report('r1', a, None, {'charts':[{'spec':{'kind':'bar'}}]}, 'k', 't')
assert db.claim_anon_reports(__import__('uuid').UUID(a), __import__('uuid').UUID(u)) == 1
assert db.claim_anon_reports(__import__('uuid').UUID(a), __import__('uuid').UUID(u)) == 0
rows = db.list_user_reports(__import__('uuid').UUID(u))
assert rows[0]['chartCount']==1 and rows[0]['kinds']==['bar'], rows
print('FakeDB parity OK')
"
```
Expected: prints `FakeDB parity OK`.

- [ ] **Step 4: Commit**

```bash
git add src/api/db.py tests/helpers/fake_db.py
git commit -m "feat(api): db.claim_anon_reports + db.list_user_reports (+ FakeDB parity)"
```

---

## Task 5: Identity-aware gating in `main.py`

Swap `get_anon_id` → `get_identity` on the three gated endpoints, make Generate-more return 402 for anonymous callers, and update the two anon Generate-more tests to authenticate.

**Files:**
- Modify: `src/api/main.py`
- Modify: `tests/integration/test_api_layout.py`
- Test (new): `tests/integration/test_auth_gating.py`
- Test helper (new): `tests/helpers/fake_auth.py`

- [ ] **Step 1: Add the test helper** — `tests/helpers/fake_auth.py`:

```python
"""Identity builders for integration tests."""
from uuid import UUID

from deps import Identity


def auth_identity(user_id: str) -> Identity:
    return Identity(user_id=UUID(user_id))


def anon_identity(anon_id: str) -> Identity:
    return Identity(anon_id=UUID(anon_id))
```

- [ ] **Step 2: Write the failing gating tests** — `tests/integration/test_auth_gating.py`:

```python
import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from uuid import uuid4

from tests.helpers.fake_claude import FakeClaude, tool_use
from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_storage import FakeStorage
from tests.helpers.fake_posthog import FakePostHog
from tests.helpers.fake_auth import auth_identity, anon_identity


def _csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _ten_chart_fake():
    calls = [tool_use("frequency_bar_chart",
                      {"column": "region", "title": f"T{i}", "intent": f"i{i}"},
                      id_=f"tu_{i}") for i in range(10)]
    return FakeClaude([
        {"tool_calls": calls},
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "S.", "captions": [f"c{i}" for i in range(10)],
                                  "data_quality": []})]},
    ])


class _Holder:
    def __init__(self):
        self.current = None

    def __call__(self):
        return self.current


@pytest.fixture
def ctx(sales):
    fake_db, fake_storage, fake_posthog = FakeDB(), FakeStorage(), FakePostHog()
    holder = _Holder()
    from main import app, get_claude_client, get_db, get_storage, get_posthog, get_identity
    # Fresh FakeClaude per request (dependency override is called per request).
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=_ten_chart_fake())
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[get_storage] = lambda: fake_storage
    app.dependency_overrides[get_posthog] = lambda: fake_posthog
    app.dependency_overrides[get_identity] = holder
    yield TestClient(app), fake_db, fake_posthog, holder
    app.dependency_overrides.clear()


def test_authenticated_user_unlimited_reports(ctx, sales):
    tc, db, _, holder = ctx
    user = str(uuid4())
    holder.current = auth_identity(user)
    for _ in range(3):
        resp = tc.post("/generate-report",
                       files={"file": ("s.csv", _csv_bytes(sales), "text/csv")})
        assert resp.status_code == 200
    rows = list(db._rows.values())
    assert len(rows) == 3
    assert all(r["user_id"] == user and r["anon_id"] is None for r in rows)


def test_anonymous_generate_more_returns_402(ctx, sales):
    tc, db, _, holder = ctx
    holder.current = anon_identity(str(uuid4()))
    resp = tc.post("/generate-report",
                   files={"file": ("s.csv", _csv_bytes(sales), "text/csv")})
    assert resp.status_code == 200
    sid = resp.json()["session_id"]
    resp2 = tc.post(f"/report/{sid}/generate-more")
    assert resp2.status_code == 402
    assert resp2.json()["detail"]["code"] == "UPGRADE_REQUIRED"


def test_authenticated_generate_more_allowed(ctx, sales):
    tc, db, _, holder = ctx
    holder.current = auth_identity(str(uuid4()))
    resp = tc.post("/generate-report",
                   files={"file": ("s.csv", _csv_bytes(sales), "text/csv")})
    sid = resp.json()["session_id"]
    new = FakeClaude([
        {"tool_calls": [tool_use("histogram_chart",
                                 {"column": "revenue", "title": "More", "intent": "n"}, id_="m0")]},
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "U.", "captions": ["c"], "data_quality": []})]},
    ])
    from main import app, get_claude_client
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=new)
    resp2 = tc.post(f"/report/{sid}/generate-more")
    assert resp2.status_code == 200
    assert len(resp2.json()["charts"]) == 11
```

- [ ] **Step 3: Run it to verify it fails**

Run: `./venv/bin/python -m pytest tests/integration/test_auth_gating.py -v`
Expected: FAIL — `test_anonymous_generate_more_returns_402` gets 200 (no gate yet); the others may error on the `get_identity` override / unlimited behaviour. (Confirms the gate isn't there yet.)

- [ ] **Step 4: Edit `main.py` imports** — change the deps import line (currently `from deps import get_anon_id`) to:

```python
from deps import Identity, get_identity
```

And add `Header` to the FastAPI import (used by Task 6). The line becomes:
```python
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
```

- [ ] **Step 5: Rewrite `generate_report`'s signature + gate.**

Change the dependency parameter from `anon_id: UUID = Depends(get_anon_id),` to:
```python
    identity: Identity = Depends(get_identity),
```

Replace the existing anon-limit block (the `bypass = ...` / `existing_count = ...` / `if not bypass ...` lines) with:
```python
    # Identity-aware gating: authenticated users are unlimited; anonymous
    # visitors keep the 1-free-report cap (with an owner/QA allowlist bypass).
    if not identity.is_authenticated:
        anon_id = identity.anon_id
        bypass = str(anon_id) in UNLIMITED_ANON_IDS
        existing_count = db.count_anon_reports(anon_id)
        if not bypass and existing_count >= ANON_REPORT_LIMIT:
            posthog.capture(identity.distinct_id, "anon_limit_blocked", {})
            raise HTTPException(
                status_code=403,
                detail={"code": "ANON_LIMIT_REACHED",
                        "message": "You've used your free report. Sign in to do more."},
            )
```

In the same function, replace every `str(anon_id)` in the `posthog.capture(...)` calls with `identity.distinct_id` (there are several: `report_generation_started`, `claude_overloaded`, the two `report_generation_failed`, and `report_generation_succeeded`).

Update the `db.save_report(...)` call's identity args:
```python
        db.save_report(
            report_id=report_id,
            anon_id=identity.anon_id,
            user_id=identity.user_id,
            report_json=report.model_dump(),
            csv_storage_key=csv_key,
            title=_title_from_summary(report.summary),
        )
```

- [ ] **Step 6: Gate `generate_more`.**

Change its dependency parameter from `anon_id: UUID = Depends(get_anon_id),` to:
```python
    identity: Identity = Depends(get_identity),
```

Insert this as the **first statement** in the function body (before `started = time.perf_counter()`):
```python
    if not identity.is_authenticated:
        raise HTTPException(
            status_code=402,
            detail={"code": "UPGRADE_REQUIRED",
                    "message": "Create a free account to generate more charts."},
        )
```

Replace every `str(anon_id)` in this function's `posthog.capture(...)` calls with `identity.distinct_id` (`generate_more_started`, `generate_more_failed` ×2, `generate_more_succeeded`).

- [ ] **Step 7: Update `export_pdf`.**

Change its dependency parameter from `anon_id: UUID = Depends(get_anon_id),` to:
```python
    identity: Identity = Depends(get_identity),
```
Replace every `str(anon_id)` in its `posthog.capture(...)` calls with `identity.distinct_id` (`pdf_export_started`, `pdf_export_failed`, `pdf_export_succeeded`). No gating change.

- [ ] **Step 8: Authenticate the two anon Generate-more tests** — edit `tests/integration/test_api_layout.py`.

In `test_generate_more_appends_charts`, the block currently reads:
```python
    from main import app, get_claude_client
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=new_fake)
```
Change it to also authenticate:
```python
    from main import app, get_claude_client, get_identity
    from deps import Identity
    from uuid import uuid4 as _uuid4
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=new_fake)
    app.dependency_overrides[get_identity] = lambda: Identity(user_id=_uuid4())
```

In `test_generate_more_unknown_session`, change the body to authenticate before posting:
```python
def test_generate_more_unknown_session(client_with_report):
    tc, _, _, anon, *_ = client_with_report
    from main import app, get_identity
    from deps import Identity
    from uuid import uuid4 as _uuid4
    app.dependency_overrides[get_identity] = lambda: Identity(user_id=_uuid4())
    resp = tc.post("/report/nope/generate-more", headers={"X-Anon-Id": anon})
    assert resp.status_code == 404
```

- [ ] **Step 9: Run the focused suites**

Run: `./venv/bin/python -m pytest tests/integration/test_auth_gating.py tests/integration/test_api_layout.py tests/integration/test_anon_limit.py tests/integration/test_api_errors.py -v`
Expected: all passed — new gating tests green, the 2 updated generate-more tests green, anon-limit + missing-anon-id (`MISSING_ANON_ID`) still green.

- [ ] **Step 10: Run the FULL backend suite (no regressions)**

Run: `./venv/bin/python -m pytest -q`
Expected: all passed (previously-green tests + the new ones; the 2 generate-more tests now authenticated).

- [ ] **Step 11: Commit**

```bash
git add src/api/main.py tests/integration/test_auth_gating.py tests/integration/test_api_layout.py tests/helpers/fake_auth.py
git commit -m "feat(api): identity-aware gating — authed unlimited, anon generate-more 402"
```

---

## Task 6: Migration + dashboard endpoints

**Files:**
- Modify: `src/api/main.py`
- Test (new): `tests/integration/test_claim_and_reports.py`

- [ ] **Step 1: Write the failing tests** — `tests/integration/test_claim_and_reports.py`:

```python
import pytest
from fastapi.testclient import TestClient
from uuid import uuid4

from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_posthog import FakePostHog
from tests.helpers.fake_auth import auth_identity, anon_identity


class _Holder:
    def __init__(self):
        self.current = None

    def __call__(self):
        return self.current


def _report_json(n_charts):
    charts = [{"chart_id": f"c{i}", "spec": {"kind": "bar"}, "caption": f"cap{i}"}
              for i in range(n_charts)]
    return {"generated_at": "2026-01-01T00:00:00Z", "summary": "S",
            "data_quality": [], "charts": charts, "layout": [], "metadata": {}}


@pytest.fixture
def ctx():
    fake_db, fake_posthog = FakeDB(), FakePostHog()
    holder = _Holder()
    from main import app, get_db, get_posthog, get_identity
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[get_posthog] = lambda: fake_posthog
    app.dependency_overrides[get_identity] = holder
    yield TestClient(app), fake_db, holder
    app.dependency_overrides.clear()


def test_claim_moves_anon_reports_to_user(ctx):
    tc, db, holder = ctx
    anon, user = str(uuid4()), str(uuid4())
    db.save_report("r1", anon, None, _report_json(3), "k", "t")
    holder.current = auth_identity(user)
    resp = tc.post("/claim-anon-reports", headers={"X-Anon-Id": anon})
    assert resp.status_code == 200
    assert resp.json()["claimed"] == 1
    row = db.get_report("r1")
    assert row["user_id"] == user and row["anon_id"] is None


def test_claim_is_idempotent(ctx):
    tc, db, holder = ctx
    anon, user = str(uuid4()), str(uuid4())
    db.save_report("r1", anon, None, _report_json(1), "k", "t")
    holder.current = auth_identity(user)
    tc.post("/claim-anon-reports", headers={"X-Anon-Id": anon})
    resp2 = tc.post("/claim-anon-reports", headers={"X-Anon-Id": anon})
    assert resp2.json()["claimed"] == 0


def test_claim_without_anon_header_is_noop(ctx):
    tc, _, holder = ctx
    holder.current = auth_identity(str(uuid4()))
    resp = tc.post("/claim-anon-reports")
    assert resp.status_code == 200
    assert resp.json()["claimed"] == 0


def test_claim_requires_auth(ctx):
    tc, _, holder = ctx
    holder.current = anon_identity(str(uuid4()))
    resp = tc.post("/claim-anon-reports", headers={"X-Anon-Id": str(uuid4())})
    assert resp.status_code == 401


def test_my_reports_isolates_users(ctx):
    tc, db, holder = ctx
    u1, u2 = str(uuid4()), str(uuid4())
    db.save_report("r1", None, u1, _report_json(3), "k", "One")
    db.save_report("r2", None, u2, _report_json(2), "k", "Two")
    holder.current = auth_identity(u1)
    resp = tc.get("/my-reports")
    assert resp.status_code == 200
    body = resp.json()
    assert [r["id"] for r in body] == ["r1"]
    assert body[0]["chartCount"] == 3
    assert body[0]["title"] == "One"
    assert body[0]["kinds"] == ["bar"]


def test_my_reports_requires_auth(ctx):
    tc, _, holder = ctx
    holder.current = anon_identity(str(uuid4()))
    resp = tc.get("/my-reports")
    assert resp.status_code == 401
```

- [ ] **Step 2: Run it to verify it fails**

Run: `./venv/bin/python -m pytest tests/integration/test_claim_and_reports.py -v`
Expected: FAIL — 404s (endpoints don't exist yet).

- [ ] **Step 3: Implement** — add both endpoints to `src/api/main.py` (place them after `patch_report_layout`, before `generate_more`):

```python
@app.post("/claim-anon-reports")
async def claim_anon_reports(
    identity: Identity = Depends(get_identity),
    x_anon_id: str | None = Header(None),
    db: SupabaseDB = Depends(get_db),
    posthog: PostHogServer = Depends(get_posthog),
):
    if not identity.is_authenticated:
        raise HTTPException(status_code=401, detail={
            "code": "AUTH_REQUIRED", "message": "Sign in required."})
    if not x_anon_id:
        return {"claimed": 0}
    try:
        anon_uuid = UUID(x_anon_id)
    except (ValueError, AttributeError):
        return {"claimed": 0}
    claimed = db.claim_anon_reports(anon_uuid, identity.user_id)
    if claimed:
        posthog.capture(identity.distinct_id, "anon_reports_claimed", {"count": claimed})
    return {"claimed": claimed}


@app.get("/my-reports")
async def my_reports(
    identity: Identity = Depends(get_identity),
    db: SupabaseDB = Depends(get_db),
):
    if not identity.is_authenticated:
        raise HTTPException(status_code=401, detail={
            "code": "AUTH_REQUIRED", "message": "Sign in required."})
    return db.list_user_reports(identity.user_id)
```

> Note: `claim_anon_reports` reads `x_anon_id` as its **own** `Header` param (not via `get_identity`), because the caller is authenticated (Bearer) but we still need the anon cookie value to know which rows to migrate.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/integration/test_claim_and_reports.py -v`
Expected: 6 passed.

- [ ] **Step 5: Full backend suite again**

Run: `./venv/bin/python -m pytest -q`
Expected: all passed.

- [ ] **Step 6: Commit**

```bash
git add src/api/main.py tests/integration/test_claim_and_reports.py
git commit -m "feat(api): POST /claim-anon-reports + GET /my-reports"
```

---

## Task 7: Frontend Supabase client + deps + env

**Files:**
- Modify: `package.json`
- Create: `src/app/lib/supabase.ts`
- Modify: `.env` (local dev)

- [ ] **Step 1: Install the deps**

Run: `npm install @supabase/ssr@^0.5.2 @supabase/supabase-js@^2.45.4`
Expected: both added to `package.json` dependencies; lockfile updated.

- [ ] **Step 2: Add the local env var.** Append to `.env` (gitignored — do NOT commit it):
```
NEXT_PUBLIC_SUPABASE_URL=https://xxwtbegkgozufftuhbil.supabase.co
```
(`NEXT_PUBLIC_SUPABASE_ANON_KEY` = the `sb_publishable_...` key is already present.)

- [ ] **Step 3: Create the browser client** — `src/app/lib/supabase.ts`:

```typescript
'use client';
import { createBrowserClient } from '@supabase/ssr';

let _client: ReturnType<typeof createBrowserClient> | null = null;

/** Singleton browser Supabase client (cookie-backed session, auto-refresh). */
export function getSupabaseBrowser() {
  if (!_client) {
    _client = createBrowserClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    );
  }
  return _client;
}

/** Current access token, or null when signed out. Used by apiFetch. */
export async function getAccessToken(): Promise<string | null> {
  try {
    const { data } = await getSupabaseBrowser().auth.getSession();
    return data.session?.access_token ?? null;
  } catch {
    return null;
  }
}
```

- [ ] **Step 4: Typecheck**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 5: Commit** (package.json + lockfile + new file; `.env` is gitignored):

```bash
git add package.json package-lock.json src/app/lib/supabase.ts
git commit -m "feat(web): Supabase browser client + @supabase/ssr deps"
```

---

## Task 8: Inject Bearer token in `apiFetch`

**Files:**
- Modify: `src/app/lib/api.ts`

- [ ] **Step 1: Edit `apiFetch`** in `src/app/lib/api.ts`. Add the supabase import and await the token. Replace the file's top section through the end of `apiFetch` with:

```typescript
'use client';
import { getAnonId } from './anon';
import { getAccessToken } from './supabase';

export interface ApiError extends Error {
  status: number;
  code?: string;
  detail?: unknown;
}

export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers || {});
  const anonId = getAnonId();
  if (anonId) headers.set('X-Anon-Id', anonId);

  const token = await getAccessToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);

  const url = `${process.env.NEXT_PUBLIC_API_URL}${path}`;
  return fetch(url, { ...init, headers });
}
```

(`apiJSON` below it is unchanged.)

- [ ] **Step 2: Typecheck**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/app/lib/api.ts
git commit -m "feat(web): apiFetch injects Supabase Bearer token when signed in"
```

---

## Task 9: Auth callback route (exchange + claim + redirect)

**Files:**
- Create: `src/app/auth/callback/route.ts`

- [ ] **Step 1: Implement** — `src/app/auth/callback/route.ts`:

```typescript
import { NextRequest, NextResponse } from 'next/server';
import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';

export const dynamic = 'force-dynamic';

/** Only allow internal relative paths as redirect targets (no open redirect). */
function safeNext(raw: string | null): string {
  if (raw && raw.startsWith('/') && !raw.startsWith('//')) return raw;
  return '/';
}

export async function GET(request: NextRequest) {
  const { searchParams, origin } = new URL(request.url);
  const code = searchParams.get('code');
  const next = safeNext(searchParams.get('next'));

  if (code) {
    const cookieStore = cookies();
    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          getAll() {
            return cookieStore.getAll();
          },
          setAll(cookiesToSet) {
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options),
            );
          },
        },
      },
    );

    const { data } = await supabase.auth.exchangeCodeForSession(code);

    // Migrate the anonymous visitor's existing report(s) into the new account.
    // Idempotent on the backend, so a failure here is non-fatal.
    const token = data.session?.access_token;
    const anon = cookieStore.get('chartsage_anon')?.value;
    if (token && anon) {
      try {
        await fetch(`${process.env.NEXT_PUBLIC_API_URL}/claim-anon-reports`, {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}`, 'X-Anon-Id': anon },
        });
      } catch {
        /* non-fatal */
      }
    }

    return NextResponse.redirect(`${origin}/welcome?next=${encodeURIComponent(next)}`);
  }

  // No code present — OAuth was cancelled or the link expired.
  return NextResponse.redirect(`${origin}/login`);
}
```

- [ ] **Step 2: Typecheck**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/app/auth/callback/route.ts
git commit -m "feat(web): /auth/callback — code exchange + anon-report claim"
```

---

## Task 10: SessionWatcher + AuthNav + layout mount

**Files:**
- Create: `src/app/components/SessionWatcher.tsx`
- Create: `src/app/components/AuthNav.tsx`
- Modify: `src/app/layout.tsx`

- [ ] **Step 1: SessionWatcher** — `src/app/components/SessionWatcher.tsx`. On a real sign-in, identify the user in PostHog (merging the prior anonymous person); on sign-out, reset.

```typescript
'use client';
import { useEffect } from 'react';
import { getSupabaseBrowser } from '../lib/supabase';
import { initPostHog, posthog } from '../lib/posthog';

export default function SessionWatcher() {
  useEffect(() => {
    initPostHog(); // idempotent — guarantees posthog is ready before identify()
    const supabase = getSupabaseBrowser();
    const { data: sub } = supabase.auth.onAuthStateChange((event, session) => {
      if (session?.user) {
        posthog.identify?.(session.user.id, { email: session.user.email });
      }
      if (event === 'SIGNED_IN') {
        posthog.capture?.('logged_in', {
          method: session?.user?.app_metadata?.provider,
        });
      }
      if (event === 'SIGNED_OUT') {
        posthog.reset?.();
      }
    });
    return () => sub.subscription.unsubscribe();
  }, []);
  return null;
}
```

> Note on events: `identify` is safe to call on every load (idempotent, stitches anon→user). We gate the `logged_in` *event* on `event === 'SIGNED_IN'` (a genuine sign-in) rather than `INITIAL_SESSION` (page load with an existing session) to avoid inflating counts. We deliberately do **not** emit a separate `signed_up` event — `onboarding_viewed` (Task 12) is the new-user proxy.

- [ ] **Step 2: AuthNav** — `src/app/components/AuthNav.tsx`:

```typescript
'use client';
import { useEffect, useState } from 'react';
import { getSupabaseBrowser } from '../lib/supabase';
import { posthog } from '../lib/posthog';

export default function AuthNav() {
  const [email, setEmail] = useState<string | null>(null);

  useEffect(() => {
    const supabase = getSupabaseBrowser();
    supabase.auth.getSession().then(({ data }) =>
      setEmail(data.session?.user?.email ?? null),
    );
    const { data: sub } = supabase.auth.onAuthStateChange((_e, session) =>
      setEmail(session?.user?.email ?? null),
    );
    return () => sub.subscription.unsubscribe();
  }, []);

  async function signOut() {
    posthog.capture?.('signed_out', {});
    await getSupabaseBrowser().auth.signOut();
    window.location.href = '/';
  }

  return (
    <nav className="fixed top-3 right-4 z-50 flex items-center gap-3 text-sm">
      {email ? (
        <>
          <a
            href="/reports"
            className="px-3 py-1.5 rounded-lg bg-white/90 ring-1 ring-stone-200 text-stone-700 hover:bg-white"
          >
            My reports
          </a>
          <span className="hidden sm:inline text-stone-400 max-w-[160px] truncate">{email}</span>
          <button
            type="button"
            onClick={signOut}
            className="px-3 py-1.5 rounded-lg text-stone-500 hover:text-stone-900"
          >
            Sign out
          </button>
        </>
      ) : (
        <a
          href="/login"
          className="px-3 py-1.5 rounded-lg bg-stone-900 text-white hover:bg-stone-800"
        >
          Sign in
        </a>
      )}
    </nav>
  );
}
```

- [ ] **Step 3: Mount both in layout** — edit `src/app/layout.tsx`. Add the imports and render `<SessionWatcher />` + `<AuthNav />` inside `<body>`:

```typescript
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import PostHogInit from './PostHogInit'
import SessionWatcher from './components/SessionWatcher'
import AuthNav from './components/AuthNav'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'ChartSage - AI-Powered Data Visualization',
  description: 'Turn any Excel file into a beautiful, interactive dashboard with AI-generated insights in seconds.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <PostHogInit />
        <SessionWatcher />
        <AuthNav />
        <main className="min-h-screen bg-gray-50">
          {children}
        </main>
      </body>
    </html>
  )
}
```

(The nav is `fixed`, so it overlays without reflowing existing pages.)

- [ ] **Step 4: Typecheck**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/app/components/SessionWatcher.tsx src/app/components/AuthNav.tsx src/app/layout.tsx
git commit -m "feat(web): SessionWatcher (PostHog identify) + AuthNav in layout"
```

---

## Task 11: Login page

**Files:**
- Create: `src/app/login/page.tsx`

- [ ] **Step 1: Implement** — `src/app/login/page.tsx`. Reads an optional `next` query param and threads it through the callback.

```typescript
'use client';
import { useState } from 'react';
import { getSupabaseBrowser } from '../lib/supabase';
import { posthog } from '../lib/posthog';

function callbackUrl(): string {
  const params = new URLSearchParams(window.location.search);
  const next = params.get('next') || '/';
  const safe = next.startsWith('/') && !next.startsWith('//') ? next : '/';
  return `${window.location.origin}/auth/callback?next=${encodeURIComponent(safe)}`;
}

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function google() {
    posthog.capture?.('login_method_selected', { method: 'google' });
    await getSupabaseBrowser().auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo: callbackUrl() },
    });
  }

  async function magicLink(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    posthog.capture?.('login_method_selected', { method: 'magic_link' });
    const { error } = await getSupabaseBrowser().auth.signInWithOtp({
      email,
      options: { emailRedirectTo: callbackUrl() },
    });
    if (error) setError(error.message);
    else setSent(true);
  }

  return (
    <div className="min-h-screen bg-stone-50 flex items-center justify-center px-4">
      <div className="w-full max-w-sm">
        <p className="text-xs uppercase tracking-widest text-stone-400 mb-2">ChartSage</p>
        <h1 className="text-3xl font-semibold tracking-tight text-stone-900 mb-6">Sign in</h1>

        <button
          type="button"
          onClick={google}
          className="w-full px-4 py-2.5 bg-white ring-1 ring-stone-300 rounded-lg text-stone-800 font-medium hover:bg-stone-50 transition-colors"
        >
          Continue with Google
        </button>

        <div className="my-5 flex items-center gap-3 text-xs text-stone-400">
          <div className="flex-1 h-px bg-stone-200" /> or <div className="flex-1 h-px bg-stone-200" />
        </div>

        {sent ? (
          <p className="text-sm text-stone-600">
            Check your inbox — we sent a magic link to <strong>{email}</strong>.
          </p>
        ) : (
          <form onSubmit={magicLink} className="space-y-3">
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full px-3 py-2.5 rounded-lg ring-1 ring-stone-300 focus:ring-2 focus:ring-teal-500 outline-none text-stone-900"
            />
            <button
              type="submit"
              className="w-full px-4 py-2.5 bg-stone-900 text-white rounded-lg font-medium hover:bg-stone-800 transition-colors"
            >
              Email me a magic link
            </button>
            {error && <p className="text-sm text-red-600">{error}</p>}
          </form>
        )}

        <p className="mt-6 text-sm text-stone-400 text-center">
          <a href="/" className="hover:text-stone-700">← Back to home</a>
        </p>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Typecheck**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/app/login/page.tsx
git commit -m "feat(web): login page (Google + magic link)"
```

---

## Task 12: Welcome / onboarding page

**Files:**
- Create: `src/app/welcome/page.tsx`

- [ ] **Step 1: Implement** — `src/app/welcome/page.tsx`. Client-side first-visit gate: if already onboarded, bounce straight to `next`.

```typescript
'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { posthog } from '../lib/posthog';

const FLAG = 'chartsage_onboarded';

function safeNext(raw: string | null): string {
  if (raw && raw.startsWith('/') && !raw.startsWith('//')) return raw;
  return '/';
}

const UNLOCKS = [
  { title: 'Unlimited reports', body: 'Turn as many CSVs into dashboards as you like.' },
  { title: 'Generate more charts', body: 'Ask for extra angles on any report, on demand.' },
  { title: 'Saved & revisitable', body: 'Every report is kept in My Reports for later.' },
];

export default function WelcomePage() {
  const router = useRouter();
  const [ready, setReady] = useState(false);
  const [next, setNext] = useState('/');

  useEffect(() => {
    const dest = safeNext(new URLSearchParams(window.location.search).get('next'));
    setNext(dest);
    if (localStorage.getItem(FLAG)) {
      router.replace(dest);
      return;
    }
    posthog.capture?.('onboarding_viewed', {});
    setReady(true);
  }, [router]);

  function finish() {
    localStorage.setItem(FLAG, '1');
    posthog.capture?.('onboarding_completed', {});
    router.replace(next);
  }

  if (!ready) return null;

  return (
    <div className="min-h-screen bg-stone-50 flex items-center justify-center px-4">
      <div className="w-full max-w-lg">
        <p className="text-xs uppercase tracking-widest text-stone-400 mb-2">Welcome to ChartSage</p>
        <h1 className="text-3xl font-semibold tracking-tight text-stone-900 mb-6">
          You're in. Here's what your account unlocks.
        </h1>
        <ul className="space-y-4 mb-8">
          {UNLOCKS.map((u) => (
            <li key={u.title} className="flex gap-3">
              <span className="mt-1 h-2 w-2 rounded-full bg-teal-500 flex-shrink-0" />
              <div>
                <p className="font-medium text-stone-900">{u.title}</p>
                <p className="text-sm text-stone-600">{u.body}</p>
              </div>
            </li>
          ))}
        </ul>
        <button
          type="button"
          onClick={finish}
          className="px-5 py-2.5 bg-stone-900 text-white text-sm font-medium rounded-lg hover:bg-stone-800 transition-colors"
        >
          Get started →
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Typecheck**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/app/welcome/page.tsx
git commit -m "feat(web): onboarding/welcome page with client-side first-visit gate"
```

---

## Task 13: My Reports page

**Files:**
- Create: `src/app/reports/page.tsx`

- [ ] **Step 1: Implement** — `src/app/reports/page.tsx`:

```typescript
'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { apiFetch } from '../lib/api';
import { getSupabaseBrowser } from '../lib/supabase';

interface ReportRow {
  id: string;
  title: string;
  chartCount: number;
  kinds: string[];
  createdAt: string | null;
}

export default function MyReportsPage() {
  const router = useRouter();
  const [reports, setReports] = useState<ReportRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      const { data } = await getSupabaseBrowser().auth.getSession();
      if (!data.session) {
        router.replace('/login?next=/reports');
        return;
      }
      const res = await apiFetch('/my-reports');
      if (res.status === 401) {
        router.replace('/login?next=/reports');
        return;
      }
      if (!res.ok) {
        setError('Could not load your reports.');
        return;
      }
      setReports(await res.json());
    })();
  }, [router]);

  if (error) {
    return (
      <div className="min-h-screen bg-stone-50 flex items-center justify-center">
        <p className="text-stone-600">{error}</p>
      </div>
    );
  }
  if (!reports) {
    return (
      <div className="min-h-screen bg-stone-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-2 border-stone-300 border-t-stone-900" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-stone-50">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-16">
        <header className="mb-8 flex items-baseline justify-between">
          <h1 className="text-3xl font-semibold tracking-tight text-stone-900">My reports</h1>
          <a href="/" className="text-sm text-stone-500 hover:text-stone-900">New report →</a>
        </header>

        {reports.length === 0 ? (
          <div className="p-8 bg-white border border-stone-200 rounded-2xl text-center">
            <p className="text-stone-600 mb-4">You haven't created any reports yet.</p>
            <a href="/" className="inline-block px-5 py-2.5 bg-stone-900 text-white text-sm rounded-lg hover:bg-stone-800">
              Create your first report
            </a>
          </div>
        ) : (
          <ul className="space-y-3">
            {reports.map((r) => (
              <li key={r.id}>
                <a
                  href={`/report/${r.id}`}
                  className="block p-5 bg-white border border-stone-200 rounded-2xl hover:border-stone-300 hover:shadow-sm transition-all"
                >
                  <div className="flex items-baseline justify-between gap-4">
                    <p className="font-medium text-stone-900 truncate">{r.title}</p>
                    <span className="text-xs text-stone-400 flex-shrink-0">
                      {r.createdAt ? new Date(r.createdAt).toLocaleDateString() : ''}
                    </span>
                  </div>
                  <div className="mt-2 flex items-center gap-2 flex-wrap">
                    <span className="text-xs text-stone-500">{r.chartCount} charts</span>
                    {r.kinds.slice(0, 5).map((k) => (
                      <span key={k} className="text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-stone-100 text-stone-500">
                        {k}
                      </span>
                    ))}
                  </div>
                </a>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Typecheck**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/app/reports/page.tsx
git commit -m "feat(web): My Reports dashboard"
```

---

## Task 14: Upsell modal + Toolbar wiring

**Files:**
- Create: `src/app/components/UpsellModal.tsx`
- Modify: `src/app/report/[id]/Toolbar.tsx`

- [ ] **Step 1: UpsellModal** — `src/app/components/UpsellModal.tsx`. Auth returns the user to the current report via `next`.

```typescript
'use client';
import { useState } from 'react';
import { getSupabaseBrowser } from '../lib/supabase';
import { posthog } from '../lib/posthog';

export default function UpsellModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [email, setEmail] = useState('');
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!open) return null;

  const redirectTo = `${window.location.origin}/auth/callback?next=${encodeURIComponent(window.location.pathname)}`;

  async function google() {
    posthog.capture?.('generate_more_upsell_cta', { method: 'google' });
    await getSupabaseBrowser().auth.signInWithOAuth({ provider: 'google', options: { redirectTo } });
  }
  async function magic(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    posthog.capture?.('generate_more_upsell_cta', { method: 'magic_link' });
    const { error } = await getSupabaseBrowser().auth.signInWithOtp({ email, options: { emailRedirectTo: redirectTo } });
    if (error) setError(error.message);
    else setSent(true);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-stone-900/40 px-4" onClick={onClose}>
      <div className="w-full max-w-sm bg-white rounded-2xl p-6 shadow-xl" onClick={(e) => e.stopPropagation()}>
        <h2 className="text-lg font-semibold text-stone-900 mb-1">Create a free account</h2>
        <p className="text-sm text-stone-600 mb-5">
          Sign in to generate more charts — your current report comes with you.
        </p>

        <button
          type="button"
          onClick={google}
          className="w-full px-4 py-2.5 bg-white ring-1 ring-stone-300 rounded-lg text-stone-800 font-medium hover:bg-stone-50"
        >
          Continue with Google
        </button>

        <div className="my-4 flex items-center gap-3 text-xs text-stone-400">
          <div className="flex-1 h-px bg-stone-200" /> or <div className="flex-1 h-px bg-stone-200" />
        </div>

        {sent ? (
          <p className="text-sm text-stone-600">Check your inbox for a magic link.</p>
        ) : (
          <form onSubmit={magic} className="space-y-3">
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full px-3 py-2.5 rounded-lg ring-1 ring-stone-300 focus:ring-2 focus:ring-teal-500 outline-none text-stone-900"
            />
            <button type="submit" className="w-full px-4 py-2.5 bg-stone-900 text-white rounded-lg font-medium hover:bg-stone-800">
              Email me a magic link
            </button>
            {error && <p className="text-sm text-red-600">{error}</p>}
          </form>
        )}

        <button type="button" onClick={onClose} className="mt-4 w-full text-sm text-stone-400 hover:text-stone-700">
          Maybe later
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Wire Toolbar** — edit `src/app/report/[id]/Toolbar.tsx`.

Add the import (top of file, after the existing imports):
```typescript
import UpsellModal from '../../components/UpsellModal';
```

Add modal state (after the existing `useState` lines):
```typescript
  const [showUpsell, setShowUpsell] = useState(false);
```

In `handleGenerateMore`, add a 402 branch immediately after the `res.status === 503` block:
```typescript
      if (res.status === 402) {
        posthog.capture?.('generate_more_upsell_shown', { reportId: sessionId });
        setShowUpsell(true);
        return;
      }
```

Render the modal — replace the component's entire `return ( ... );` with this (the toolbar markup is the existing markup, now wrapped in a fragment with the modal appended):
```typescript
  return (
    <>
      <div className="sticky top-0 z-10 -mx-4 sm:-mx-6 lg:-mx-8 px-4 sm:px-6 lg:px-8 py-3 mb-6 bg-stone-50/90 backdrop-blur border-b border-stone-200 flex items-center justify-end gap-3">
        {error && <span className="text-sm text-red-600 mr-auto">{error}</span>}
        <button
          type="button"
          onClick={handleGenerateMore}
          disabled={generating}
          className="px-4 py-2 text-sm font-medium text-stone-700 bg-white ring-1 ring-stone-200 rounded-lg hover:bg-stone-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {generating ? 'Generating…' : 'Generate 5 more'}
        </button>
        <button
          type="button"
          onClick={handleExportPdf}
          disabled={exporting}
          className="px-4 py-2 text-sm font-medium text-white bg-stone-900 rounded-lg hover:bg-stone-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {exporting ? 'Preparing PDF…' : 'Export PDF'}
        </button>
      </div>
      <UpsellModal open={showUpsell} onClose={() => setShowUpsell(false)} />
    </>
  );
```

- [ ] **Step 3: Typecheck**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add src/app/components/UpsellModal.tsx src/app/report/[id]/Toolbar.tsx
git commit -m "feat(web): generate-more upsell modal (402 → sign-in, returns to report)"
```

---

## Task 15: Real sign-in link on anon-limit page

**Files:**
- Modify: `src/app/anon-limit/page.tsx`

- [ ] **Step 1: Replace the disabled button** in `src/app/anon-limit/page.tsx` with a real link to `/login`:

```typescript
        <a
          href="/login?next=/"
          onClick={() => posthog.capture?.('signin_cta_clicked', { from: 'anonLimit' })}
          className="inline-block px-5 py-2.5 bg-stone-900 text-white text-sm font-medium rounded-lg hover:bg-stone-800 transition-colors"
        >
          Sign in
        </a>
```

(Remove the old `<button disabled ...>Sign in · coming soon</button>`.) Also update the body copy line — change "Accounts are coming soon." to "It only takes a few seconds.".

- [ ] **Step 2: Typecheck**

Run: `npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add src/app/anon-limit/page.tsx
git commit -m "feat(web): anon-limit page links to real /login"
```

---

## Task 16: Provisioning, deploy & live smoke (user-driven runbook)

This task is **executed with the user** — it needs dashboard clicks and a Google OAuth client. Present these steps, wait for confirmation at each external step, then deploy and smoke-test.

**Files:**
- Modify: `README.md` (document the SP2 auth setup at the end)

- [ ] **Step 1: Full local build gate** (before any deploy)

Run: `npm run build && ./venv/bin/python -m pytest -q`
Expected: Next build succeeds; full backend suite passes.

- [ ] **Step 2: Google OAuth client (user, Google Cloud Console)**
  - APIs & Services → OAuth consent screen → configure (External; app name; support email). 
  - APIs & Services → Credentials → Create credentials → OAuth client ID → Web application.
  - **Authorized redirect URI:** `https://xxwtbegkgozufftuhbil.supabase.co/auth/v1/callback`
  - Copy the **Client ID** and **Client secret**.

- [ ] **Step 3: Enable the provider (user, Supabase dashboard)**
  - Authentication → Providers → **Google** → enable → paste Client ID + Client secret → save.
  - Magic link (Email provider) is on by default — confirm it's enabled.

- [ ] **Step 4: URL configuration (user, Supabase dashboard)**
  - Authentication → URL Configuration → **Site URL:** `https://chartsage-xi.vercel.app`
  - **Redirect URLs (add both):** `https://chartsage-xi.vercel.app/auth/callback` and `http://localhost:3000/auth/callback`

- [ ] **Step 5: Confirm asymmetric JWTs (the JWKS safeguard)**

Run: `curl -s https://xxwtbegkgozufftuhbil.supabase.co/auth/v1/.well-known/jwks.json`
Expected: a JSON object with a non-empty `keys` array (e.g. `"alg":"ES256"`).
  - **If `keys` is empty / 404** → the project still signs with symmetric HS256. Contingency: in Supabase dashboard, Authentication → JWT settings, migrate to asymmetric keys (recommended), OR add the JWT secret to Cloud Run (`gcloud secrets create supabase-jwt-secret`) and extend `auth.verify_token` to fall back to `jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated")`. Re-run this step until `keys` is populated (preferred path).

- [ ] **Step 6: Enable RLS (user, Supabase SQL editor)** — run:

```sql
ALTER TABLE reports ENABLE ROW LEVEL SECURITY;

CREATE POLICY reports_owner_select ON reports
  FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY reports_owner_update ON reports
  FOR UPDATE USING (auth.uid() = user_id);
```
(The backend uses the service-role key, which bypasses RLS, so existing flows are unaffected.)

- [ ] **Step 7: Vercel env var (user, Vercel dashboard)**
  - Project → Settings → Environment Variables → add `NEXT_PUBLIC_SUPABASE_URL = https://xxwtbegkgozufftuhbil.supabase.co` for Production, Preview, Development.
  - Confirm `NEXT_PUBLIC_SUPABASE_ANON_KEY` (the `sb_publishable_...` key) and `NEXT_PUBLIC_API_URL` are already present.

- [ ] **Step 8: Deploy backend** (Claude)

Run: `gcloud builds submit --substitutions=_TAG=$(git rev-parse --short HEAD),_SUPABASE_URL=https://xxwtbegkgozufftuhbil.supabase.co,_FRONTEND_BASE_URL=https://chartsage-xi.vercel.app`
Expected: build + deploy succeed; new Cloud Run revision serving. (No new env vars/secrets — `SUPABASE_URL` is already set; JWKS is public.)

- [ ] **Step 9: Deploy frontend** (Claude) — push to `main`; Vercel auto-deploys.

```bash
git push origin main
```
Expected: Vercel build succeeds with the new `NEXT_PUBLIC_SUPABASE_URL`.

- [ ] **Step 10: Live smoke (incognito)**
  - Visit the site → upload a CSV → generate the 1 free report (anon).
  - Click **Generate 5 more** → upsell modal appears (anon → 402).
  - Sign in via **Google**; then in a second incognito, sign in via **magic link**.
  - Land on `/welcome` (onboarding) → Get started.
  - The previously-anonymous report now appears in **My Reports** (claim worked).
  - Open it → **Generate 5 more** now appends charts (authenticated → allowed).
  - **Sign out** → confirm anon gating returns (Generate-more upsell again).
  - In Supabase → Table editor → `reports`: the claimed row has `user_id` set, `anon_id` NULL.
  - In PostHog: `logged_in`, `onboarding_viewed`, `anon_reports_claimed`, `generate_more_upsell_shown` events present.

- [ ] **Step 11: Document + commit** — append an "SP2 — Accounts" section to `README.md` (auth flow, the provisioning steps above, the JWKS safeguard note). Then:

```bash
git add README.md
git commit -m "docs: SP2 accounts setup (Supabase Auth, Google OAuth, RLS, env)"
git push origin main
```

---

## Definition of done

- Backend: full `pytest -q` green (existing tests + Task 2/3/5/6 additions; the 2 generate-more tests now authenticate). Authenticated callers are unlimited; anonymous Generate-more returns `402 UPGRADE_REQUIRED`; `/claim-anon-reports` and `/my-reports` behave per tests.
- Frontend: `npm run build` clean. Login (Google + magic link), onboarding-once, anon→account report migration, generate-more upsell for anon, My Reports list + reopen, sign out — all verified in the live smoke.
- Supabase: Google provider enabled, redirect URLs set, RLS on `reports`. Vercel has `NEXT_PUBLIC_SUPABASE_URL`. Both services deployed.
- No secrets committed; `.env` remains gitignored.
