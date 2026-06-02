# Admin Console v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A minimal, secure internal admin console to find accounts, view a credit balance + recent history, and grant credits — then grant `cj.silver@me.com` 1000 credits.

**Architecture:** A shared-secret `X-Admin-Token` guard (`require_admin`) protects three new `/admin` FastAPI endpoints (search / detail / grant) that reuse the existing service-role DB helpers + the atomic `grant_credits` RPC. A gated `/admin` Next.js route (its own `adminFetch` that sends the token, not the Supabase Bearer) provides search → view → grant. Grants emit an `admin_credit_grant` PostHog event on top of the durable `credit_transactions` ledger row.

**Tech Stack:** FastAPI + supabase-py (service-role, GoTrue admin API) on Cloud Run · Next.js 14 App Router on Vercel · PostHog · Google Secret Manager.

**Branch:** `admin-console` (already created off `main`, carries the spec commit). **Forbid subagents from running `git checkout`/`switch`/`reset`/`stash`.**

---

## File Structure

- `src/api/deps.py` (modify) — add `require_admin` dependency (reads `ADMIN_API_TOKEN` from env at call time; constant-time compare; fail-closed 403).
- `src/api/db.py` (modify) — add `search_accounts(query, limit)`, `get_account_detail(user_id)`, and a private `_all_balances()` / `_list_auth_users()` helper, over the service-role client + GoTrue admin API.
- `src/api/main.py` (modify) — add `GrantIn` model + three `/admin` endpoints behind `Depends(require_admin)`; emit `admin_credit_grant`.
- `cloudbuild.yaml` (modify) — add `ADMIN_API_TOKEN=admin-api-token:latest` to `--set-secrets`.
- `tests/helpers/fake_db.py` (modify) — fake user directory (`_users`) + `add_user`, `search_accounts`, `get_account_detail`.
- `tests/unit/test_require_admin.py` (create) — the guard.
- `tests/integration/test_admin_console.py` (create) — endpoints (search/detail/grant, gating, validation).
- `src/app/lib/adminApi.ts` (create) — `getAdminToken`/`setAdminToken`/`clearAdminToken` + `adminFetch`.
- `src/app/admin/layout.tsx` (create) — server layout exporting `metadata` with `robots: noindex`.
- `src/app/admin/page.tsx` (create) — the console UI (token gate → search → detail → grant).

---

## Phase 1 — Backend (TDD)

### Task 1: `require_admin` guard

**Files:**
- Modify: `src/api/deps.py`
- Test: `tests/unit/test_require_admin.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_require_admin.py`:

```python
import pytest
from fastapi import HTTPException

from deps import require_admin


def test_rejects_when_no_token(monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret-abc")
    with pytest.raises(HTTPException) as e:
        require_admin(None)
    assert e.value.status_code == 403
    assert e.value.detail["code"] == "FORBIDDEN"


def test_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret-abc")
    with pytest.raises(HTTPException) as e:
        require_admin("nope")
    assert e.value.status_code == 403


def test_accepts_correct_token(monkeypatch):
    monkeypatch.setenv("ADMIN_API_TOKEN", "secret-abc")
    assert require_admin("secret-abc") is None  # no raise


def test_fail_closed_when_env_unset(monkeypatch):
    monkeypatch.delenv("ADMIN_API_TOKEN", raising=False)
    with pytest.raises(HTTPException) as e:
        require_admin("anything")
    assert e.value.status_code == 403
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/chrissilver/Documents/ChartSage && venv/bin/python -m pytest tests/unit/test_require_admin.py -q`
Expected: FAIL with `ImportError: cannot import name 'require_admin' from 'deps'`.

- [ ] **Step 3: Implement `require_admin`**

In `src/api/deps.py`, add at the top with the other imports:

```python
import hmac
import os
```

Append this function (after `get_identity`):

```python
def require_admin(x_admin_token: str | None = Header(None)) -> None:
    """Gate admin endpoints on a shared secret.

    Reads ADMIN_API_TOKEN from the environment at call time (so it is test- and
    rotation-friendly) and compares it to the X-Admin-Token header with a
    constant-time compare. Fail-closed: if the env var is unset/empty or the
    header is missing or mismatched, 403. The token is never logged.
    """
    expected = os.environ.get("ADMIN_API_TOKEN", "")
    if not expected or not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(
            status_code=403,
            detail={"code": "FORBIDDEN", "message": "Admin access required."},
        )
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/chrissilver/Documents/ChartSage && venv/bin/python -m pytest tests/unit/test_require_admin.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/api/deps.py tests/unit/test_require_admin.py
git commit -m "feat(admin): require_admin shared-token guard (fail-closed, constant-time)"
```

---

### Task 2: DB admin helpers + FakeDB

**Files:**
- Modify: `src/api/db.py`
- Modify: `tests/helpers/fake_db.py`
- Test: `tests/integration/test_admin_db.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_admin_db.py`:

```python
from tests.helpers.fake_db import FakeDB


def _seed():
    db = FakeDB()
    db.add_user("11111111-1111-1111-1111-111111111111", "alice@example.com")
    db.add_user("22222222-2222-2222-2222-222222222222", "bob@test.io")
    db.ensure_profile("11111111-1111-1111-1111-111111111111", 300)
    return db


def test_search_filters_by_email_substring():
    db = _seed()
    res = db.search_accounts("alice", 50)
    assert len(res) == 1
    assert res[0]["email"] == "alice@example.com"
    assert res[0]["credits_balance"] == 300
    assert res[0]["user_id"] == "11111111-1111-1111-1111-111111111111"


def test_search_empty_query_returns_all_capped():
    db = _seed()
    assert len(db.search_accounts("", 50)) == 2
    assert len(db.search_accounts("", 1)) == 1


def test_account_detail_has_balance_and_txns():
    db = _seed()
    d = db.get_account_detail("11111111-1111-1111-1111-111111111111")
    assert d["credits_balance"] == 300
    assert d["email"] == "alice@example.com"
    assert any(t["reason"] == "signup_grant" for t in d["transactions"])


def test_account_detail_unknown_user_is_none():
    db = _seed()
    assert db.get_account_detail("99999999-9999-9999-9999-999999999999") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/chrissilver/Documents/ChartSage && venv/bin/python -m pytest tests/integration/test_admin_db.py -q`
Expected: FAIL with `AttributeError: 'FakeDB' object has no attribute 'add_user'`.

- [ ] **Step 3: Add the FakeDB helpers**

In `tests/helpers/fake_db.py`, add `self._users: list[dict] = []` in `__init__` (next to `self._profiles`), then add these methods (after `record_upgrade_intent`):

```python
    # --- admin (fake user directory) ---
    def add_user(self, user_id, email, created_at: str = "2026-01-01T00:00:00Z") -> None:
        self._users.append({"id": str(user_id), "email": email, "created_at": created_at})

    def _find_user(self, user_id):
        return next((u for u in self._users if u["id"] == str(user_id)), None)

    def search_accounts(self, query: str, limit: int = 50) -> list[dict]:
        q = (query or "").strip().lower()
        out = []
        for u in self._users:
            if q and q not in (u["email"] or "").lower():
                continue
            out.append({
                "user_id": u["id"],
                "email": u["email"],
                "credits_balance": self.get_balance(u["id"]),
                "created_at": u["created_at"],
            })
            if len(out) >= limit:
                break
        return out

    def get_account_detail(self, user_id) -> dict | None:
        u = self._find_user(user_id)
        if u is None:
            return None
        return {
            "user_id": u["id"],
            "email": u["email"],
            "credits_balance": self.get_balance(u["id"]),
            "transactions": self.list_transactions(u["id"]),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/chrissilver/Documents/ChartSage && venv/bin/python -m pytest tests/integration/test_admin_db.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Implement the real `SupabaseDB` helpers**

In `src/api/db.py`, add these methods to the `SupabaseDB` class (after `list_transactions`). They use the service-role client (which bypasses RLS) and the GoTrue admin API. `list_users()` returns an iterable of `User` objects (`.id`, `.email`, `.created_at`); we handle either a bare list or an object exposing `.users`:

```python
    # --- admin ---
    def _list_auth_users(self, cap_pages: int = 20, per_page: int = 200) -> list:
        """All auth users via the GoTrue admin API, paginated up to a cap."""
        users: list = []
        page = 1
        while page <= cap_pages:
            resp = self.client.auth.admin.list_users(page=page, per_page=per_page)
            batch = resp if isinstance(resp, list) else getattr(resp, "users", []) or []
            if not batch:
                break
            users.extend(batch)
            if len(batch) < per_page:
                break
            page += 1
        return users

    def _all_balances(self) -> dict:
        res = self.client.table("profiles").select("user_id, credits_balance").execute()
        return {r["user_id"]: r["credits_balance"] for r in (res.data or [])}

    def search_accounts(self, query: str, limit: int = 50) -> list[dict]:
        q = (query or "").strip().lower()
        balances = self._all_balances()
        out: list[dict] = []
        for u in self._list_auth_users():
            email = getattr(u, "email", None) or ""
            if q and q not in email.lower():
                continue
            uid = str(getattr(u, "id", "") or "")
            created = getattr(u, "created_at", None)
            out.append({
                "user_id": uid,
                "email": email,
                "credits_balance": int(balances.get(uid, 0)),
                "created_at": created.isoformat() if hasattr(created, "isoformat") else created,
            })
            if len(out) >= limit:
                break
        return out

    def get_account_detail(self, user_id) -> dict | None:
        uid = str(user_id)
        try:
            resp = self.client.auth.admin.get_user_by_id(uid)
            user = getattr(resp, "user", None) or resp
        except Exception:
            user = None
        if user is None or getattr(user, "id", None) is None:
            return None
        return {
            "user_id": uid,
            "email": getattr(user, "email", None),
            "credits_balance": self.get_balance(uid),
            "transactions": self.list_transactions(uid),
        }
```

- [ ] **Step 6: Run the full suite (no regressions) + commit**

Run: `cd /Users/chrissilver/Documents/ChartSage && venv/bin/python -m pytest tests/unit tests/integration -p no:warnings -q`
Expected: all pass (prior count + 4 new).

```bash
git add src/api/db.py tests/helpers/fake_db.py tests/integration/test_admin_db.py
git commit -m "feat(admin): search_accounts + get_account_detail DB helpers (+ FakeDB directory)"
```

---

### Task 3: `/admin` endpoints + `admin_credit_grant` event

**Files:**
- Modify: `src/api/main.py`
- Test: `tests/integration/test_admin_console.py`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_admin_console.py`:

```python
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_posthog import FakePostHog

ADMIN = "test-admin-token"
UID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def client_and_fakes():
    os.environ["ADMIN_API_TOKEN"] = ADMIN
    db = FakeDB()
    db.add_user(UID, "alice@example.com")
    db.ensure_profile(UID, 300)
    ph = FakePostHog()
    from main import app, get_db, get_posthog
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_posthog] = lambda: ph
    yield TestClient(app), db, ph
    app.dependency_overrides.clear()
    os.environ.pop("ADMIN_API_TOKEN", None)


def _h(tok=ADMIN):
    return {"X-Admin-Token": tok}


def test_search_requires_token(client_and_fakes):
    tc, _, _ = client_and_fakes
    assert tc.get("/admin/accounts").status_code == 403
    assert tc.get("/admin/accounts", headers=_h("wrong")).status_code == 403


def test_search_returns_filtered(client_and_fakes):
    tc, _, _ = client_and_fakes
    r = tc.get("/admin/accounts", params={"q": "alice"}, headers=_h())
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1 and body[0]["email"] == "alice@example.com"
    assert body[0]["credits_balance"] == 300


def test_detail_and_unknown(client_and_fakes):
    tc, _, _ = client_and_fakes
    assert tc.get(f"/admin/accounts/{UID}", headers=_h()).json()["credits_balance"] == 300
    assert tc.get("/admin/accounts/00000000-0000-0000-0000-000000000000", headers=_h()).status_code == 404


def test_grant_happy_path(client_and_fakes):
    tc, db, ph = client_and_fakes
    r = tc.post(f"/admin/accounts/{UID}/grant", json={"amount": 1000}, headers=_h())
    assert r.status_code == 200
    assert r.json()["credits_balance"] == 1300
    assert db.get_balance(UID) == 1300
    granted = ph.find("admin_credit_grant")
    assert len(granted) == 1
    p = granted[0]["properties"]
    assert p["amount"] == 1000 and p["newBalance"] == 1300
    assert p["targetEmail"] == "alice@example.com" and p["source"] == "admin_console"
    # camelCase only
    for k in p:
        assert "_" not in k or k.startswith("$")
    assert any(t["delta"] == 1000 and t["reason"] == "admin_grant" for t in db.list_transactions(UID))


def test_grant_rejects_token_and_amount_and_unknown(client_and_fakes):
    tc, _, _ = client_and_fakes
    assert tc.post(f"/admin/accounts/{UID}/grant", json={"amount": 1000}).status_code == 403
    assert tc.post(f"/admin/accounts/{UID}/grant", json={"amount": 0}, headers=_h()).status_code == 422
    assert tc.post(f"/admin/accounts/{UID}/grant", json={"amount": -5}, headers=_h()).status_code == 422
    assert tc.post(f"/admin/accounts/{UID}/grant", json={"amount": 999999}, headers=_h()).status_code == 422
    assert tc.post("/admin/accounts/00000000-0000-0000-0000-000000000000/grant",
                   json={"amount": 10}, headers=_h()).status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/chrissilver/Documents/ChartSage && venv/bin/python -m pytest tests/integration/test_admin_console.py -q`
Expected: FAIL (404s — routes not defined yet).

- [ ] **Step 3: Implement the endpoints**

In `src/api/main.py`: add `require_admin` to the deps import — change `from deps import Identity, get_identity` to:

```python
from deps import Identity, get_identity, require_admin
```

Confirm `from pydantic import BaseModel` is already imported (it is — other request bodies use it). Add this block at the end of the endpoints section:

```python
class GrantIn(BaseModel):
    amount: int
    reason: str | None = None


@app.get("/admin/accounts")
async def admin_search_accounts(
    q: str = "",
    limit: int = 50,
    _admin: None = Depends(require_admin),
    db: SupabaseDB = Depends(get_db),
):
    return db.search_accounts(q, min(max(limit, 1), 200))


@app.get("/admin/accounts/{user_id}")
async def admin_account_detail(
    user_id: str,
    _admin: None = Depends(require_admin),
    db: SupabaseDB = Depends(get_db),
):
    detail = db.get_account_detail(user_id)
    if detail is None:
        raise HTTPException(status_code=404, detail={
            "code": "USER_NOT_FOUND", "message": "No such account."})
    return detail


@app.post("/admin/accounts/{user_id}/grant")
async def admin_grant_credits(
    user_id: str,
    body: GrantIn,
    _admin: None = Depends(require_admin),
    db: SupabaseDB = Depends(get_db),
    posthog: PostHogServer = Depends(get_posthog),
):
    if body.amount < 1 or body.amount > 100000:
        raise HTTPException(status_code=422, detail={
            "code": "INVALID_AMOUNT", "message": "Amount must be between 1 and 100000."})
    detail = db.get_account_detail(user_id)
    if detail is None:
        raise HTTPException(status_code=404, detail={
            "code": "USER_NOT_FOUND", "message": "No such account."})
    reason = (body.reason or "admin_grant").strip()[:60] or "admin_grant"
    db.ensure_profile(user_id, 0)  # create the profiles row if missing (no signup bonus)
    new_balance = db.grant_credits(user_id, body.amount, reason, ref="admin")
    posthog.capture(str(user_id), "admin_credit_grant", {
        "amount": body.amount,
        "newBalance": new_balance,
        "reason": reason,
        "targetEmail": detail.get("email"),
        "source": "admin_console",
    })
    return {"user_id": str(user_id), "credits_balance": new_balance}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/chrissilver/Documents/ChartSage && venv/bin/python -m pytest tests/integration/test_admin_console.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Run the full suite + commit**

Run: `cd /Users/chrissilver/Documents/ChartSage && venv/bin/python -m pytest tests/unit tests/integration -p no:warnings -q`
Expected: all pass.

```bash
git add src/api/main.py tests/integration/test_admin_console.py
git commit -m "feat(admin): /admin search + detail + grant endpoints (admin_credit_grant event)"
```

---

### Task 4: Cloud Run secret wiring

**Files:**
- Modify: `cloudbuild.yaml`

- [ ] **Step 1: Add the secret to `--set-secrets`**

In `cloudbuild.yaml`, find the `--set-secrets=` line:

```yaml
      - --set-secrets=ANTHROPIC_API_KEY=anthropic-key:latest,SUPABASE_SERVICE_ROLE_KEY=supabase-srk:latest,POSTHOG_API_KEY=posthog-key:latest
```

Append the admin token:

```yaml
      - --set-secrets=ANTHROPIC_API_KEY=anthropic-key:latest,SUPABASE_SERVICE_ROLE_KEY=supabase-srk:latest,POSTHOG_API_KEY=posthog-key:latest,ADMIN_API_TOKEN=admin-api-token:latest
```

- [ ] **Step 2: Commit**

```bash
git add cloudbuild.yaml
git commit -m "chore(admin): wire ADMIN_API_TOKEN secret into Cloud Run deploy"
```

> Note: the secret `admin-api-token` is created in Secret Manager in Phase 3 before the deploy. The Cloud Run runtime SA needs `roles/secretmanager.secretAccessor` on it (Phase 3, Task 7).

---

## Phase 2 — Frontend (implement → `tsc --noEmit` → commit; `next build` at end)

### Task 5: `adminApi.ts` (token store + `adminFetch`)

**Files:**
- Create: `src/app/lib/adminApi.ts`

- [ ] **Step 1: Implement**

```ts
'use client';

const TOKEN_KEY = 'cs_admin_token';

export function getAdminToken(): string {
  if (typeof window === 'undefined') return '';
  return sessionStorage.getItem(TOKEN_KEY) || '';
}

export function setAdminToken(token: string): void {
  sessionStorage.setItem(TOKEN_KEY, token.trim());
}

export function clearAdminToken(): void {
  sessionStorage.removeItem(TOKEN_KEY);
}

/** Fetch against the backend with the admin token header (NOT the Supabase Bearer). */
export async function adminFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers || {});
  headers.set('X-Admin-Token', getAdminToken());
  return fetch(`${process.env.NEXT_PUBLIC_API_URL}${path}`, { ...init, headers });
}
```

- [ ] **Step 2: Typecheck + commit**

Run: `cd /Users/chrissilver/Documents/ChartSage && npx tsc --noEmit`
Expected: exit 0.

```bash
git add src/app/lib/adminApi.ts
git commit -m "feat(admin): adminFetch + sessionStorage token store"
```

---

### Task 6: `/admin` route (gated UI)

**Files:**
- Create: `src/app/admin/layout.tsx`
- Create: `src/app/admin/page.tsx`

- [ ] **Step 1: Create the noindex layout**

`src/app/admin/layout.tsx`:

```tsx
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Admin · ChartSage',
  robots: { index: false, follow: false },
};

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
```

- [ ] **Step 2: Create the console page**

`src/app/admin/page.tsx` (dark theme, semantic classes consistent with the app; token gate → search → detail → grant):

```tsx
'use client';

import { useEffect, useState } from 'react';
import { adminFetch, getAdminToken, setAdminToken, clearAdminToken } from '../lib/adminApi';

interface Account { user_id: string; email: string; credits_balance: number; created_at?: string; }
interface Txn { delta: number; reason: string; ref?: string | null; created_at?: string; }
interface Detail extends Account { transactions: Txn[]; }

export default function AdminPage() {
  const [hasToken, setHasToken] = useState(false);
  const [tokenInput, setTokenInput] = useState('');
  const [q, setQ] = useState('');
  const [results, setResults] = useState<Account[]>([]);
  const [selected, setSelected] = useState<Detail | null>(null);
  const [amount, setAmount] = useState('');
  const [reason, setReason] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { setHasToken(!!getAdminToken()); }, []);

  const saveToken = () => { setAdminToken(tokenInput); setHasToken(!!tokenInput.trim()); setError(null); };
  const signOut = () => { clearAdminToken(); setHasToken(false); setResults([]); setSelected(null); };

  const search = async () => {
    setError(null); setBusy(true);
    try {
      const res = await adminFetch(`/admin/accounts?q=${encodeURIComponent(q)}&limit=50`);
      if (res.status === 403) { setError('Invalid or missing admin token.'); setHasToken(false); return; }
      if (!res.ok) { setError('Search failed.'); return; }
      setResults(await res.json());
    } finally { setBusy(false); }
  };

  const open = async (id: string) => {
    setError(null); setSelected(null);
    const res = await adminFetch(`/admin/accounts/${id}`);
    if (res.status === 403) { setError('Invalid or missing admin token.'); setHasToken(false); return; }
    if (!res.ok) { setError('Could not load account.'); return; }
    setSelected(await res.json()); setAmount(''); setReason('');
  };

  const grant = async () => {
    if (!selected) return;
    const n = parseInt(amount, 10);
    if (!Number.isFinite(n) || n < 1) { setError('Enter a positive amount.'); return; }
    setBusy(true); setError(null);
    try {
      const res = await adminFetch(`/admin/accounts/${selected.user_id}/grant`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ amount: n, reason: reason || undefined }),
      });
      if (res.status === 403) { setError('Invalid or missing admin token.'); setHasToken(false); return; }
      if (!res.ok) {
        const b = await res.json().catch(() => null);
        setError(b?.detail?.message || 'Grant failed.'); return;
      }
      const { credits_balance } = await res.json();
      setToast(`Granted ${n} credits → new balance ${credits_balance}`);
      setTimeout(() => setToast(null), 4000);
      await open(selected.user_id);
      await search();
    } finally { setBusy(false); }
  };

  if (!hasToken) {
    return (
      <main className="min-h-screen bg-bg text-ink flex items-center justify-center p-6">
        <div className="card rounded-2xl p-6 max-w-sm w-full">
          <h1 className="font-display text-xl font-semibold mb-2">Admin access</h1>
          <p className="text-sm text-ink-2 mb-4">Paste the admin token to continue.</p>
          <input type="password" value={tokenInput} onChange={(e) => setTokenInput(e.target.value)}
                 placeholder="Admin token"
                 className="w-full rounded-lg bg-surface border border-line px-3 py-2 text-sm mb-3" />
          <button onClick={saveToken} className="btn btn-primary w-full">Continue</button>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-bg text-ink p-6 max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="font-display text-2xl font-semibold">Admin · Accounts</h1>
        <button onClick={signOut} className="btn btn-ghost text-sm">Clear token</button>
      </div>

      <div className="flex gap-2 mb-4">
        <input value={q} onChange={(e) => setQ(e.target.value)}
               onKeyDown={(e) => e.key === 'Enter' && search()}
               placeholder="Search by email…"
               className="flex-1 rounded-lg bg-surface border border-line px-3 py-2 text-sm" />
        <button onClick={search} disabled={busy} className="btn btn-primary">Search</button>
      </div>

      {error && <p className="text-sm text-ember mb-3">{error}</p>}
      {toast && <p className="text-sm text-teal mb-3">{toast}</p>}

      <div className="grid gap-2 mb-6">
        {results.map((a) => (
          <button key={a.user_id} onClick={() => open(a.user_id)}
                  className="card rounded-xl px-4 py-3 text-left flex items-center justify-between hover:border-line-strong">
            <span className="text-sm">{a.email}</span>
            <span className="font-mono text-sm text-ink-2">{a.credits_balance} cr</span>
          </button>
        ))}
        {!results.length && <p className="text-sm text-ink-3">No accounts loaded — search above.</p>}
      </div>

      {selected && (
        <div className="card rounded-2xl p-5">
          <div className="flex items-center justify-between mb-1">
            <h2 className="font-display text-lg font-semibold">{selected.email}</h2>
            <span className="font-mono text-sm text-ink-2">{selected.credits_balance} cr</span>
          </div>
          <p className="font-mono text-xs text-ink-3 mb-4">{selected.user_id}</p>

          <div className="flex gap-2 items-end mb-5">
            <label className="flex-1">
              <span className="block text-xs text-ink-3 mb-1">Amount</span>
              <input type="number" value={amount} onChange={(e) => setAmount(e.target.value)}
                     placeholder="1000"
                     className="w-full rounded-lg bg-surface border border-line px-3 py-2 text-sm" />
            </label>
            <label className="flex-1">
              <span className="block text-xs text-ink-3 mb-1">Reason (optional)</span>
              <input value={reason} onChange={(e) => setReason(e.target.value)}
                     placeholder="admin_grant"
                     className="w-full rounded-lg bg-surface border border-line px-3 py-2 text-sm" />
            </label>
            <button onClick={grant} disabled={busy} className="btn btn-primary">Grant</button>
          </div>

          <h3 className="text-xs uppercase tracking-wide text-ink-3 mb-2">Recent transactions</h3>
          <div className="grid gap-1">
            {selected.transactions.map((t, i) => (
              <div key={i} className="flex items-center justify-between text-sm border-b border-line py-1">
                <span className="text-ink-2">{t.reason}{t.ref ? ` · ${t.ref}` : ''}</span>
                <span className={`font-mono ${t.delta >= 0 ? 'text-teal' : 'text-ember'}`}>
                  {t.delta >= 0 ? '+' : ''}{t.delta}
                </span>
              </div>
            ))}
            {!selected.transactions.length && <p className="text-sm text-ink-3">No transactions.</p>}
          </div>
        </div>
      )}
    </main>
  );
}
```

> **Note on class names:** match whatever the design system actually exposes. The tokens above (`bg-bg`, `text-ink`/`ink-2`/`ink-3`, `card`, `surface`, `line`/`line-strong`, `btn`/`btn-primary`/`btn-ghost`, `text-teal`/`text-ember`, `font-display`/`font-mono`) follow the established redesign tokens — verify each against `tailwind.config.js` / `globals.css` and swap any that differ before committing.

- [ ] **Step 3: Typecheck + commit**

Run: `cd /Users/chrissilver/Documents/ChartSage && npx tsc --noEmit`
Expected: exit 0.

```bash
git add src/app/admin/layout.tsx src/app/admin/page.tsx
git commit -m "feat(admin): gated /admin console (token gate, search, detail, grant)"
```

---

## Phase 3 — Build, deploy, bootstrap (production — requires explicit user authorization)

### Task 7: Final verification, deploy, and the 1000-credit grant

**Files:** none (ops).

- [ ] **Step 1: Full backend suite + frontend build**

```bash
cd /Users/chrissilver/Documents/ChartSage
venv/bin/python -m pytest tests/unit tests/integration -p no:warnings -q   # all pass
npx tsc --noEmit                                                            # exit 0
npm run build                                                              # compiles, /admin route present
```

- [ ] **Step 2: Final code review (subagent)**

Dispatch an independent reviewer over the diff `main..admin-console` focused on the money/security path: `require_admin` is fail-closed and constant-time; the grant validates amount + existence before debiting; the token is never logged; `admin_credit_grant` fires with the right props; no secret committed.

- [ ] **Step 3: Create the Secret Manager secret (user-authorized)**

```bash
TOKEN=$(openssl rand -hex 32)
printf '%s' "$TOKEN" | gcloud secrets create admin-api-token --data-file=- --project=chartsage-497909
gcloud secrets add-iam-policy-binding admin-api-token \
  --member="serviceAccount:chartsage-runner@chartsage-497909.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" --project=chartsage-497909
```
Keep `$TOKEN` in the shell for Step 5; do not print it.

- [ ] **Step 4: Deploy backend to Cloud Run**

Pass the real `_SUPABASE_URL` from `.env` (it is the shared cloud project), but let `_FRONTEND_BASE_URL` use the now-correct cloudbuild default — do NOT source `.env`'s `FRONTEND_BASE_URL` (it is localhost):

```bash
SUPA=$(grep -E '^SUPABASE_URL=' .env | cut -d= -f2-)
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_SUPABASE_URL="$SUPA",_TAG="$(git rev-parse --short HEAD)"
```
Confirm the new revision serves 100% traffic.

- [ ] **Step 5: Grant `cj.silver@me.com` 1000 credits**

```bash
BASE="https://chartsage-backend-112026133429.us-central1.run.app"
UID=$(curl -s "$BASE/admin/accounts?q=cj.silver@me.com" -H "X-Admin-Token: $TOKEN" \
      | python3 -c "import sys,json; r=json.load(sys.stdin); print(r[0]['user_id'] if r else '')")
echo "resolved user_id=$UID"
curl -s -X POST "$BASE/admin/accounts/$UID/grant" -H "X-Admin-Token: $TOKEN" \
  -H "Content-Type: application/json" -d '{"amount":1000,"reason":"admin_grant"}'
# expect {"user_id": "...", "credits_balance": <old+1000>}
```

- [ ] **Step 6: Deploy frontend**

Merge `admin-console` → `main` and push (Vercel auto-deploys the `/admin` route):

```bash
git checkout main && git merge --ff-only admin-console && git push origin main
```

- [ ] **Step 7: Verify + hand off the token**

- Confirm the grant response balance, the `admin_credit_grant` event in PostHog, and the `admin_grant` ledger row.
- Confirm a wrong/absent `X-Admin-Token` returns 403 in prod.
- Tell the user to retrieve the token for the console UI:
  `gcloud secrets versions access latest --secret=admin-api-token --project=chartsage-497909`
  then open `https://chartsage-xi.vercel.app/admin`, paste it, search, and grant.

---

## Self-Review

**Spec coverage:** require_admin shared-token guard (T1) ✓ · search/detail/grant endpoints (T3) ✓ · DB helpers via service-role + GoTrue admin (T2) ✓ · admin_credit_grant event (T3) ✓ · cloudbuild secret (T4) ✓ · adminFetch + token store (T5) ✓ · gated /admin UI with noindex (T6) ✓ · secret + deploy + 1000 grant + verify (T7) ✓.

**Placeholder scan:** every code/test step has concrete code + exact commands. The one judgment call (frontend class-token names) is called out explicitly with a verification instruction, not left vague.

**Type consistency:** `search_accounts(query, limit)`, `get_account_detail(user_id)`, `add_user(user_id, email, created_at)`, `grant_credits(user_id, amount, reason, ref)`, `GrantIn{amount, reason}`, and the `admin_credit_grant` props (`amount`/`newBalance`/`reason`/`targetEmail`/`source`) are identical across the DB layer, FakeDB, endpoints, tests, and frontend.
