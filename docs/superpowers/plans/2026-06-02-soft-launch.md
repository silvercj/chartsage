# Soft-Launch Readiness v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound the free-tier cost exposure, add the legal basics, and get error visibility — the guardrails to open a free public beta.

**Architecture:** Backend layered anonymous caps (per-IP daily + global daily) backed by a new `anon_report_log` table that also records IP + a coarse fingerprint for tracking; drafted `/terms` + `/privacy` pages with footer + login-consent links; Sentry error capture (backend + frontend) plus infra spend alarms. Caps and Sentry are env-gated and fail-open/no-op when unconfigured.

**Tech Stack:** FastAPI + supabase-py (service-role) on Cloud Run · Next.js 14 App Router on Vercel · PostHog · Sentry · Google Secret Manager.

**Branch:** `soft-launch` (created off `main`, carries the spec commit). **Forbid subagents from running `git checkout`/`switch`/`reset`/`stash`.** Python interpreter: `venv/bin/python`.

---

## File Structure

- `docs/migrations/soft-launch.sql` (create) — `anon_report_log` table + indexes + RLS.
- `src/api/db.py` (modify) — `log_anon_report`, `count_anon_reports_today`, `count_anon_reports_today_by_ip`.
- `src/api/alerting.py` (create) — `report_alert()` Sentry-safe no-op helper.
- `src/api/main.py` (modify) — caps + IP/fingerprint capture in `generate_report`; constants; Sentry init (Phase 3).
- `cloudbuild.yaml` (modify) — cap env vars (P1) + `SENTRY_DSN` secret (P3).
- `requirements.txt` (modify) — `sentry-sdk[fastapi]` (P3).
- `tests/helpers/fake_db.py` (modify) — `_anon_log` + the 3 methods.
- `tests/integration/test_anon_caps.py` (create) — the caps.
- `src/app/terms/page.tsx`, `src/app/privacy/page.tsx` (create) — legal pages.
- `src/app/components/marketing/MarketingFooter.tsx` (modify) — legal links.
- `src/app/login/page.tsx` (modify) — consent line.
- `src/app/instrumentation.ts`, `src/app/instrumentation-client.ts` (create) + `package.json` (modify) — frontend Sentry (P3).

---

## Phase 1 — Abuse / cost guard (backend, TDD)

### Task 1: Migration + DB methods + FakeDB

**Files:** Create `docs/migrations/soft-launch.sql`; Modify `src/api/db.py`, `tests/helpers/fake_db.py`; Test `tests/integration/test_anon_caps.py` (DB portion).

- [ ] **Step 1: Create the migration** `docs/migrations/soft-launch.sql`:

```sql
-- Soft-launch: anonymous report log (abuse tracking + daily caps). Run once in Supabase SQL editor.
create table if not exists anon_report_log (
  id          uuid primary key default gen_random_uuid(),
  anon_id     uuid,
  ip          text,
  fingerprint text,
  created_at  timestamptz not null default now()
);
create index if not exists anon_report_log_created_idx on anon_report_log (created_at desc);
create index if not exists anon_report_log_ip_idx on anon_report_log (ip, created_at desc);
alter table anon_report_log enable row level security;  -- service-role only; no client policies
```

- [ ] **Step 2: Write the failing test** — create `tests/integration/test_anon_caps.py` with the DB tests first:

```python
from datetime import datetime, timezone, timedelta
from tests.helpers.fake_db import FakeDB


def test_log_and_count_today():
    db = FakeDB()
    db.log_anon_report("11111111-1111-1111-1111-111111111111", "1.2.3.4", "abc123")
    db.log_anon_report("22222222-2222-2222-2222-222222222222", "1.2.3.4", "def456")
    db.log_anon_report("33333333-3333-3333-3333-333333333333", "9.9.9.9", "ghi789")
    assert db.count_anon_reports_today() == 3
    assert db.count_anon_reports_today_by_ip("1.2.3.4") == 2
    assert db.count_anon_reports_today_by_ip("9.9.9.9") == 1


def test_count_today_excludes_old_rows():
    db = FakeDB()
    db.log_anon_report("11111111-1111-1111-1111-111111111111", "1.2.3.4", "abc")
    # inject a row from two days ago directly
    db._anon_log.append({"anon_id": None, "ip": "1.2.3.4", "fingerprint": "old",
                         "created_at": datetime.now(timezone.utc) - timedelta(days=2)})
    assert db.count_anon_reports_today() == 1
    assert db.count_anon_reports_today_by_ip("1.2.3.4") == 1
```

- [ ] **Step 3: Run → FAIL**

Run: `cd /Users/chrissilver/Documents/ChartSage && venv/bin/python -m pytest tests/integration/test_anon_caps.py -q`
Expected: FAIL (`FakeDB` has no `log_anon_report`).

- [ ] **Step 4: Implement FakeDB methods.** In `tests/helpers/fake_db.py`: add `from datetime import datetime, timezone` at the top; add `self._anon_log: list[dict] = []` in `__init__`; add after `record_upgrade_intent`:

```python
    # --- anon abuse log (soft-launch) ---
    def log_anon_report(self, anon_id, ip, fingerprint) -> None:
        self._anon_log.append({
            "anon_id": str(anon_id) if anon_id else None,
            "ip": ip, "fingerprint": fingerprint,
            "created_at": datetime.now(timezone.utc),
        })

    def _utc_today_start(self):
        n = datetime.now(timezone.utc)
        return n.replace(hour=0, minute=0, second=0, microsecond=0)

    def count_anon_reports_today(self) -> int:
        s = self._utc_today_start()
        return sum(1 for r in self._anon_log if r["created_at"] >= s)

    def count_anon_reports_today_by_ip(self, ip) -> int:
        s = self._utc_today_start()
        return sum(1 for r in self._anon_log if r["ip"] == ip and r["created_at"] >= s)
```

- [ ] **Step 5: Run → PASS** (2 passed).

- [ ] **Step 6: Implement real SupabaseDB methods.** In `src/api/db.py`: ensure `from datetime import datetime, timezone` is imported; add to `SupabaseDB` (after `list_transactions`):

```python
    # --- anon abuse log (soft-launch) ---
    def _utc_today_start_iso(self) -> str:
        n = datetime.now(timezone.utc)
        return n.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    def log_anon_report(self, anon_id, ip, fingerprint) -> None:
        self.client.table("anon_report_log").insert({
            "anon_id": str(anon_id) if anon_id else None,
            "ip": ip, "fingerprint": fingerprint,
        }).execute()

    def count_anon_reports_today(self) -> int:
        res = (self.client.table("anon_report_log").select("id", count="exact")
               .gte("created_at", self._utc_today_start_iso()).execute())
        return res.count or 0

    def count_anon_reports_today_by_ip(self, ip) -> int:
        res = (self.client.table("anon_report_log").select("id", count="exact")
               .eq("ip", ip).gte("created_at", self._utc_today_start_iso()).execute())
        return res.count or 0
```

- [ ] **Step 7: Run full suite + commit**

Run: `cd /Users/chrissilver/Documents/ChartSage && venv/bin/python -m pytest tests/unit tests/integration -p no:warnings -q` → all pass.

```bash
git add docs/migrations/soft-launch.sql src/api/db.py tests/helpers/fake_db.py tests/integration/test_anon_caps.py
git commit -m "feat(launch): anon_report_log table + daily-count DB methods"
```

---

### Task 2: `report_alert` helper + caps in `generate_report`

**Files:** Create `src/api/alerting.py`; Modify `src/api/main.py`; Test `tests/integration/test_anon_caps.py` (endpoint portion).

- [ ] **Step 1: Create `src/api/alerting.py`**

```python
"""Lightweight alerting. Routes to Sentry when configured; always a safe no-op otherwise.

Sentry is initialized in main.py only when SENTRY_DSN is set (added in the monitoring
phase). Until then — and whenever Sentry is unconfigured — this logs and returns.
"""
import logging


def report_alert(message: str, **context) -> None:
    try:
        import sentry_sdk
        with sentry_sdk.push_scope() as scope:
            for k, v in context.items():
                scope.set_extra(k, v)
            sentry_sdk.capture_message(message, level="warning")
    except Exception:
        logging.warning("[ALERT] %s %s", message, context or "")
```

- [ ] **Step 2: Write the failing endpoint tests** — append to `tests/integration/test_anon_caps.py`:

```python
import io
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from uuid import uuid4

import main as main_module
from tests.helpers.fake_claude import FakeClaude, tool_use
from tests.helpers.fake_storage import FakeStorage
from tests.helpers.fake_posthog import FakePostHog


def _csv(df):
    b = io.StringIO(); df.to_csv(b, index=False); return b.getvalue().encode("utf-8")


@pytest.fixture
def caps_client(sales):
    calls = [tool_use("frequency_bar_chart", {"column": "region", "title": "T", "intent": "i"}, id_="t0")]
    fake_claude = FakeClaude([
        {"tool_calls": calls},
        {"tool_calls": [tool_use("submit_narrative", {"summary": "S.", "captions": ["c"], "data_quality": []})]},
    ])
    db = FakeDB(); storage = FakeStorage(); ph = FakePostHog()
    from main import app, get_claude_client, get_db, get_storage, get_posthog
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=fake_claude)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_storage] = lambda: storage
    app.dependency_overrides[get_posthog] = lambda: ph
    yield TestClient(app), db, ph, fake_claude
    app.dependency_overrides.clear()


def _post(tc, anon, ip="5.5.5.5"):
    return tc.post("/generate-report",
                   files={"file": ("s.csv", _csv(pd.DataFrame({"region": ["N", "S"], "rev": [1, 2]})), "text/csv")},
                   headers={"X-Anon-Id": anon, "X-Forwarded-For": ip})


def test_per_ip_cap_blocks_with_429(caps_client, monkeypatch, sales):
    tc, db, ph, claude = caps_client
    monkeypatch.setattr(main_module, "ANON_IP_DAILY_CAP", 1)
    db.log_anon_report(str(uuid4()), "5.5.5.5", "x")  # already 1 from this IP today
    r = _post(tc, str(uuid4()), ip="5.5.5.5")
    assert r.status_code == 429 and r.json()["detail"]["code"] == "RATE_LIMITED"
    assert len(ph.find("anon_cap_hit")) == 1 and ph.find("anon_cap_hit")[0]["properties"]["scope"] == "ip"


def test_global_cap_blocks_with_503(caps_client, monkeypatch, sales):
    tc, db, ph, claude = caps_client
    monkeypatch.setattr(main_module, "ANON_IP_DAILY_CAP", 100)
    monkeypatch.setattr(main_module, "ANON_GLOBAL_DAILY_CAP", 1)
    db.log_anon_report(str(uuid4()), "9.9.9.9", "x")  # 1 globally today
    r = _post(tc, str(uuid4()), ip="5.5.5.5")
    assert r.status_code == 503 and r.json()["detail"]["code"] == "FREE_TIER_AT_CAPACITY"
    assert ph.find("anon_cap_hit")[0]["properties"]["scope"] == "global"


def test_anon_success_logs_row(caps_client, sales):
    tc, db, ph, claude = caps_client
    r = _post(tc, str(uuid4()), ip="5.5.5.5")
    assert r.status_code == 200
    assert len(db._anon_log) == 1 and db._anon_log[0]["ip"] == "5.5.5.5" and db._anon_log[0]["fingerprint"]


def test_authed_unaffected_by_new_caps(caps_client, monkeypatch, sales):
    tc, db, ph, claude = caps_client
    monkeypatch.setattr(main_module, "ANON_GLOBAL_DAILY_CAP", 0)  # would block any anon
    monkeypatch.setattr(main_module, "ANON_IP_DAILY_CAP", 0)
    from deps import get_identity, Identity
    uid = uuid4()
    main_module.app.dependency_overrides[get_identity] = lambda: Identity(user_id=uid)
    db.ensure_profile(uid, 300)
    r = tc.post("/generate-report",
                files={"file": ("s.csv", _csv(pd.DataFrame({"region": ["N", "S"], "rev": [1, 2]})), "text/csv")})
    assert r.status_code == 200
    main_module.app.dependency_overrides.pop(get_identity, None)
```

- [ ] **Step 3: Run → FAIL**

Run: `cd /Users/chrissilver/Documents/ChartSage && venv/bin/python -m pytest tests/integration/test_anon_caps.py -q`
Expected: caps tests FAIL (no caps yet; per-IP/global return 200).

- [ ] **Step 4: Implement.** In `src/api/main.py`:

(a) Ensure imports: `Request` from fastapi (add to the existing `from fastapi import ...`), `import hashlib`, and `from alerting import report_alert`.

(b) Add constants next to `ANON_REPORT_LIMIT` (~line 40):
```python
ANON_IP_DAILY_CAP = int(os.environ.get("ANON_IP_DAILY_CAP", "5"))
ANON_GLOBAL_DAILY_CAP = int(os.environ.get("ANON_GLOBAL_DAILY_CAP", "200"))
```

(c) Add module-level helpers (near `_title_from_summary`):
```python
def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else ""


def _client_fingerprint(request: Request) -> str:
    ua = request.headers.get("user-agent", "")
    lang = request.headers.get("accept-language", "")
    return hashlib.sha256(f"{ua}|{lang}".encode()).hexdigest()[:16]
```

(d) Add `request: Request` as the first parameter of `generate_report`. Right after the function opens (before the gating block), compute:
```python
    client_ip = _client_ip(request)
    fingerprint = _client_fingerprint(request)
```

(e) In the anonymous branch, immediately AFTER the existing lifetime per-anon `ANON_LIMIT_REACHED` check (the `if not bypass and existing_count >= ANON_REPORT_LIMIT:` block), add:
```python
        if not bypass:
            if db.count_anon_reports_today_by_ip(client_ip) >= ANON_IP_DAILY_CAP:
                posthog.capture(identity.distinct_id, "anon_cap_hit", {"scope": "ip"})
                report_alert("anon per-IP daily cap hit", ip=client_ip)
                raise HTTPException(status_code=429, detail={
                    "code": "RATE_LIMITED",
                    "message": "Too many free reports from your network today. Sign in to keep going."})
            if db.count_anon_reports_today() >= ANON_GLOBAL_DAILY_CAP:
                posthog.capture(identity.distinct_id, "anon_cap_hit", {"scope": "global"})
                report_alert("anon global daily cap hit")
                raise HTTPException(status_code=503, detail={
                    "code": "FREE_TIER_AT_CAPACITY",
                    "message": "The free tier is at capacity for today — sign in to keep going."})
```

(f) After the `db.save_report(...)` block, change the existing `if identity.is_authenticated:` credit-spend block to add an `else` that logs the anon report:
```python
    if identity.is_authenticated:
        try:
            new_balance = db.spend_credits(identity.user_id, report_cost, spend_reason, report_id)
            posthog.capture(identity.distinct_id, "credits_spent",
                            {"amount": report_cost, "balance": new_balance, "reason": spend_reason})
        except InsufficientCredits:
            logging.warning("Credit spend lost a race on report %s; serving free.", report_id)
    else:
        try:
            db.log_anon_report(identity.anon_id, client_ip, fingerprint)
        except Exception:
            logging.warning("anon_report_log insert failed for %s", report_id, exc_info=True)
```

- [ ] **Step 5: Run → PASS** (`tests/integration/test_anon_caps.py` all pass).

- [ ] **Step 6: Full suite + commit**

Run: `cd /Users/chrissilver/Documents/ChartSage && venv/bin/python -m pytest tests/unit tests/integration -p no:warnings -q` → all pass.

```bash
git add src/api/alerting.py src/api/main.py tests/integration/test_anon_caps.py
git commit -m "feat(launch): per-IP + global daily anon caps with IP/fingerprint logging"
```

---

### Task 3: Cloud Run cap env vars

**Files:** Modify `cloudbuild.yaml`.

- [ ] **Step 1:** In `cloudbuild.yaml`, append to the `--set-env-vars=` line (after `UNLIMITED_ANON_IDS=$_UNLIMITED_ANON_IDS`): `,ANON_IP_DAILY_CAP=$_ANON_IP_DAILY_CAP,ANON_GLOBAL_DAILY_CAP=$_ANON_GLOBAL_DAILY_CAP`. Add to `substitutions:`:
```yaml
  _ANON_IP_DAILY_CAP: '5'
  _ANON_GLOBAL_DAILY_CAP: '200'
```

- [ ] **Step 2: Commit**
```bash
git add cloudbuild.yaml
git commit -m "chore(launch): wire anon cap env vars into Cloud Run"
```

---

## Phase 2 — Legal pages (frontend; implement → `tsc --noEmit` → commit)

### Task 4: `/terms` + `/privacy` pages

**Files:** Create `src/app/terms/page.tsx`, `src/app/privacy/page.tsx`.

- [ ] **Step 1: Verify design tokens.** Run `grep -nE "bg-canvas|bg-surface|text-ink|\.card|border-line|font-display|max-w" src/app/globals.css tailwind.config.js | head -40` and skim `src/app/credits/page.tsx` for the page-shell pattern (background, max-width container, prose text colors). Use the real tokens.

- [ ] **Step 2: Create `src/app/terms/page.tsx`** — a server component, public (normal metadata, not noindex), dark theme. Content is a starting-point ToS (cover: acceptance, description of service, accounts & credits, acceptable-use incl. no abuse/automated farming, **AI-generated output is provided "as is" — verify before relying on it**, intellectual property, disclaimers & limitation of liability, changes, contact). Begin the file with `{/* NOT LEGAL ADVICE — starting-point template; have a professional review before the paid launch. */}`. Include a "Last updated: June 2, 2026" line. Structure:

```tsx
import type { Metadata } from 'next';

export const metadata: Metadata = { title: 'Terms of Service · ChartSage' };

export default function TermsPage() {
  // NOT LEGAL ADVICE — starting-point template; have a professional review before the paid launch.
  return (
    <main className="min-h-screen bg-canvas text-ink">
      <div className="max-w-2xl mx-auto px-6 py-16">
        <h1 className="font-display text-3xl font-semibold mb-2">Terms of Service</h1>
        <p className="font-mono text-xs text-ink-3 mb-10">Last updated: June 2, 2026</p>
        <div className="space-y-6 text-ink-2 leading-relaxed text-sm">
          <section><h2 className="font-display text-lg text-ink mb-2">1. Acceptance</h2>
            <p>By accessing or using ChartSage ("the Service") you agree to these Terms. If you do not agree, do not use the Service.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">2. The Service</h2>
            <p>ChartSage generates data visualizations and written analysis from files you upload, using automated AI processing. Output is generated automatically and may contain errors — review it before relying on it for any decision.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">3. Accounts & credits</h2>
            <p>Some features require an account and consume credits. Credits have no cash value, are non-transferable, and (during the beta) are provided at our discretion.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">4. Acceptable use</h2>
            <p>Do not abuse, overload, or attempt to circumvent usage limits of the Service (including automated or large-scale generation of free reports), upload unlawful content, or infringe others' rights.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">5. Your content</h2>
            <p>You retain ownership of files you upload and the reports generated for you. You grant us the limited rights needed to operate the Service (storing and processing your data to produce your reports). See our <a className="text-accent underline" href="/privacy">Privacy Policy</a>.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">6. Disclaimers & liability</h2>
            <p>The Service and its AI-generated output are provided "as is", without warranties of any kind. To the maximum extent permitted by law, we are not liable for any indirect or consequential damages arising from your use of the Service or reliance on its output.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">7. Changes</h2>
            <p>We may update these Terms; material changes will be reflected by the "last updated" date above.</p></section>
          <section><h2 className="font-display text-lg text-ink mb-2">8. Contact</h2>
            <p>Questions? Contact us at support@chartsage (update with your real support address).</p></section>
        </div>
      </div>
    </main>
  );
}
```

- [ ] **Step 3: Create `src/app/privacy/page.tsx`** — same shell; Privacy Policy covering: what we collect (account email; **files you upload (CSV/Excel)**; generated reports; **IP address and a coarse device fingerprint, used to prevent abuse of the free tier**; product analytics via PostHog); how we use it (to provide the Service, including **AI processing by a third-party provider**); **Data handling & retention** (stored in our cloud infrastructure; retained while your account is active / as needed to operate the Service); **deletion** (email us to request deletion of your account and data); cookies/analytics; changes; contact. Same `{/* NOT LEGAL ADVICE ... */}` comment + "Last updated" line + same token classes. Keep the third-party AI provider generic (do not name the vendor).

- [ ] **Step 4: tsc + commit**

Run: `cd /Users/chrissilver/Documents/ChartSage && npx tsc --noEmit` → exit 0.
```bash
git add src/app/terms/page.tsx src/app/privacy/page.tsx
git commit -m "feat(launch): /terms + /privacy starting-point pages"
```

---

### Task 5: Footer links + login consent line

**Files:** Modify `src/app/components/marketing/MarketingFooter.tsx`, `src/app/login/page.tsx`.

- [ ] **Step 1: Read both files** to match their structure/classes.

- [ ] **Step 2: Footer links.** In `MarketingFooter.tsx`, add `Terms` (`/terms`) and `Privacy` (`/privacy`) links alongside the existing footer links, using the existing link styling. If there's no link row, add a small one: `<a href="/terms" className="...">Terms</a>` · `<a href="/privacy" className="...">Privacy</a>`.

- [ ] **Step 3: Consent line.** In `src/app/login/page.tsx`, below the auth buttons, add:
```tsx
<p className="text-xs text-ink-3 mt-6 text-center">
  By continuing, you agree to our{' '}
  <a href="/terms" className="text-accent underline">Terms</a> and{' '}
  <a href="/privacy" className="text-accent underline">Privacy Policy</a>.
</p>
```
(Adjust the token classes to the real ones used on that page.)

- [ ] **Step 4: tsc + commit**

Run: `cd /Users/chrissilver/Documents/ChartSage && npx tsc --noEmit` → exit 0.
```bash
git add src/app/components/marketing/MarketingFooter.tsx src/app/login/page.tsx
git commit -m "feat(launch): footer legal links + login consent line"
```

---

## Phase 3 — Monitoring (Sentry + alarms)

### Task 6: Backend Sentry

**Files:** Modify `requirements.txt`, `src/api/main.py`.

- [ ] **Step 1: Add the dep.** In `requirements.txt`, add a line: `sentry-sdk[fastapi]`.

- [ ] **Step 2: Init (no-op when unset).** In `src/api/main.py`, after the imports and before `app = FastAPI(...)`, add:
```python
_SENTRY_DSN = os.environ.get("SENTRY_DSN", "")
if _SENTRY_DSN:
    import sentry_sdk
    sentry_sdk.init(dsn=_SENTRY_DSN, traces_sample_rate=0.0, send_default_pii=False)
```
The FastAPI integration auto-captures unhandled exceptions; `report_alert()` (from Task 2) now routes to Sentry whenever this init has run.

- [ ] **Step 3: Verify no-op locally + commit.** Confirm the app imports with `SENTRY_DSN` unset:

Run: `cd /Users/chrissilver/Documents/ChartSage && venv/bin/python -c "import main; print('ok')"` → prints `ok`.
Run the full suite: `venv/bin/python -m pytest tests/unit tests/integration -p no:warnings -q` → all pass.
```bash
git add requirements.txt src/api/main.py
git commit -m "feat(launch): backend Sentry init (no-op when unconfigured)"
```

> Note: `cloudbuild.yaml` gets `SENTRY_DSN=sentry-dsn:latest` added to `--set-secrets` in Task 8 (alongside creating the secret), so the build never references a non-existent secret before it exists.

---

### Task 7: Frontend Sentry

**Files:** Create `src/app/instrumentation.ts`, `src/app/instrumentation-client.ts`; Modify `package.json`.

- [ ] **Step 1: Install.** Run: `cd /Users/chrissilver/Documents/ChartSage && npm install @sentry/nextjs`.

- [ ] **Step 2: Server instrumentation.** Create `src/app/instrumentation.ts`:
```ts
import * as Sentry from '@sentry/nextjs';

export function register() {
  const dsn = process.env.NEXT_PUBLIC_SENTRY_DSN;
  if (dsn) Sentry.init({ dsn, tracesSampleRate: 0 });
}
```

- [ ] **Step 3: Client instrumentation.** Create `src/app/instrumentation-client.ts`:
```ts
import * as Sentry from '@sentry/nextjs';

const dsn = process.env.NEXT_PUBLIC_SENTRY_DSN;
if (dsn) Sentry.init({ dsn, tracesSampleRate: 0 });
```

(Do NOT add `withSentryConfig` to `next.config` — keep the build simple; source-map upload is a later nicety. Init is a no-op when `NEXT_PUBLIC_SENTRY_DSN` is unset, so dev/build are unaffected.)

- [ ] **Step 4: tsc + build + commit**

Run: `cd /Users/chrissilver/Documents/ChartSage && npx tsc --noEmit` → exit 0.
Run: `npm run build` → compiles clean.
```bash
git add package.json package-lock.json src/app/instrumentation.ts src/app/instrumentation-client.ts
git commit -m "feat(launch): frontend Sentry init (no-op when unconfigured)"
```

---

### Task 8: Deploy + bootstrap (production — requires explicit user authorization)

**Files:** Modify `cloudbuild.yaml` (secret line).

- [ ] **Step 1: Add the Sentry secret to cloudbuild.** In `cloudbuild.yaml`, append `,SENTRY_DSN=sentry-dsn:latest` to the `--set-secrets=` line. Commit:
```bash
git add cloudbuild.yaml
git commit -m "chore(launch): wire SENTRY_DSN secret into Cloud Run"
```

- [ ] **Step 2: Full verification**
```bash
cd /Users/chrissilver/Documents/ChartSage
venv/bin/python -m pytest tests/unit tests/integration -p no:warnings -q   # all pass
npx tsc --noEmit && npm run build                                         # clean; /terms + /privacy present
```

- [ ] **Step 3: Final review (subagent).** Dispatch an independent reviewer over `main..soft-launch` focused on: caps checked before the Claude call (no spend on block), authed path unaffected, IP/fingerprint parsing safe (no crash on missing headers), Sentry init truly no-op when unset, legal pages public.

- [ ] **Step 4: Run the migration.** In the Supabase SQL editor, run `docs/migrations/soft-launch.sql`. (User action.)

- [ ] **Step 5: Create the Sentry secret + access (user-authorized).** With the backend DSN from the user's Sentry project:
```bash
printf '%s' "<BACKEND_SENTRY_DSN>" | gcloud secrets create sentry-dsn --replication-policy=automatic --data-file=- --project=chartsage-497909
gcloud secrets add-iam-policy-binding sentry-dsn \
  --member="serviceAccount:chartsage-runner@chartsage-497909.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" --project=chartsage-497909
```

- [ ] **Step 6: Set the frontend DSN in Vercel.** Add `NEXT_PUBLIC_SENTRY_DSN=<FRONTEND_SENTRY_DSN>` to the Vercel project env (all environments). (User action or via Vercel CLI.)

- [ ] **Step 7: Deploy backend.** (Real `_SUPABASE_URL` from `.env`; do NOT source `FRONTEND_BASE_URL` — the cloudbuild default is correct.)
```bash
SUPA=$(grep -E '^SUPABASE_URL=' .env | cut -d= -f2-)
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_SUPABASE_URL="$SUPA",_TAG="$(git rev-parse --short HEAD)"
```

- [ ] **Step 8: Deploy frontend.** Merge `soft-launch` → `main` and push (Vercel deploys `/terms`, `/privacy`, Sentry):
```bash
git checkout main && git merge --ff-only soft-launch && git push origin main
```

- [ ] **Step 9: Infra alarms.** Create a **GCP billing budget + alert** on project `chartsage-497909` (a monthly cap with email alerts at 50/90/100% — via Console → Billing → Budgets & alerts, or `gcloud billing budgets create` if the billing account is accessible). Set an **Anthropic workspace spend limit** in the Anthropic console. (User actions; assist where possible.)

- [ ] **Step 10: Smoke.**
```bash
BASE="https://chartsage-backend-112026133429.us-central1.run.app"
# anon report logs IP/fingerprint (then check the anon_report_log table in Supabase)
ANON=$(python3 -c 'import uuid;print(uuid.uuid4())')
curl -s -o /dev/null -w "anon report -> %{http_code}\n" -X POST "$BASE/generate-report" \
  -H "X-Anon-Id: $ANON" -H "X-Forwarded-For: 203.0.113.7" -F "file=@/tmp/smoke.csv;type=text/csv"
# /terms + /privacy are public
curl -s -o /dev/null -w "/terms -> %{http_code}\n" https://chartsage-xi.vercel.app/terms
curl -s -o /dev/null -w "/privacy -> %{http_code}\n" https://chartsage-xi.vercel.app/privacy
```
Confirm: the `anon_report_log` row exists with the IP + fingerprint; lowering the caps (temporarily, via env) trips 429/503; a forced backend error appears in Sentry; the legal pages are public.

---

## Self-Review

**Spec coverage:** anon_report_log + IP/fingerprint capture (T1, T2) ✓ · per-IP daily cap 429 (T2) ✓ · global daily cap 503 hard-stop (T2) ✓ · anon_cap_hit event + Sentry alert (T2, T6) ✓ · authed unaffected (T2 test) ✓ · cap env vars (T3) ✓ · /terms + /privacy with data/retention + deletion-by-email + generic AI vendor (T4) ✓ · footer links + login consent (T5) ✓ · backend Sentry no-op-guarded (T6) ✓ · frontend Sentry no-op-guarded (T7) ✓ · migration + secret + Vercel env + deploy + GCP budget + Anthropic cap + smoke (T8) ✓.

**Placeholder scan:** every code step has concrete code; the two judgment calls (frontend token-class names; the support email address) are flagged inline with explicit "verify/replace" instructions, not left vague.

**Type consistency:** `log_anon_report(anon_id, ip, fingerprint)`, `count_anon_reports_today()`, `count_anon_reports_today_by_ip(ip)`, `report_alert(message, **context)`, constants `ANON_IP_DAILY_CAP`/`ANON_GLOBAL_DAILY_CAP`, and the `anon_cap_hit {scope}` event are identical across db.py, FakeDB, main.py, and the tests.
