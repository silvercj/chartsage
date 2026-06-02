# Beta Enablers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a support contact form and a report feedback widget so the free-beta traffic can reach us and rate reports.

**Architecture:** A self-contained `/contact` form → `POST /contact` → a new `support_messages` Supabase table + a PostHog `support_request` event (no email/domain needed). A frontend-only report feedback widget → a PostHog `report_feedback` event (no DB).

**Tech Stack:** FastAPI + supabase-py (Cloud Run) · Next.js 14 App Router (Vercel) · PostHog.

**Branch:** `beta-enablers` (off main, carries the spec). **Forbid subagents from `git checkout`/`switch`/`reset`/`stash`.** Interpreter: `venv/bin/python`. **Deploy note:** gcloud needs `CLOUDSDK_PYTHON=/opt/homebrew/opt/python@3.12/bin/python3.12` (local python3 is 3.13, which crashes gcloud's source upload).

---

## File Structure
- `docs/migrations/support.sql` (create) — `support_messages` table.
- `src/api/db.py` (modify) — `save_support_message`.
- `src/api/main.py` (modify) — `ContactIn` + `POST /contact`.
- `tests/helpers/fake_db.py` (modify) — `_support_messages` + `save_support_message`.
- `tests/integration/test_contact.py` (create) — endpoint tests.
- `src/app/contact/page.tsx` (create) — the form.
- `src/app/components/marketing/MarketingFooter.tsx` (modify) — Contact link.
- `src/app/terms/page.tsx`, `src/app/privacy/page.tsx` (modify) — repoint support refs to `/contact`.
- `src/app/report/[id]/ReportFeedback.tsx` (create) + `src/app/report/[id]/page.tsx` (modify) — feedback widget.

---

## Phase 1 — Contact backend (TDD)

### Task 1: `support_messages` migration + DB method + FakeDB

**Files:** Create `docs/migrations/support.sql`; Modify `src/api/db.py`, `tests/helpers/fake_db.py`.

- [ ] **Step 1: Migration** — `docs/migrations/support.sql`:
```sql
-- Beta: support contact messages. Run once in the Supabase SQL editor.
create table if not exists support_messages (
  id         uuid primary key default gen_random_uuid(),
  email      text,
  message    text not null,
  user_id    uuid,
  anon_id    uuid,
  created_at timestamptz not null default now()
);
create index if not exists support_messages_created_idx on support_messages (created_at desc);
alter table support_messages enable row level security;  -- service-role only; no client policies
```

- [ ] **Step 2: FakeDB** — in `tests/helpers/fake_db.py`, add `self._support_messages: list[dict] = []` in `__init__`, and a method (after `record_upgrade_intent`):
```python
    def save_support_message(self, email, message, user_id, anon_id) -> None:
        self._support_messages.append({
            "email": email, "message": message,
            "user_id": str(user_id) if user_id else None,
            "anon_id": str(anon_id) if anon_id else None,
        })
```

- [ ] **Step 3: SupabaseDB** — in `src/api/db.py`, add to `SupabaseDB` (after `list_transactions`), mirroring the existing `.insert(...).execute()` pattern (see `save_report` at line 45):
```python
    def save_support_message(self, email, message, user_id, anon_id) -> None:
        self.client.table("support_messages").insert({
            "email": email,
            "message": message,
            "user_id": str(user_id) if user_id else None,
            "anon_id": str(anon_id) if anon_id else None,
        }).execute()
```

- [ ] **Step 4: Commit** (with Task 2's tests — proceed to Task 2). 

---

### Task 2: `POST /contact` endpoint

**Files:** Modify `src/api/main.py`; Test `tests/integration/test_contact.py`.

- [ ] **Step 1: Write the failing test** — `tests/integration/test_contact.py`:
```python
import pytest
from fastapi.testclient import TestClient
from uuid import uuid4

from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_posthog import FakePostHog


@pytest.fixture
def client_and_fakes():
    db = FakeDB()
    ph = FakePostHog()
    from main import app, get_db, get_posthog
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_posthog] = lambda: ph
    yield TestClient(app), db, ph
    app.dependency_overrides.clear()


def _h():
    return {"X-Anon-Id": str(uuid4())}


def test_valid_message_stored_and_tracked(client_and_fakes):
    tc, db, ph = client_and_fakes
    r = tc.post("/contact", json={"email": "a@b.com", "message": "Help please"}, headers=_h())
    assert r.status_code == 200 and r.json()["ok"] is True
    assert len(db._support_messages) == 1
    assert db._support_messages[0]["message"] == "Help please"
    assert db._support_messages[0]["anon_id"] is not None
    ev = ph.find("support_request")
    assert len(ev) == 1 and ev[0]["properties"]["hasEmail"] is True and ev[0]["properties"]["length"] == 11


def test_honeypot_silently_dropped(client_and_fakes):
    tc, db, ph = client_and_fakes
    r = tc.post("/contact", json={"message": "spam", "company": "Acme Bots"}, headers=_h())
    assert r.status_code == 200 and r.json()["ok"] is True
    assert len(db._support_messages) == 0          # NOT stored
    assert len(ph.find("support_request")) == 0    # no event


def test_empty_message_422(client_and_fakes):
    tc, db, ph = client_and_fakes
    r = tc.post("/contact", json={"message": "   "}, headers=_h())
    assert r.status_code == 422 and r.json()["detail"]["code"] == "INVALID_MESSAGE"
    assert len(db._support_messages) == 0


def test_oversize_message_422(client_and_fakes):
    tc, db, ph = client_and_fakes
    r = tc.post("/contact", json={"message": "x" * 4001}, headers=_h())
    assert r.status_code == 422 and r.json()["detail"]["code"] == "INVALID_MESSAGE"
```

- [ ] **Step 2: Run → FAIL** (`cd /Users/chrissilver/Documents/ChartSage && venv/bin/python -m pytest tests/integration/test_contact.py -q`) — route 404 / no endpoint.

- [ ] **Step 3: Implement** — in `src/api/main.py` (confirm `BaseModel` is imported; `get_identity`/`get_db`/`get_posthog`/`Identity`/`HTTPException`/`Depends` already are). Add at the end of the endpoints section:
```python
class ContactIn(BaseModel):
    message: str
    email: str | None = None
    company: str | None = None   # honeypot: real users never fill this


@app.post("/contact")
async def contact(
    body: ContactIn,
    identity: Identity = Depends(get_identity),
    db: SupabaseDB = Depends(get_db),
    posthog: PostHogServer = Depends(get_posthog),
):
    # Honeypot: a filled "company" field means a bot — accept silently, store nothing.
    if body.company and body.company.strip():
        return {"ok": True}
    message = (body.message or "").strip()
    if not (1 <= len(message) <= 4000):
        raise HTTPException(status_code=422, detail={
            "code": "INVALID_MESSAGE", "message": "Message must be 1–4000 characters."})
    email = (body.email or "").strip()[:320] or None
    db.save_support_message(email, message, identity.user_id, identity.anon_id)
    posthog.capture(identity.distinct_id, "support_request",
                    {"hasEmail": bool(email), "length": len(message)})
    return {"ok": True}
```

- [ ] **Step 4: Run → PASS** (4 passed).

- [ ] **Step 5: Full suite + commit**
```bash
venv/bin/python -m pytest tests/unit tests/integration -p no:warnings -q   # all pass
git add docs/migrations/support.sql src/api/db.py src/api/main.py tests/helpers/fake_db.py tests/integration/test_contact.py
git commit -m "feat(beta): /contact endpoint + support_messages store + support_request event"
```

---

## Phase 2 — Contact frontend (implement → tsc → commit)

### Task 3: `/contact` page + footer link + legal repoint

**Files:** Create `src/app/contact/page.tsx`; Modify `src/app/components/marketing/MarketingFooter.tsx`, `src/app/terms/page.tsx`, `src/app/privacy/page.tsx`.

- [ ] **Step 1: Verify tokens** — `grep -nE "bg-canvas|card|text-ink|text-accent|btn-primary|border-line|font-display|bg-surface" src/app/globals.css tailwind.config.js | head -30` and skim `src/app/contact`-adjacent pages (`/login`, `/terms`) for the page-shell pattern. Use the real tokens.

- [ ] **Step 2: Create `src/app/contact/page.tsx`**:
```tsx
'use client';

import { useState } from 'react';
import { apiFetch } from '../lib/api';
import { posthog } from '../lib/posthog';

export default function ContactPage() {
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [company, setCompany] = useState('');   // honeypot
  const [status, setStatus] = useState<'idle' | 'sending' | 'sent' | 'error'>('idle');

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!message.trim()) { setStatus('error'); return; }
    setStatus('sending');
    try {
      const res = await apiFetch('/contact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, message, company }),
      });
      if (!res.ok) { setStatus('error'); return; }
      posthog.capture?.('contact_submitted', {});
      setStatus('sent');
    } catch {
      setStatus('error');
    }
  }

  return (
    <main className="min-h-screen bg-canvas text-ink">
      <div className="max-w-lg mx-auto px-6 py-16">
        <h1 className="font-display text-3xl font-semibold mb-2">Contact us</h1>
        <p className="text-sm text-ink-2 mb-8">Questions, bugs, or a deletion request? Send us a message and we’ll get back to you.</p>
        {status === 'sent' ? (
          <div className="card rounded-2xl p-6">
            <p className="text-ink">Thanks — we’ll get back to you.</p>
          </div>
        ) : (
          <form onSubmit={submit} className="space-y-4">
            <label className="block">
              <span className="block text-xs text-ink-3 mb-1">Your email (so we can reply)</span>
              <input type="email" value={email} onChange={(e) => setEmail(e.target.value)}
                     placeholder="you@example.com"
                     className="w-full rounded-lg bg-surface border border-line px-3 py-2 text-sm" />
            </label>
            <label className="block">
              <span className="block text-xs text-ink-3 mb-1">Message</span>
              <textarea value={message} onChange={(e) => setMessage(e.target.value)} required rows={6}
                        placeholder="How can we help?"
                        className="w-full rounded-lg bg-surface border border-line px-3 py-2 text-sm" />
            </label>
            {/* honeypot: hidden from users, bots fill it */}
            <input type="text" name="company" value={company} onChange={(e) => setCompany(e.target.value)}
                   tabIndex={-1} autoComplete="off" aria-hidden="true"
                   className="absolute -left-[9999px] h-0 w-0 opacity-0" />
            {status === 'error' && <p className="text-sm text-ember">Couldn’t send — please add a message and try again.</p>}
            <button type="submit" disabled={status === 'sending'} className="btn btn-primary">
              {status === 'sending' ? 'Sending…' : 'Send message'}
            </button>
          </form>
        )}
      </div>
    </main>
  );
}
```
(Adjust token classes — `bg-canvas`/`bg-surface`/`text-ink*`/`btn btn-primary`/`border-line`/`text-ember`/`font-display` — to the verified names.)

- [ ] **Step 3: Footer link** — in `MarketingFooter.tsx`, add to the link row (alongside Terms/Privacy, matching their `className="hover:text-ink transition-colors"`):
```tsx
          <a href="/contact" className="hover:text-ink transition-colors">Contact</a>
```

- [ ] **Step 4: Repoint legal refs** — replace the three `support@chartsage.app` mentions with a link to `/contact`:
  - `src/app/terms/page.tsx:28`: `Questions about these Terms? <a href="/contact" className="text-accent underline">Contact us</a>.`
  - `src/app/privacy/page.tsx:20`: `You can request deletion of your account and associated data by <a href="/contact" className="text-accent underline">contacting us</a>. Self-serve deletion is coming.`
  - `src/app/privacy/page.tsx:26`: `Questions about your privacy? <a href="/contact" className="text-accent underline">Contact us</a>.`
  (Use the page's real accent/link token.)

- [ ] **Step 5: tsc + commit**
```bash
cd /Users/chrissilver/Documents/ChartSage && npx tsc --noEmit   # exit 0
git add src/app/contact/page.tsx src/app/components/marketing/MarketingFooter.tsx src/app/terms/page.tsx src/app/privacy/page.tsx
git commit -m "feat(beta): /contact form + footer link + legal contact links"
```

---

## Phase 3 — Feedback widget (implement → tsc → commit)

### Task 4: Report feedback widget

**Files:** Create `src/app/report/[id]/ReportFeedback.tsx`; Modify `src/app/report/[id]/page.tsx`.

- [ ] **Step 1: Create `src/app/report/[id]/ReportFeedback.tsx`**:
```tsx
'use client';

import { useState } from 'react';
import { posthog } from '../../lib/posthog';

export default function ReportFeedback({ reportId }: { reportId: string }) {
  const key = `cs_feedback_${reportId}`;
  const [done, setDone] = useState<boolean>(
    typeof window !== 'undefined' && sessionStorage.getItem(key) === '1',
  );
  const [rating, setRating] = useState<'up' | 'down' | null>(null);
  const [comment, setComment] = useState('');

  function pick(r: 'up' | 'down') { setRating(r); }

  function send() {
    posthog.capture?.('report_feedback', { rating, comment: comment.trim() || undefined, reportId });
    try { sessionStorage.setItem(key, '1'); } catch {}
    setDone(true);
  }

  if (done) {
    return <p className="text-sm text-ink-3 text-center py-4">Thanks for the feedback!</p>;
  }

  return (
    <div className="card rounded-xl p-4 flex flex-col items-center gap-3 my-6">
      <span className="text-sm text-ink-2">Was this report useful?</span>
      <div className="flex gap-2">
        <button onClick={() => pick('up')}
                className={`btn btn-ghost ${rating === 'up' ? 'border-accent text-accent' : ''}`}>👍</button>
        <button onClick={() => pick('down')}
                className={`btn btn-ghost ${rating === 'down' ? 'border-accent text-accent' : ''}`}>👎</button>
      </div>
      {rating && (
        <div className="w-full max-w-md flex flex-col gap-2">
          <textarea value={comment} onChange={(e) => setComment(e.target.value)} rows={2}
                    placeholder="Anything we could improve? (optional)"
                    className="w-full rounded-lg bg-surface border border-line px-3 py-2 text-sm" />
          <button onClick={send} className="btn btn-primary self-end">Send</button>
        </div>
      )}
    </div>
  );
}
```
(Adjust token classes to the verified names; match the `btn`/`card` styling used elsewhere.)

- [ ] **Step 2: Wire into the report page** — in `src/app/report/[id]/page.tsx`, import `ReportFeedback` and render `<ReportFeedback reportId={sessionId} />` near the bottom of `ReportView` — after the charts/sidebar block, just above the footer line that shows `Report id: {sessionId.slice(0, 8)}` (~line 166). It's a client component inside the already-client `ReportView`.

- [ ] **Step 3: tsc + commit**
```bash
cd /Users/chrissilver/Documents/ChartSage && npx tsc --noEmit   # exit 0
git add src/app/report/[id]/ReportFeedback.tsx "src/app/report/[id]/page.tsx"
git commit -m "feat(beta): report feedback widget (report_feedback event)"
```

---

## Phase 4 — Build + deploy (production — requires explicit user authorization)

### Task 5: Verify + deploy

**Files:** none (ops).

- [ ] **Step 1: Full verification**
```bash
cd /Users/chrissilver/Documents/ChartSage
venv/bin/python -m pytest tests/unit tests/integration -p no:warnings -q   # all pass
npx tsc --noEmit                                                           # exit 0
PATH="/opt/homebrew/opt/node@22/bin:$PATH" npm run build 2>/dev/null || echo "local build blocked by Node 23 — Vercel is the build gate (it type-checks on push)"
```

- [ ] **Step 2: Run the migration** — execute `docs/migrations/support.sql` in the Supabase SQL editor. (User action.)

- [ ] **Step 3: Deploy backend** (user-authorized):
```bash
SUPA=$(grep -E '^SUPABASE_URL=' .env | cut -d= -f2-)
CLOUDSDK_PYTHON="/opt/homebrew/opt/python@3.12/bin/python3.12" gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_SUPABASE_URL="$SUPA",_TAG="$(git rev-parse --short HEAD)"
```
(Do NOT source `FRONTEND_BASE_URL` — the cloudbuild default is correct.)

- [ ] **Step 4: Deploy frontend** — merge `beta-enablers` → main + push (Vercel builds `/contact` + the feedback widget); confirm the deployment goes Ready.
```bash
git checkout main && git merge --ff-only beta-enablers && git push origin main
```

- [ ] **Step 5: Smoke** (against prod, with `X-Anon-Id`):
```bash
BASE="https://chartsage-backend-112026133429.us-central1.run.app"
curl -s -X POST "$BASE/contact" -H "X-Anon-Id: $(python3 -c 'import uuid;print(uuid.uuid4())')" \
  -H "Content-Type: application/json" -d '{"email":"smoke@test.com","message":"smoke test"}'
# expect {"ok":true}; then confirm a support_messages row in Supabase + the support_request event in PostHog.
# Honeypot: same POST with "company":"x" -> {"ok":true} but NO new row.
```
Then load a report in the browser → rate 👍/👎 + comment → confirm the `report_feedback` event in PostHog.

---

## Self-Review

**Spec coverage:** support_messages migration + `save_support_message` (Task 1) ✓ · `POST /contact` with honeypot/validation/identity/event (Task 2) ✓ · `/contact` page + footer link + legal repoint (Task 3) ✓ · feedback widget + report-page wiring + `report_feedback` event (Task 4) ✓ · build + migration + deploy + smoke (Task 5) ✓.

**Placeholder scan:** every code/test step has concrete content; the only judgment calls (token-class verification in Tasks 3–4, the exact footer/legal edit lines) are explicit "verify against the real file" instructions.

**Type consistency:** `save_support_message(email, message, user_id, anon_id)` (db.py + FakeDB + endpoint), `ContactIn{message, email, company}`, the `support_request {hasEmail, length}` + `report_feedback {rating, comment, reportId}` event shapes, and `ReportFeedback({reportId})` are identical across the tasks and tests.
