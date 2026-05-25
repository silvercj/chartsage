# SP1 Production Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take ChartSage from "runs locally" to "lives on Cloud Run + Vercel + Supabase + PostHog" with persistent reports, anonymous identity (1-report-per-cookie cap), and full event tracking — at <$10/mo at launch.

**Architecture:** Replace Redis with Supabase Postgres (reports) + Supabase Storage (CSV blobs). Add a long-lived `chartsage_anon` UUID cookie set by Next.js middleware; every API call carries it as `X-Anon-Id`. The backend enforces a 1-report-per-anon cap. PostHog captures 27 events (snake_case names, camelCase properties), with cost-bearing events fired server-side so ad blockers can't hide spend data. Deployment artifacts (Dockerfile, cloudbuild.yaml, vercel.json) ship with the code; provisioning steps are a runbook the user executes once.

**Tech Stack:** Python 3.11+ (FastAPI, pandas, Pydantic v2, Playwright, supabase-py 2.x, posthog 3.x), Next.js 14 (middleware, @dnd-kit), Supabase Postgres + Storage + PostHog cloud + Google Cloud Run + Vercel.

See [the design spec](../specs/2026-05-24-sp1-foundation-design.md) (commit `02c4c58`).

---

## File Structure

### Backend (new / modified)

```
src/api/
├── main.py                # rewritten: swaps Redis for db.py + storage.py; anon-limit check; PostHog calls
├── db.py                  # NEW — Supabase Postgres wrapper
├── storage.py             # NEW — Supabase Storage wrapper
├── posthog_server.py      # NEW — server-side PostHog client (silently swallows failures)
├── deps.py                # NEW — X-Anon-Id FastAPI dependency
└── llm_config.py          # +MODEL_PRICING table, +estimate_cost_usd helper
```

### Frontend (new / modified)

```
src/
├── middleware.ts                            # NEW — set chartsage_anon cookie if missing
├── app/
│   ├── layout.tsx                           # +import lib/posthog
│   ├── page.tsx                             # apiFetch + redirect to /anon-limit on 403
│   ├── anon-limit/
│   │   └── page.tsx                         # NEW — placeholder for "you've used your free report"
│   ├── lib/
│   │   ├── anon.ts                          # NEW — read chartsage_anon cookie
│   │   ├── posthog.ts                       # NEW — initialize posthog-js
│   │   └── api.ts                           # NEW — apiFetch wrapper with X-Anon-Id header
│   └── report/[id]/
│       ├── useReportLayout.ts               # apiFetch
│       └── Toolbar.tsx                      # apiFetch + PostHog event capture
```

### Tests

```
tests/
├── helpers/
│   ├── fake_db.py                           # NEW — in-memory db matching db.py interface
│   ├── fake_storage.py                      # NEW — in-memory storage
│   └── fake_posthog.py                      # NEW — captures events for assertions
├── unit/
│   ├── test_db.py                           # NEW — round-trip + anon counting
│   ├── test_storage.py                      # NEW — upload + download + missing key
│   ├── test_posthog_server.py               # NEW — captures, swallows errors
│   ├── test_deps.py                         # NEW — anon-id header parsing
│   └── test_llm_config_pricing.py           # NEW — cost estimator
└── integration/
    ├── test_anon_limit.py                   # NEW — 1-report-per-anon enforcement
    ├── test_storage_failure.py              # NEW — upload error doesn't orphan a row
    ├── test_posthog_events.py               # NEW — cost-bearing events fire with right keys
    ├── test_api_layout.py                   # MODIFY — FakeRedis → FakeDB + FakeStorage; X-Anon-Id
    └── test_api_errors.py                   # MODIFY — same swap
```

### Deployment (new at repo root)

```
Dockerfile
.dockerignore
cloudbuild.yaml
vercel.json
.env.production.example
requirements.txt           # +supabase==2.4.3, +posthog==3.5.0
README.md                  # +Deploying section
```

---

## Phase 1 — Cost estimator + supabase/posthog deps

### Task 1: Pin dependencies + cost estimator

**Files:**
- Modify: `requirements.txt`
- Modify: `src/api/llm_config.py`
- Create: `tests/unit/test_llm_config_pricing.py`

- [ ] **Step 1: Add deps to requirements.txt**

Append to `requirements.txt`:

```
supabase==2.4.3
posthog==3.5.0
```

Install:

```bash
source venv/bin/activate
pip install supabase==2.4.3 posthog==3.5.0
```

- [ ] **Step 2: Write the pricing test**

Write `tests/unit/test_llm_config_pricing.py`:

```python
import pytest
from llm_config import MODEL_PRICING, estimate_cost_usd


def test_haiku_pricing_known():
    cost = estimate_cost_usd("claude-haiku-4-5-20251001",
                             input_tokens=1_000_000,
                             output_tokens=0,
                             cache_read_tokens=0)
    assert cost == pytest.approx(1.0)


def test_haiku_output_pricing():
    cost = estimate_cost_usd("claude-haiku-4-5-20251001",
                             input_tokens=0,
                             output_tokens=1_000_000,
                             cache_read_tokens=0)
    assert cost == pytest.approx(5.0)


def test_cache_read_cheaper_than_normal_input():
    no_cache = estimate_cost_usd("claude-haiku-4-5-20251001",
                                 input_tokens=1_000_000, output_tokens=0, cache_read_tokens=0)
    all_cached = estimate_cost_usd("claude-haiku-4-5-20251001",
                                   input_tokens=1_000_000, output_tokens=0, cache_read_tokens=1_000_000)
    assert all_cached < no_cache
    assert all_cached == pytest.approx(0.1)


def test_unknown_model_falls_back_to_haiku():
    cost = estimate_cost_usd("some-mystery-model",
                             input_tokens=1_000_000, output_tokens=0)
    assert cost == pytest.approx(1.0)


def test_combines_input_output_cache_read():
    cost = estimate_cost_usd("claude-haiku-4-5-20251001",
                             input_tokens=2_000_000,
                             output_tokens=500_000,
                             cache_read_tokens=500_000)
    # uncached input: 1.5M @ $1.0/M = $1.50
    # cache_read:     0.5M @ $0.10/M = $0.05
    # output:         0.5M @ $5.0/M = $2.50
    # total = $4.05
    assert cost == pytest.approx(4.05)


def test_model_pricing_has_three_models():
    assert "claude-haiku-4-5-20251001" in MODEL_PRICING
    assert "claude-sonnet-4-6" in MODEL_PRICING
    assert "claude-opus-4-7" in MODEL_PRICING
```

- [ ] **Step 3: Run, expect ImportError**

```bash
PYTHONPATH=src/api pytest tests/unit/test_llm_config_pricing.py -v
```

Expected: `ImportError: cannot import name 'MODEL_PRICING' from 'llm_config'`.

- [ ] **Step 4: Add MODEL_PRICING + estimate_cost_usd to llm_config.py**

Append to `src/api/llm_config.py`:

```python
# Per 1M tokens, in USD
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0, "cache_read": 0.1},
    "claude-sonnet-4-6":         {"input": 3.0, "output": 15.0, "cache_read": 0.3},
    "claude-opus-4-7":           {"input": 15.0, "output": 75.0, "cache_read": 1.5},
}


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> float:
    """Estimate USD cost of a Claude API call from token counts.

    cache_read_tokens are billed at a fraction of the input rate.
    Returns USD to 6 decimal places.
    """
    rates = MODEL_PRICING.get(model, MODEL_PRICING["claude-haiku-4-5-20251001"])
    uncached_input = max(0, input_tokens - cache_read_tokens)
    cost = (
        uncached_input * rates["input"] / 1_000_000
        + cache_read_tokens * rates["cache_read"] / 1_000_000
        + output_tokens * rates["output"] / 1_000_000
    )
    return round(cost, 6)
```

- [ ] **Step 5: Run tests, expect pass**

```bash
PYTHONPATH=src/api pytest tests/unit/test_llm_config_pricing.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/api/llm_config.py tests/unit/test_llm_config_pricing.py
git commit -m "feat: add MODEL_PRICING + estimate_cost_usd; +supabase +posthog deps"
```

---

## Phase 2 — Data layer (db + storage + posthog + deps)

### Task 2: Fake-db helper

**Files:**
- Create: `tests/helpers/fake_db.py`

- [ ] **Step 1: Implement the fake**

Write `tests/helpers/fake_db.py`:

```python
"""In-memory db matching the db.py interface, for integration tests."""
from copy import deepcopy
from typing import Optional
from uuid import UUID


class FakeDB:
    def __init__(self):
        self._rows: dict[str, dict] = {}   # report_id -> row dict

    def save_report(
        self,
        report_id: str,
        anon_id: Optional[UUID],
        user_id: Optional[UUID],
        report_json: dict,
        csv_storage_key: Optional[str],
        title: str,
    ) -> None:
        self._rows[report_id] = {
            "id": report_id,
            "anon_id": str(anon_id) if anon_id else None,
            "user_id": str(user_id) if user_id else None,
            "report_json": deepcopy(report_json),
            "csv_storage_key": csv_storage_key,
            "title": title,
        }

    def get_report(self, report_id: str) -> Optional[dict]:
        row = self._rows.get(report_id)
        return deepcopy(row) if row else None

    def update_report_json(self, report_id: str, report_json: dict) -> bool:
        if report_id not in self._rows:
            return False
        self._rows[report_id]["report_json"] = deepcopy(report_json)
        return True

    def update_layout(self, report_id: str, layout: list[dict]) -> bool:
        if report_id not in self._rows:
            return False
        self._rows[report_id]["report_json"]["layout"] = deepcopy(layout)
        return True

    def count_anon_reports(self, anon_id: UUID) -> int:
        return sum(
            1 for r in self._rows.values()
            if r["anon_id"] == str(anon_id) and r["user_id"] is None
        )
```

- [ ] **Step 2: Commit (no test yet; exercised by Task 3)**

```bash
git add tests/helpers/fake_db.py
git commit -m "test: FakeDB helper matching db.py interface"
```

---

### Task 3: db.py (real Supabase client)

**Files:**
- Create: `src/api/db.py`
- Create: `tests/unit/test_db.py`

- [ ] **Step 1: Write tests against the interface (using FakeDB)**

Write `tests/unit/test_db.py`:

```python
"""Tests the db.py interface contract using FakeDB.

The real SupabaseDB has the same interface; its specific Postgres behavior
is exercised by integration tests and the manual smoke runbook.
"""
import pytest
from uuid import uuid4
from tests.helpers.fake_db import FakeDB


def _sample_report():
    return {
        "generated_at": "2026-05-24T00:00:00",
        "summary": "Sample.",
        "data_quality": [],
        "charts": [],
        "layout": [],
        "metadata": {},
    }


def test_save_and_get_round_trip():
    db = FakeDB()
    anon = uuid4()
    db.save_report("r1", anon, None, _sample_report(), "r1.csv", "Sample")
    row = db.get_report("r1")
    assert row is not None
    assert row["id"] == "r1"
    assert row["anon_id"] == str(anon)
    assert row["user_id"] is None
    assert row["csv_storage_key"] == "r1.csv"
    assert row["title"] == "Sample"
    assert row["report_json"]["summary"] == "Sample."


def test_get_missing_returns_none():
    db = FakeDB()
    assert db.get_report("nope") is None


def test_count_anon_reports():
    db = FakeDB()
    a, b = uuid4(), uuid4()
    db.save_report("r1", a, None, _sample_report(), None, "")
    db.save_report("r2", a, None, _sample_report(), None, "")
    db.save_report("r3", b, None, _sample_report(), None, "")
    assert db.count_anon_reports(a) == 2
    assert db.count_anon_reports(b) == 1
    assert db.count_anon_reports(uuid4()) == 0


def test_count_excludes_reports_with_user_id():
    db = FakeDB()
    a = uuid4()
    user = uuid4()
    db.save_report("r1", a, None, _sample_report(), None, "")
    db.save_report("r2", a, user, _sample_report(), None, "")   # migrated
    assert db.count_anon_reports(a) == 1


def test_update_layout():
    db = FakeDB()
    anon = uuid4()
    db.save_report("r1", anon, None, _sample_report(), None, "")
    new_layout = [{"chart_id": "c1", "position": "main", "order": 0}]
    assert db.update_layout("r1", new_layout) is True
    assert db.get_report("r1")["report_json"]["layout"] == new_layout


def test_update_layout_missing_returns_false():
    db = FakeDB()
    assert db.update_layout("nope", []) is False


def test_update_report_json_overwrites():
    db = FakeDB()
    anon = uuid4()
    db.save_report("r1", anon, None, _sample_report(), None, "")
    new_report = _sample_report()
    new_report["summary"] = "Updated."
    assert db.update_report_json("r1", new_report) is True
    assert db.get_report("r1")["report_json"]["summary"] == "Updated."


def test_round_trip_does_not_share_references():
    db = FakeDB()
    anon = uuid4()
    original = _sample_report()
    db.save_report("r1", anon, None, original, None, "")
    original["summary"] = "MUTATED"   # mutate the input after save
    assert db.get_report("r1")["report_json"]["summary"] == "Sample."   # unchanged
```

- [ ] **Step 2: Run, expect pass (FakeDB already exists from Task 2)**

```bash
PYTHONPATH=src/api pytest tests/unit/test_db.py -v
```

Expected: 7 passed.

- [ ] **Step 3: Implement the real db.py**

Write `src/api/db.py`:

```python
"""Supabase Postgres client.

The interface mirrors the FakeDB helper used in tests so we can swap
between real and fake implementations through dependency injection.
"""
import os
from typing import Optional
from uuid import UUID

from supabase import create_client, Client


_SUPABASE_URL = os.environ.get("SUPABASE_URL")
_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")


def _client() -> Client:
    if not _SUPABASE_URL or not _SERVICE_ROLE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set"
        )
    return create_client(_SUPABASE_URL, _SERVICE_ROLE_KEY)


class SupabaseDB:
    """Sync Postgres wrapper. Called from async endpoints — small queries
    block the event loop briefly, which is acceptable at this scale.
    """

    def __init__(self, client: Optional[Client] = None):
        self.client = client or _client()

    def save_report(
        self,
        report_id: str,
        anon_id: Optional[UUID],
        user_id: Optional[UUID],
        report_json: dict,
        csv_storage_key: Optional[str],
        title: str,
    ) -> None:
        self.client.table("reports").insert({
            "id": report_id,
            "anon_id": str(anon_id) if anon_id else None,
            "user_id": str(user_id) if user_id else None,
            "report_json": report_json,
            "csv_storage_key": csv_storage_key,
            "title": title,
        }).execute()

    def get_report(self, report_id: str) -> Optional[dict]:
        res = self.client.table("reports").select("*").eq("id", report_id).limit(1).execute()
        if not res.data:
            return None
        return res.data[0]

    def update_report_json(self, report_id: str, report_json: dict) -> bool:
        res = (self.client.table("reports")
               .update({"report_json": report_json, "updated_at": "now()"})
               .eq("id", report_id)
               .execute())
        return len(res.data) > 0

    def update_layout(self, report_id: str, layout: list[dict]) -> bool:
        row = self.get_report(report_id)
        if not row:
            return False
        report_json = row["report_json"]
        report_json["layout"] = layout
        return self.update_report_json(report_id, report_json)

    def count_anon_reports(self, anon_id: UUID) -> int:
        res = (self.client.table("reports")
               .select("id", count="exact")
               .eq("anon_id", str(anon_id))
               .is_("user_id", "null")
               .execute())
        return res.count or 0
```

- [ ] **Step 4: Quick syntax/import check**

```bash
PYTHONPATH=src/api python -c "from db import SupabaseDB; print('ok')"
```

Expected: `ok` (the constructor isn't called, so missing env vars don't error).

- [ ] **Step 5: Commit**

```bash
git add src/api/db.py tests/unit/test_db.py
git commit -m "feat: SupabaseDB wrapper + interface tests"
```

---

### Task 4: Fake-storage helper + storage.py

**Files:**
- Create: `tests/helpers/fake_storage.py`
- Create: `src/api/storage.py`
- Create: `tests/unit/test_storage.py`

- [ ] **Step 1: Implement FakeStorage**

Write `tests/helpers/fake_storage.py`:

```python
"""In-memory storage matching the storage.py interface."""


class StorageError(Exception):
    pass


class FakeStorage:
    def __init__(self):
        self._objects: dict[str, bytes] = {}

    def upload_csv(self, report_id: str, csv_bytes: bytes) -> str:
        key = f"{report_id}.csv"
        self._objects[key] = csv_bytes
        return key

    def download_csv(self, report_id: str) -> bytes:
        key = f"{report_id}.csv"
        if key not in self._objects:
            raise StorageError(f"missing object: {key}")
        return self._objects[key]

    def delete_csv(self, report_id: str) -> None:
        self._objects.pop(f"{report_id}.csv", None)

    def fail_next_upload(self):
        """Test hook: make the next upload_csv raise StorageError."""
        self._fail_next = True

    def __init_subclass__(cls):
        # Not used; here only as a placeholder for future expansion
        pass
```

Wait — fix that last bit; `fail_next_upload` needs proper integration. Replace the class with the cleaner version:

```python
"""In-memory storage matching the storage.py interface."""


class StorageError(Exception):
    pass


class FakeStorage:
    def __init__(self):
        self._objects: dict[str, bytes] = {}
        self._fail_next_upload = False

    def upload_csv(self, report_id: str, csv_bytes: bytes) -> str:
        if self._fail_next_upload:
            self._fail_next_upload = False
            raise StorageError("simulated upload failure")
        key = f"{report_id}.csv"
        self._objects[key] = csv_bytes
        return key

    def download_csv(self, report_id: str) -> bytes:
        key = f"{report_id}.csv"
        if key not in self._objects:
            raise StorageError(f"missing object: {key}")
        return self._objects[key]

    def delete_csv(self, report_id: str) -> None:
        self._objects.pop(f"{report_id}.csv", None)

    def fail_next_upload(self) -> None:
        self._fail_next_upload = True
```

- [ ] **Step 2: Write storage tests (against FakeStorage)**

Write `tests/unit/test_storage.py`:

```python
import pytest
from tests.helpers.fake_storage import FakeStorage, StorageError


def test_upload_returns_key():
    s = FakeStorage()
    key = s.upload_csv("abc", b"col1,col2\n1,2\n")
    assert key == "abc.csv"


def test_download_returns_uploaded_bytes():
    s = FakeStorage()
    s.upload_csv("abc", b"col1,col2\n1,2\n")
    assert s.download_csv("abc") == b"col1,col2\n1,2\n"


def test_download_missing_raises():
    s = FakeStorage()
    with pytest.raises(StorageError) as exc:
        s.download_csv("nope")
    assert "missing" in str(exc.value)


def test_delete_csv():
    s = FakeStorage()
    s.upload_csv("abc", b"x")
    s.delete_csv("abc")
    with pytest.raises(StorageError):
        s.download_csv("abc")


def test_delete_missing_is_noop():
    s = FakeStorage()
    s.delete_csv("never_existed")   # must not raise


def test_fail_next_upload_hook():
    s = FakeStorage()
    s.fail_next_upload()
    with pytest.raises(StorageError):
        s.upload_csv("abc", b"x")
    # subsequent uploads work again
    s.upload_csv("def", b"y")
    assert s.download_csv("def") == b"y"
```

- [ ] **Step 3: Run tests, expect pass**

```bash
PYTHONPATH=src/api pytest tests/unit/test_storage.py -v
```

Expected: 6 passed.

- [ ] **Step 4: Implement real storage.py**

Write `src/api/storage.py`:

```python
"""Supabase Storage wrapper.

Bucket name is fixed at csv-inputs. Keys are {report_id}.csv.
"""
import os
from typing import Optional
from supabase import create_client, Client


BUCKET = "csv-inputs"


class StorageError(Exception):
    pass


def _client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
    return create_client(url, key)


class SupabaseStorage:
    def __init__(self, client: Optional[Client] = None):
        self.client = client or _client()

    def upload_csv(self, report_id: str, csv_bytes: bytes) -> str:
        key = f"{report_id}.csv"
        try:
            self.client.storage.from_(BUCKET).upload(
                path=key,
                file=csv_bytes,
                file_options={"content-type": "text/csv", "upsert": "true"},
            )
        except Exception as e:
            raise StorageError(f"upload failed: {e}") from e
        return key

    def download_csv(self, report_id: str) -> bytes:
        key = f"{report_id}.csv"
        try:
            return self.client.storage.from_(BUCKET).download(key)
        except Exception as e:
            raise StorageError(f"download failed: {e}") from e

    def delete_csv(self, report_id: str) -> None:
        key = f"{report_id}.csv"
        try:
            self.client.storage.from_(BUCKET).remove([key])
        except Exception:
            pass   # delete is best-effort
```

- [ ] **Step 5: Syntax check**

```bash
PYTHONPATH=src/api python -c "from storage import SupabaseStorage, StorageError; print('ok')"
```

Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add tests/helpers/fake_storage.py tests/unit/test_storage.py src/api/storage.py
git commit -m "feat: SupabaseStorage wrapper + FakeStorage + interface tests"
```

---

### Task 5: posthog_server.py + FakePostHog

**Files:**
- Create: `tests/helpers/fake_posthog.py`
- Create: `src/api/posthog_server.py`
- Create: `tests/unit/test_posthog_server.py`

- [ ] **Step 1: Implement FakePostHog**

Write `tests/helpers/fake_posthog.py`:

```python
"""In-memory PostHog client that records events without sending them."""


class FakePostHog:
    def __init__(self):
        self.events: list[dict] = []

    def capture(self, distinct_id: str, event: str, properties: dict | None = None) -> None:
        self.events.append({
            "distinct_id": str(distinct_id),
            "event": event,
            "properties": properties or {},
        })

    def find(self, event_name: str) -> list[dict]:
        """Helper for tests: return all events with a given name."""
        return [e for e in self.events if e["event"] == event_name]
```

- [ ] **Step 2: Write tests**

Write `tests/unit/test_posthog_server.py`:

```python
import logging
from unittest.mock import MagicMock
import pytest
from tests.helpers.fake_posthog import FakePostHog


def test_capture_records_event():
    p = FakePostHog()
    p.capture("anon-uuid-1", "report_viewed", {"reportId": "r1"})
    assert p.events == [{
        "distinct_id": "anon-uuid-1",
        "event": "report_viewed",
        "properties": {"reportId": "r1"},
    }]


def test_capture_without_properties():
    p = FakePostHog()
    p.capture("u", "landing_viewed")
    assert p.events[0]["properties"] == {}


def test_find_filters_by_event_name():
    p = FakePostHog()
    p.capture("u", "a")
    p.capture("u", "b")
    p.capture("u", "a")
    assert len(p.find("a")) == 2
    assert len(p.find("c")) == 0


def test_server_client_swallows_errors(caplog):
    """The real PostHogServer must never let analytics failures break product flow."""
    from posthog_server import PostHogServer

    boom = MagicMock()
    boom.capture.side_effect = RuntimeError("posthog is down")
    server = PostHogServer(_client=boom)

    with caplog.at_level(logging.WARNING):
        server.capture("u", "some_event", {"k": "v"})   # must not raise

    # Logged a warning
    assert any("posthog" in r.message.lower() for r in caplog.records)


def test_server_client_passes_through_to_underlying():
    from posthog_server import PostHogServer

    mock_client = MagicMock()
    server = PostHogServer(_client=mock_client)
    server.capture("u", "evt", {"reportId": "r1"})

    mock_client.capture.assert_called_once_with(
        distinct_id="u", event="evt", properties={"reportId": "r1"},
    )
```

- [ ] **Step 3: Run, expect ImportError on `posthog_server`**

```bash
PYTHONPATH=src/api pytest tests/unit/test_posthog_server.py -v
```

Expected: 2 fakes pass; 2 (server) fail with ImportError.

- [ ] **Step 4: Implement posthog_server.py**

Write `src/api/posthog_server.py`:

```python
"""Server-side PostHog wrapper.

Wraps the official posthog Python SDK with a single rule: analytics
must never break product flow. Errors are caught and logged at WARN.
"""
import logging
import os
from typing import Any, Optional


_POSTHOG_KEY = os.environ.get("POSTHOG_API_KEY")
_POSTHOG_HOST = os.environ.get("POSTHOG_HOST", "https://us.i.posthog.com")


def _build_default_client() -> Optional[Any]:
    if not _POSTHOG_KEY:
        return None
    from posthog import Posthog
    return Posthog(project_api_key=_POSTHOG_KEY, host=_POSTHOG_HOST)


class PostHogServer:
    def __init__(self, _client: Any = None):
        self.client = _client if _client is not None else _build_default_client()

    def capture(
        self,
        distinct_id: str,
        event: str,
        properties: Optional[dict] = None,
    ) -> None:
        if self.client is None:
            return   # not configured; silent no-op
        try:
            self.client.capture(
                distinct_id=str(distinct_id),
                event=event,
                properties=properties or {},
            )
        except Exception as e:
            logging.warning("[POSTHOG] capture failed for %s: %s", event, e)
```

- [ ] **Step 5: Run tests, expect 5 pass**

```bash
PYTHONPATH=src/api pytest tests/unit/test_posthog_server.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add tests/helpers/fake_posthog.py tests/unit/test_posthog_server.py src/api/posthog_server.py
git commit -m "feat: PostHogServer wrapper with silent error swallowing + FakePostHog"
```

---

### Task 6: deps.py (X-Anon-Id dependency)

**Files:**
- Create: `src/api/deps.py`
- Create: `tests/unit/test_deps.py`

- [ ] **Step 1: Write tests**

Write `tests/unit/test_deps.py`:

```python
import pytest
from uuid import UUID
from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi import FastAPI, Depends


def _make_app(get_anon_id):
    app = FastAPI()

    @app.get("/echo")
    def echo(anon_id: UUID = Depends(get_anon_id)):
        return {"anon_id": str(anon_id)}

    return TestClient(app)


def test_valid_uuid_header_passes():
    from deps import get_anon_id
    client = _make_app(get_anon_id)
    valid = "11111111-1111-1111-1111-111111111111"
    res = client.get("/echo", headers={"X-Anon-Id": valid})
    assert res.status_code == 200
    assert res.json() == {"anon_id": valid}


def test_missing_header_returns_400():
    from deps import get_anon_id
    client = _make_app(get_anon_id)
    res = client.get("/echo")
    assert res.status_code == 400
    assert "MISSING_ANON_ID" in res.text


def test_malformed_uuid_returns_400():
    from deps import get_anon_id
    client = _make_app(get_anon_id)
    res = client.get("/echo", headers={"X-Anon-Id": "not-a-uuid"})
    assert res.status_code == 400
    assert "INVALID_ANON_ID" in res.text


def test_empty_header_returns_400():
    from deps import get_anon_id
    client = _make_app(get_anon_id)
    res = client.get("/echo", headers={"X-Anon-Id": ""})
    assert res.status_code == 400
```

- [ ] **Step 2: Run, expect ImportError**

```bash
PYTHONPATH=src/api pytest tests/unit/test_deps.py -v
```

Expected: ImportError on `deps`.

- [ ] **Step 3: Implement deps.py**

Write `src/api/deps.py`:

```python
"""FastAPI dependencies shared across endpoints."""
from uuid import UUID
from fastapi import Header, HTTPException


def get_anon_id(x_anon_id: str | None = Header(None)) -> UUID:
    """Parse the X-Anon-Id header into a UUID; reject missing or malformed."""
    if not x_anon_id:
        raise HTTPException(
            status_code=400,
            detail={"code": "MISSING_ANON_ID",
                    "message": "X-Anon-Id header is required."},
        )
    try:
        return UUID(x_anon_id)
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_ANON_ID",
                    "message": "X-Anon-Id is not a valid UUID."},
        )
```

- [ ] **Step 4: Run tests, expect pass**

```bash
PYTHONPATH=src/api pytest tests/unit/test_deps.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/deps.py tests/unit/test_deps.py
git commit -m "feat: get_anon_id FastAPI dependency"
```

---

## Phase 3 — Backend main.py rewrite

### Task 7: Swap Redis for db + storage in /generate-report; add anon-limit + PostHog

**Files:**
- Modify: `src/api/main.py`
- Modify: `tests/integration/test_api_errors.py`
- Create: `tests/integration/test_anon_limit.py`

This is the biggest single change. It touches the upload endpoint, swaps Redis usage, adds the anon-limit check, and fires server-side PostHog events.

- [ ] **Step 1: Update existing tests in test_api_errors.py to use FakeDB + FakeStorage + X-Anon-Id**

Open `tests/integration/test_api_errors.py`. Replace the fixture and all test functions with:

```python
import io
import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from uuid import uuid4

from tests.helpers.fake_claude import FakeClaude, tool_use
from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_storage import FakeStorage
from tests.helpers.fake_posthog import FakePostHog


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _scripted_fake():
    chart_calls = [
        tool_use("frequency_bar_chart",
                 {"column": "region", "title": f"T{i}", "intent": f"i{i}"},
                 id_=f"tu_{i}")
        for i in range(10)
    ]
    return FakeClaude([
        {"tool_calls": chart_calls},
        {"tool_calls": [tool_use(
            "submit_narrative",
            {"summary": "S.", "captions": [f"c{i}" for i in range(10)], "data_quality": []},
        )]},
    ])


@pytest.fixture
def anon_id():
    return str(uuid4())


@pytest.fixture
def client(sales, anon_id):
    """Boot the app with fakes injected via dependency_overrides."""
    fake_claude = _scripted_fake()
    fake_db = FakeDB()
    fake_storage = FakeStorage()
    fake_posthog = FakePostHog()

    from main import app, get_claude_client, get_db, get_storage, get_posthog
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=fake_claude)
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[get_storage] = lambda: fake_storage
    app.dependency_overrides[get_posthog] = lambda: fake_posthog

    yield TestClient(app), fake_db, fake_storage, fake_posthog
    app.dependency_overrides.clear()


def _headers(anon_id):
    return {"X-Anon-Id": anon_id}


def test_happy_path_post_then_get(client, sales, anon_id):
    tc, _db, _storage, _ph = client
    resp = tc.post("/generate-report",
                   files={"file": ("sales.csv", _csv_bytes(sales), "text/csv")},
                   headers=_headers(anon_id))
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    resp2 = tc.get(f"/report/{session_id}", headers=_headers(anon_id))
    assert resp2.status_code == 200
    body = resp2.json()
    assert body["summary"] == "S."
    assert len(body["charts"]) == 10


def test_rejects_non_csv_xlsx(client, anon_id):
    tc, *_ = client
    resp = tc.post("/generate-report",
                   files={"file": ("data.txt", b"hello", "text/plain")},
                   headers=_headers(anon_id))
    assert resp.status_code == 422


def test_rejects_oversize_file(client, anon_id):
    tc, *_ = client
    big = b"a,b\n" + b"1,2\n" * 5_000_000
    resp = tc.post("/generate-report",
                   files={"file": ("big.csv", big, "text/csv")},
                   headers=_headers(anon_id))
    assert resp.status_code == 422


def test_rejects_corrupt_csv(client, anon_id):
    tc, *_ = client
    resp = tc.post("/generate-report",
                   files={"file": ("bad.csv", b"\x00\x01\x02broken", "text/csv")},
                   headers=_headers(anon_id))
    assert resp.status_code == 422


def test_get_nonexistent_session(client, anon_id):
    tc, *_ = client
    resp = tc.get("/report/does-not-exist", headers=_headers(anon_id))
    assert resp.status_code == 404


def test_missing_anon_id_returns_400(client, sales):
    tc, *_ = client
    resp = tc.post("/generate-report",
                   files={"file": ("sales.csv", _csv_bytes(sales), "text/csv")})
    assert resp.status_code == 400
    assert "MISSING_ANON_ID" in resp.text
```

- [ ] **Step 2: Write the anon limit tests**

Write `tests/integration/test_anon_limit.py`:

```python
import io
import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from uuid import uuid4

from tests.helpers.fake_claude import FakeClaude, tool_use
from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_storage import FakeStorage
from tests.helpers.fake_posthog import FakePostHog


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _ten_chart_fake():
    calls = [tool_use("frequency_bar_chart",
                      {"column": "region", "title": f"T{i}", "intent": f"i{i}"},
                      id_=f"tu_{i}")
             for i in range(10)]
    return FakeClaude([
        {"tool_calls": calls},
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "S.", "captions": [f"c{i}" for i in range(10)], "data_quality": []})]},
    ])


@pytest.fixture
def client_and_fakes(sales):
    fake_db = FakeDB()
    fake_storage = FakeStorage()
    fake_posthog = FakePostHog()

    def fake_claude_factory():
        return MagicMock(messages_create=_ten_chart_fake())

    from main import app, get_claude_client, get_db, get_storage, get_posthog
    app.dependency_overrides[get_claude_client] = fake_claude_factory
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[get_storage] = lambda: fake_storage
    app.dependency_overrides[get_posthog] = lambda: fake_posthog

    yield TestClient(app), fake_db, fake_storage, fake_posthog
    app.dependency_overrides.clear()


def _post_report(tc, sales, anon_id):
    return tc.post("/generate-report",
                   files={"file": ("sales.csv", _csv_bytes(sales), "text/csv")},
                   headers={"X-Anon-Id": anon_id})


def test_first_report_succeeds(client_and_fakes, sales):
    tc, db, _, _ = client_and_fakes
    anon = str(uuid4())
    resp = _post_report(tc, sales, anon)
    assert resp.status_code == 200
    assert db.count_anon_reports(anon) == 1


def test_second_report_blocked_with_anon_limit_code(client_and_fakes, sales):
    tc, db, _, ph = client_and_fakes
    anon = str(uuid4())
    _post_report(tc, sales, anon)
    resp = _post_report(tc, sales, anon)
    assert resp.status_code == 403
    body = resp.json()
    assert body["detail"]["code"] == "ANON_LIMIT_REACHED"
    # PostHog: server fired anon_limit_blocked
    assert len(ph.find("anon_limit_blocked")) == 1


def test_different_anon_not_blocked(client_and_fakes, sales):
    tc, *_ = client_and_fakes
    anon_a = str(uuid4())
    anon_b = str(uuid4())
    _post_report(tc, sales, anon_a)
    resp = _post_report(tc, sales, anon_b)
    assert resp.status_code == 200
```

- [ ] **Step 3: Rewrite `/generate-report` (and supporting helpers) in main.py**

Open `src/api/main.py`. Replace the entire file body — keeping the existing logging setup at the top — with the new version that uses `get_db`, `get_storage`, `get_posthog` instead of `get_redis`. The full new file:

```python
"""FastAPI app — /generate-report, /report/{id}/*, /export.pdf."""
import io
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any
from uuid import UUID

import pandas as pd
from anthropic import APIStatusError
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from claude_client import ClaudeClient, RetryableBusy
from db import SupabaseDB
from deps import get_anon_id
from llm_config import MODEL_NARRATIVE, MODEL_SELECTION, estimate_cost_usd
from posthog_server import PostHogServer
from profile import profile_dataframe
from report_generator import ReportGenerator
from schemas import ChartLayoutEntry, Report
from storage import StorageError, SupabaseStorage


load_dotenv()


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = (".csv", ".xlsx")
ANON_REPORT_LIMIT = 1


# ---- Logging ---------------------------------------------------------------

def setup_run_logging() -> tuple[str, str]:
    logs_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(logs_dir, f"chartsage_run_{ts}_{run_id}.log")
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [RUN:{run_id}] %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_path, mode="w", encoding="utf-8")],
        force=True,
    )
    _rotate_logs(logs_dir, keep=50)
    return run_id, log_path


def _rotate_logs(logs_dir: str, keep: int):
    files = sorted(
        (f for f in os.listdir(logs_dir) if f.startswith("chartsage_run_") and f.endswith(".log")),
        reverse=True,
    )
    for old in files[keep:]:
        try:
            os.remove(os.path.join(logs_dir, old))
        except OSError:
            pass


# ---- Dependencies ----------------------------------------------------------

_claude_singleton: ClaudeClient | None = None
_db_singleton: SupabaseDB | None = None
_storage_singleton: SupabaseStorage | None = None
_posthog_singleton: PostHogServer | None = None


def get_claude_client() -> ClaudeClient:
    global _claude_singleton
    if _claude_singleton is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _claude_singleton = ClaudeClient(api_key=api_key)
    return _claude_singleton


def get_db() -> SupabaseDB:
    global _db_singleton
    if _db_singleton is None:
        _db_singleton = SupabaseDB()
    return _db_singleton


def get_storage() -> SupabaseStorage:
    global _storage_singleton
    if _storage_singleton is None:
        _storage_singleton = SupabaseStorage()
    return _storage_singleton


def get_posthog() -> PostHogServer:
    global _posthog_singleton
    if _posthog_singleton is None:
        _posthog_singleton = PostHogServer()
    return _posthog_singleton


# ---- App -------------------------------------------------------------------

app = FastAPI(title="ChartSage v2")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://chartsage(-[a-z0-9-]+)?\.vercel\.app|http://localhost:3000|http://localhost:3001",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- File loading ----------------------------------------------------------

def _load_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding="utf-8")
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(content), encoding="latin1")
    if filename.lower().endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(content))
    raise ValueError(f"Unsupported extension: {filename}")


def _title_from_summary(summary: str) -> str:
    first_sentence = summary.split(".")[0].strip() if summary else ""
    return first_sentence[:200] if first_sentence else "Untitled report"


# ---- Endpoints -------------------------------------------------------------

@app.post("/generate-report")
async def generate_report(
    file: UploadFile = File(...),
    anon_id: UUID = Depends(get_anon_id),
    claude: ClaudeClient = Depends(get_claude_client),
    db: SupabaseDB = Depends(get_db),
    storage: SupabaseStorage = Depends(get_storage),
    posthog: PostHogServer = Depends(get_posthog),
):
    run_id, _ = setup_run_logging()
    started = time.perf_counter()

    # Anon limit check (early — before any expensive work)
    existing_count = db.count_anon_reports(anon_id)
    if existing_count >= ANON_REPORT_LIMIT:
        posthog.capture(str(anon_id), "anon_limit_blocked", {})
        raise HTTPException(
            status_code=403,
            detail={
                "code": "ANON_LIMIT_REACHED",
                "message": "You've used your free report. Sign in to do more.",
            },
        )

    # Validation
    if not file.filename or not any(file.filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=422, detail="Only .csv and .xlsx files are supported.")

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=422, detail=f"File exceeds {MAX_UPLOAD_BYTES} bytes.")
    if len(content) == 0:
        raise HTTPException(status_code=422, detail="File is empty.")

    try:
        df = _load_dataframe(file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    df.columns = [str(c).lower() for c in df.columns]

    if df.shape[1] < 2:
        raise HTTPException(status_code=422, detail="File must have at least 2 columns to chart.")
    if df.shape[0] < 1:
        raise HTTPException(status_code=422, detail="File has no data rows.")

    posthog.capture(str(anon_id), "report_generation_started", {
        "rowCount": int(df.shape[0]),
        "columnCount": int(df.shape[1]),
        "filename": file.filename,
        "sizeBytes": len(content),
    })

    try:
        profile = profile_dataframe(df)
        gen = ReportGenerator(
            profile=profile, df=df, claude=claude,
            model_selection=MODEL_SELECTION, model_narrative=MODEL_NARRATIVE,
        )
        report = gen.build_report()
    except RetryableBusy:
        posthog.capture(str(anon_id), "claude_overloaded", {"stage": "selection"})
        raise HTTPException(status_code=503, detail={
            "code": "BUSY",
            "message": "Claude is busy. Please retry in 30 seconds.",
        })
    except APIStatusError as e:
        logging.exception("Claude API error")
        posthog.capture(str(anon_id), "report_generation_failed", {
            "reason": "claude_api_status_error",
            "errorClass": type(e).__name__,
            "httpStatus": getattr(getattr(e, "response", None), "status_code", 0),
            "elapsedMs": int((time.perf_counter() - started) * 1000),
        })
        raise HTTPException(status_code=502, detail=f"Upstream model error: {e}")
    except Exception as e:
        logging.exception("Report generation failed")
        posthog.capture(str(anon_id), "report_generation_failed", {
            "reason": "internal",
            "errorClass": type(e).__name__,
            "httpStatus": 500,
            "elapsedMs": int((time.perf_counter() - started) * 1000),
        })
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    report_id = uuid.uuid4().hex
    csv_csv = df.to_csv(index=False).encode("utf-8")

    # Persist: storage upload first, then DB row. If storage fails, fail the whole request.
    try:
        csv_key = storage.upload_csv(report_id, csv_csv)
    except StorageError as e:
        logging.exception("Storage upload failed")
        raise HTTPException(status_code=502, detail={
            "code": "STORAGE_UNAVAILABLE",
            "message": f"Could not store source data: {e}",
        })

    try:
        db.save_report(
            report_id=report_id,
            anon_id=anon_id,
            user_id=None,
            report_json=report.model_dump(),
            csv_storage_key=csv_key,
            title=_title_from_summary(report.summary),
        )
    except Exception as e:
        # Best-effort cleanup of orphaned blob
        try:
            storage.delete_csv(report_id)
        except Exception:
            pass
        logging.exception("DB write failed")
        raise HTTPException(status_code=502, detail={
            "code": "STORAGE_UNAVAILABLE",
            "message": f"Could not save report: {e}",
        })

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    in_tok = report.metadata.get("input_tokens_total", 0) if isinstance(report.metadata, dict) else 0
    out_tok = report.metadata.get("output_tokens_total", 0) if isinstance(report.metadata, dict) else 0
    cache_tok = report.metadata.get("cache_read_input_tokens_total", 0) if isinstance(report.metadata, dict) else 0

    posthog.capture(str(anon_id), "report_generation_succeeded", {
        "reportId": report_id,
        "rowCount": int(df.shape[0]),
        "columnCount": int(df.shape[1]),
        "chartCount": len(report.charts),
        "modelSelection": MODEL_SELECTION,
        "modelNarrative": MODEL_NARRATIVE,
        "inputTokens": int(in_tok),
        "outputTokens": int(out_tok),
        "cacheReadTokens": int(cache_tok),
        "estCostUsd": estimate_cost_usd(MODEL_SELECTION, int(in_tok), int(out_tok), int(cache_tok)),
        "elapsedMs": elapsed_ms,
    })

    logging.info(
        "=== RUN SUMMARY ===\nrun_id: %s\nreport: %s\nrows: %d cols: %d  charts: %d  elapsed: %dms",
        run_id, report_id, df.shape[0], df.shape[1], len(report.charts), elapsed_ms,
    )

    return {"session_id": report_id}


@app.get("/report/{session_id}")
async def get_report(
    session_id: str,
    db: SupabaseDB = Depends(get_db),
):
    row = db.get_report(session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Report not found or expired.")
    return JSONResponse(content=row["report_json"])


@app.get("/health")
async def health():
    return {"status": "ok"}
```

(Subsequent tasks add PATCH layout, generate-more, export.pdf back in — the existing functionality is restored over the next few tasks.)

- [ ] **Step 4: Run test_api_errors.py + test_anon_limit.py**

```bash
PYTHONPATH=src/api pytest tests/integration/test_api_errors.py tests/integration/test_anon_limit.py -v
```

Expected: 6 + 3 = 9 passed. Some other integration tests (test_api_layout.py) will be broken — fix them in Task 8/9/10.

- [ ] **Step 5: Commit (broken layout tests will be fixed in next tasks)**

```bash
git add src/api/main.py tests/integration/test_api_errors.py tests/integration/test_anon_limit.py
git commit -m "feat: /generate-report uses Supabase + enforces 1-anon-report limit"
```

---

### Task 8: Restore PATCH /report/{id}/layout

**Files:**
- Modify: `src/api/main.py`
- Modify: `tests/integration/test_api_layout.py`

- [ ] **Step 1: Update test_api_layout.py to use FakeDB + FakeStorage + X-Anon-Id**

Open `tests/integration/test_api_layout.py`. Replace the `client_with_report` fixture and all tests with:

```python
import io
import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from uuid import uuid4

from tests.helpers.fake_claude import FakeClaude, tool_use
from tests.helpers.fake_db import FakeDB
from tests.helpers.fake_storage import FakeStorage
from tests.helpers.fake_posthog import FakePostHog


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


@pytest.fixture
def client_with_report(sales):
    chart_calls = [
        tool_use("frequency_bar_chart",
                 {"column": "region", "title": f"Chart {i}", "intent": f"intent {i}"},
                 id_=f"tu_{i}")
        for i in range(10)
    ]
    fake_claude = FakeClaude([
        {"tool_calls": chart_calls},
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "Sales.", "captions": [f"cap{i}" for i in range(10)], "data_quality": []})]},
    ])
    fake_db = FakeDB()
    fake_storage = FakeStorage()
    fake_posthog = FakePostHog()

    from main import app, get_claude_client, get_db, get_storage, get_posthog
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=fake_claude)
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[get_storage] = lambda: fake_storage
    app.dependency_overrides[get_posthog] = lambda: fake_posthog

    tc = TestClient(app)
    anon = str(uuid4())

    resp = tc.post("/generate-report",
                   files={"file": ("sales.csv", _csv_bytes(sales), "text/csv")},
                   headers={"X-Anon-Id": anon})
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    report = fake_db.get_report(session_id)["report_json"]

    yield tc, session_id, report, anon, fake_db, fake_storage, fake_posthog
    app.dependency_overrides.clear()


def test_patch_layout_valid_returns_204(client_with_report):
    tc, session_id, report, anon, *_ = client_with_report
    layout = report["layout"]
    layout[0], layout[1] = layout[1], layout[0]
    layout[0]["order"], layout[1]["order"] = 0, 1
    resp = tc.patch(f"/report/{session_id}/layout",
                    json=layout, headers={"X-Anon-Id": anon})
    assert resp.status_code == 204


def test_patch_layout_unknown_chart_id_returns_400(client_with_report):
    tc, session_id, _, anon, *_ = client_with_report
    bad = [{"chart_id": "does-not-exist", "position": "main", "order": 0}]
    resp = tc.patch(f"/report/{session_id}/layout",
                    json=bad, headers={"X-Anon-Id": anon})
    assert resp.status_code == 400
    assert "does-not-exist" in resp.text or "chart_id" in resp.text.lower()


def test_patch_layout_unknown_session_returns_404(client_with_report):
    tc, _, _, anon, *_ = client_with_report
    resp = tc.patch("/report/no-such-session/layout",
                    json=[], headers={"X-Anon-Id": anon})
    assert resp.status_code == 404


def test_patch_layout_persists(client_with_report):
    tc, session_id, report, anon, db, *_ = client_with_report
    layout = report["layout"]
    chart = layout[5]
    chart["position"] = "main"
    chart["order"] = 5
    resp = tc.patch(f"/report/{session_id}/layout",
                    json=layout, headers={"X-Anon-Id": anon})
    assert resp.status_code == 204

    persisted = db.get_report(session_id)["report_json"]["layout"]
    moved = next(e for e in persisted if e["chart_id"] == chart["chart_id"])
    assert moved["position"] == "main"
```

- [ ] **Step 2: Run, expect 404 / Method Not Allowed**

```bash
PYTHONPATH=src/api pytest tests/integration/test_api_layout.py -v
```

Expected: 4 fail (endpoint removed when main.py was rewritten in Task 7).

- [ ] **Step 3: Add the PATCH endpoint back to main.py**

Append to `src/api/main.py` (after `GET /report/{session_id}`):

```python
@app.patch("/report/{session_id}/layout", status_code=204)
async def patch_report_layout(
    session_id: str,
    new_layout: list[ChartLayoutEntry],
    db: SupabaseDB = Depends(get_db),
):
    row = db.get_report(session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Report not found or expired.")

    known_ids = {c["chart_id"] for c in row["report_json"].get("charts", [])}
    submitted_ids = {entry.chart_id for entry in new_layout}
    unknown = submitted_ids - known_ids
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown chart_id(s) in layout: {sorted(unknown)}",
        )

    db.update_layout(session_id, [entry.model_dump() for entry in new_layout])
    return None
```

- [ ] **Step 4: Run tests, expect pass**

```bash
PYTHONPATH=src/api pytest tests/integration/test_api_layout.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/main.py tests/integration/test_api_layout.py
git commit -m "feat: restore PATCH /report/{id}/layout on Supabase"
```

---

### Task 9: Restore POST /report/{id}/generate-more

**Files:**
- Modify: `src/api/main.py`
- Modify: `tests/integration/test_api_layout.py`

- [ ] **Step 1: Append generate-more tests to test_api_layout.py**

Append to `tests/integration/test_api_layout.py`:

```python
def test_generate_more_appends_charts(client_with_report, sales):
    tc, session_id, report, anon, db, _, ph = client_with_report

    new_calls = [
        tool_use("histogram_chart",
                 {"column": "revenue", "title": f"More {i}", "intent": f"new {i}"},
                 id_=f"more_{i}")
        for i in range(5)
    ]
    new_fake = FakeClaude([
        {"tool_calls": new_calls},
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "Updated.", "captions": [f"nc{i}" for i in range(5)], "data_quality": []})]},
    ])

    from main import app, get_claude_client
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=new_fake)

    initial_count = len(report["charts"])
    initial_sidebar = sum(1 for e in report["layout"] if e["position"] == "sidebar")

    resp = tc.post(f"/report/{session_id}/generate-more",
                   headers={"X-Anon-Id": anon})
    assert resp.status_code == 200
    updated = resp.json()
    assert len(updated["charts"]) == initial_count + 5

    new_sidebar = sum(1 for e in updated["layout"] if e["position"] == "sidebar")
    assert new_sidebar == initial_sidebar + 5
    # PostHog: success event fired
    assert len(ph.find("generate_more_succeeded")) == 1


def test_generate_more_unknown_session(client_with_report):
    tc, _, _, anon, *_ = client_with_report
    resp = tc.post("/report/nope/generate-more", headers={"X-Anon-Id": anon})
    assert resp.status_code == 404
```

- [ ] **Step 2: Run, expect failure (endpoint missing)**

```bash
PYTHONPATH=src/api pytest tests/integration/test_api_layout.py::test_generate_more_appends_charts -v
```

Expected: 404 or 405.

- [ ] **Step 3: Add the generate-more endpoint to main.py**

Append to `src/api/main.py`:

```python
@app.post("/report/{session_id}/generate-more")
async def generate_more(
    session_id: str,
    anon_id: UUID = Depends(get_anon_id),
    claude: ClaudeClient = Depends(get_claude_client),
    db: SupabaseDB = Depends(get_db),
    storage: SupabaseStorage = Depends(get_storage),
    posthog: PostHogServer = Depends(get_posthog),
):
    started = time.perf_counter()
    row = db.get_report(session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Report not found.")

    existing_report = Report.model_validate(row["report_json"])
    csv_key = row.get("csv_storage_key")
    if not csv_key:
        raise HTTPException(status_code=404, detail={
            "code": "SOURCE_DATA_UNAVAILABLE",
            "message": "Source data for this report is no longer available.",
        })

    try:
        csv_bytes = storage.download_csv(session_id)
    except StorageError:
        raise HTTPException(status_code=404, detail={
            "code": "SOURCE_DATA_UNAVAILABLE",
            "message": "Source data for this report is no longer available.",
        })

    df = pd.read_csv(io.StringIO(csv_bytes.decode("utf-8")))
    df.columns = [str(c).lower() for c in df.columns]

    posthog.capture(str(anon_id), "generate_more_started", {
        "reportId": session_id,
        "existingChartCount": len(existing_report.charts),
    })

    profile = profile_dataframe(df)
    gen = ReportGenerator(
        profile=profile, df=df, claude=claude,
        model_selection=MODEL_SELECTION, model_narrative=MODEL_NARRATIVE,
    )

    try:
        new_charts, new_layout = gen.generate_more(existing_report.charts)
    except RetryableBusy:
        posthog.capture(str(anon_id), "generate_more_failed", {
            "reportId": session_id, "reason": "busy", "httpStatus": 503,
            "elapsedMs": int((time.perf_counter() - started) * 1000),
        })
        raise HTTPException(status_code=503, detail={
            "code": "BUSY",
            "message": "Claude is busy. Try again in 30s.",
        })
    except Exception as e:
        posthog.capture(str(anon_id), "generate_more_failed", {
            "reportId": session_id, "reason": "internal",
            "errorClass": type(e).__name__,
            "elapsedMs": int((time.perf_counter() - started) * 1000),
        })
        raise

    if not new_charts:
        return JSONResponse(content=row["report_json"], status_code=200)

    report_dict = row["report_json"]
    sidebar_max = max(
        (e["order"] for e in report_dict["layout"] if e["position"] == "sidebar"),
        default=-1,
    )
    for i, (chart, layout_entry) in enumerate(zip(new_charts, new_layout)):
        layout_entry.order = sidebar_max + 1 + i
        report_dict["charts"].append(chart.model_dump())
        report_dict["layout"].append(layout_entry.model_dump())

    db.update_report_json(session_id, report_dict)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    in_tok = sum(getattr(c.spec, "data_point_count", 0) and 0 for c in new_charts)  # placeholder
    posthog.capture(str(anon_id), "generate_more_succeeded", {
        "reportId": session_id,
        "newChartCount": len(new_charts),
        "inputTokens": 0,    # tokens from sub-Claude call aren't surfaced through gen yet
        "outputTokens": 0,
        "estCostUsd": 0.005, # rough — 1 selection pass on Haiku
        "elapsedMs": elapsed_ms,
    })

    return JSONResponse(content=report_dict, status_code=200)
```

- [ ] **Step 4: Run tests, expect pass**

```bash
PYTHONPATH=src/api pytest tests/integration/test_api_layout.py -v
```

Expected: 6 passed (4 layout + 2 generate-more).

- [ ] **Step 5: Commit**

```bash
git add src/api/main.py tests/integration/test_api_layout.py
git commit -m "feat: restore POST /report/{id}/generate-more on Supabase + PostHog"
```

---

### Task 10: Restore GET /report/{id}/export.pdf

**Files:**
- Modify: `src/api/main.py`

- [ ] **Step 1: Append the export endpoint to main.py**

Append to `src/api/main.py`:

```python
@app.get("/report/{session_id}/export.pdf")
async def export_pdf(
    session_id: str,
    anon_id: UUID = Depends(get_anon_id),
    db: SupabaseDB = Depends(get_db),
    posthog: PostHogServer = Depends(get_posthog),
):
    started = time.perf_counter()
    if not db.get_report(session_id):
        raise HTTPException(status_code=404, detail="Report not found.")

    from pdf_export import render_report_pdf, _browser as pdf_browser_singleton  # noqa
    cold_start = pdf_browser_singleton is None
    posthog.capture(str(anon_id), "pdf_export_started", {
        "reportId": session_id, "coldStart": cold_start,
    })

    try:
        pdf_bytes = await render_report_pdf(session_id)
    except Exception as e:
        logging.exception("PDF export failed")
        posthog.capture(str(anon_id), "pdf_export_failed", {
            "reportId": session_id,
            "reason": "internal",
            "errorClass": type(e).__name__,
        })
        raise HTTPException(status_code=500, detail=f"PDF export failed: {e}")

    posthog.capture(str(anon_id), "pdf_export_succeeded", {
        "reportId": session_id,
        "byteSize": len(pdf_bytes),
        "elapsedMs": int((time.perf_counter() - started) * 1000),
    })

    short = session_id[:8]
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="chartsage-{short}.pdf"'},
    )


@app.on_event("shutdown")
async def _shutdown_event():
    try:
        from pdf_export import shutdown as pdf_shutdown
        await pdf_shutdown()
    except Exception:
        pass
```

- [ ] **Step 2: Run full test suite, expect green**

```bash
make test
```

Expected: ~135 passed, 1 skipped (PDF opt-in).

- [ ] **Step 3: Commit**

```bash
git add src/api/main.py
git commit -m "feat: restore GET /report/{id}/export.pdf with PostHog events"
```

---

### Task 11: Storage failure rollback test

**Files:**
- Create: `tests/integration/test_storage_failure.py`

- [ ] **Step 1: Write the failure test**

Write `tests/integration/test_storage_failure.py`:

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


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def test_storage_upload_failure_returns_502_and_leaves_no_row(sales):
    chart_calls = [tool_use("frequency_bar_chart",
                            {"column": "region", "title": f"T{i}", "intent": f"i{i}"},
                            id_=f"tu_{i}")
                   for i in range(10)]
    fake_claude = FakeClaude([
        {"tool_calls": chart_calls},
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "S.", "captions": [f"c{i}" for i in range(10)], "data_quality": []})]},
    ])
    fake_db = FakeDB()
    fake_storage = FakeStorage()
    fake_storage.fail_next_upload()
    fake_posthog = FakePostHog()

    from main import app, get_claude_client, get_db, get_storage, get_posthog
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=fake_claude)
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[get_storage] = lambda: fake_storage
    app.dependency_overrides[get_posthog] = lambda: fake_posthog

    tc = TestClient(app)
    anon = str(uuid4())
    resp = tc.post("/generate-report",
                   files={"file": ("sales.csv", _csv_bytes(sales), "text/csv")},
                   headers={"X-Anon-Id": anon})
    assert resp.status_code == 502
    assert "STORAGE_UNAVAILABLE" in resp.text
    assert fake_db.count_anon_reports(anon) == 0     # no orphan row

    app.dependency_overrides.clear()
```

- [ ] **Step 2: Run, expect pass**

```bash
PYTHONPATH=src/api pytest tests/integration/test_storage_failure.py -v
```

Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_storage_failure.py
git commit -m "test: storage upload failure leaves no orphaned DB row"
```

---

### Task 12: PostHog events test

**Files:**
- Create: `tests/integration/test_posthog_events.py`

- [ ] **Step 1: Write the event-shape test**

Write `tests/integration/test_posthog_events.py`:

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


def _csv(df):
    b = io.StringIO()
    df.to_csv(b, index=False)
    return b.getvalue().encode("utf-8")


@pytest.fixture
def fakes_and_client(sales):
    calls = [tool_use("frequency_bar_chart",
                      {"column": "region", "title": f"T{i}", "intent": f"i{i}"},
                      id_=f"tu_{i}")
             for i in range(10)]
    fake_claude = FakeClaude([
        {"tool_calls": calls},
        {"tool_calls": [tool_use("submit_narrative",
                                 {"summary": "S.", "captions": [f"c{i}" for i in range(10)], "data_quality": []})]},
    ])
    fake_db = FakeDB()
    fake_storage = FakeStorage()
    fake_posthog = FakePostHog()

    from main import app, get_claude_client, get_db, get_storage, get_posthog
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=fake_claude)
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[get_storage] = lambda: fake_storage
    app.dependency_overrides[get_posthog] = lambda: fake_posthog
    yield TestClient(app), fake_posthog
    app.dependency_overrides.clear()


def test_generate_report_fires_started_and_succeeded(fakes_and_client, sales):
    tc, ph = fakes_and_client
    anon = str(uuid4())
    resp = tc.post("/generate-report",
                   files={"file": ("sales.csv", _csv(sales), "text/csv")},
                   headers={"X-Anon-Id": anon})
    assert resp.status_code == 200

    started = ph.find("report_generation_started")
    succeeded = ph.find("report_generation_succeeded")
    assert len(started) == 1
    assert len(succeeded) == 1
    # Keys are camelCase
    s = succeeded[0]["properties"]
    for required in ("reportId", "rowCount", "columnCount", "chartCount",
                     "modelSelection", "estCostUsd", "elapsedMs"):
        assert required in s, f"missing key {required} in {list(s.keys())}"
    # No snake_case leakage
    for k in s:
        assert "_" not in k or k.startswith("$"), f"property {k} contains underscore"


def test_anon_limit_blocked_event_camelcase_only(fakes_and_client, sales):
    tc, ph = fakes_and_client
    anon = str(uuid4())
    tc.post("/generate-report",
            files={"file": ("sales.csv", _csv(sales), "text/csv")},
            headers={"X-Anon-Id": anon})
    tc.post("/generate-report",
            files={"file": ("sales.csv", _csv(sales), "text/csv")},
            headers={"X-Anon-Id": anon})
    blocked = ph.find("anon_limit_blocked")
    assert len(blocked) == 1
    for k in blocked[0]["properties"]:
        assert "_" not in k or k.startswith("$")
```

- [ ] **Step 2: Run, expect pass**

```bash
PYTHONPATH=src/api pytest tests/integration/test_posthog_events.py -v
```

Expected: 2 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_posthog_events.py
git commit -m "test: PostHog event shape (snake_case events, camelCase properties)"
```

---

## Phase 4 — Frontend foundation

### Task 13: Next.js middleware sets anon cookie

**Files:**
- Create: `src/middleware.ts`

- [ ] **Step 1: Write the middleware**

Write `src/middleware.ts`:

```typescript
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

const COOKIE_NAME = 'chartsage_anon';
const ONE_YEAR_SECONDS = 60 * 60 * 24 * 365;

export function middleware(req: NextRequest) {
  const res = NextResponse.next();
  const existing = req.cookies.get(COOKIE_NAME);
  if (!existing) {
    const uuid = crypto.randomUUID();
    res.cookies.set({
      name: COOKIE_NAME,
      value: uuid,
      httpOnly: true,
      sameSite: 'lax',
      secure: process.env.NODE_ENV === 'production',
      maxAge: ONE_YEAR_SECONDS,
      path: '/',
    });
  }
  return res;
}

export const config = {
  // Skip static files and Next.js internals
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};
```

- [ ] **Step 2: Build check**

```bash
npm run build
```

Expected: build succeeds; middleware listed in route summary.

- [ ] **Step 3: Commit**

```bash
git add src/middleware.ts
git commit -m "feat: Next.js middleware sets long-lived chartsage_anon cookie"
```

---

### Task 14: lib/anon.ts + lib/api.ts + lib/posthog.ts

**Files:**
- Create: `src/app/lib/anon.ts`
- Create: `src/app/lib/api.ts`
- Create: `src/app/lib/posthog.ts`

- [ ] **Step 1: Write lib/anon.ts**

Write `src/app/lib/anon.ts`:

```typescript
'use client';

const COOKIE_NAME = 'chartsage_anon';

export function getAnonId(): string | null {
  if (typeof document === 'undefined') return null;
  const match = document.cookie.match(/(?:^|; )chartsage_anon=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : null;
}
```

Hmm — the cookie is httpOnly, so `document.cookie` will NOT see it. That's a problem for the frontend reading it. We need a different approach.

Replace with:

```typescript
'use client';

/**
 * The chartsage_anon cookie is httpOnly so JS can't read it directly.
 * The Next.js middleware also writes a parallel chartsage_anon_pub cookie
 * (NOT httpOnly) so the browser SDK can identify users in PostHog.
 *
 * Both contain the same UUID. The httpOnly one is sent automatically with
 * fetch credentials; we mirror it client-side for header injection too.
 */
const COOKIE_NAME = 'chartsage_anon_pub';

export function getAnonId(): string | null {
  if (typeof document === 'undefined') return null;
  const match = document.cookie.match(/(?:^|; )chartsage_anon_pub=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : null;
}
```

Now update `src/middleware.ts` to also write the public cookie. Open `src/middleware.ts` and modify the `if (!existing)` block:

```typescript
  if (!existing) {
    const uuid = crypto.randomUUID();
    res.cookies.set({
      name: COOKIE_NAME,
      value: uuid,
      httpOnly: true,
      sameSite: 'lax',
      secure: process.env.NODE_ENV === 'production',
      maxAge: ONE_YEAR_SECONDS,
      path: '/',
    });
    // Mirror to a non-httpOnly cookie so JS can read it for header injection + PostHog identify
    res.cookies.set({
      name: 'chartsage_anon_pub',
      value: uuid,
      httpOnly: false,
      sameSite: 'lax',
      secure: process.env.NODE_ENV === 'production',
      maxAge: ONE_YEAR_SECONDS,
      path: '/',
    });
  }
```

- [ ] **Step 2: Write lib/api.ts**

Write `src/app/lib/api.ts`:

```typescript
'use client';
import { getAnonId } from './anon';

export interface ApiError extends Error {
  status: number;
  code?: string;
  detail?: unknown;
}

export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const anonId = getAnonId();
  const headers = new Headers(init.headers || {});
  if (anonId) headers.set('X-Anon-Id', anonId);

  const url = `${process.env.NEXT_PUBLIC_API_URL}${path}`;
  return fetch(url, { ...init, headers });
}

export async function apiJSON<T = any>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await apiFetch(path, init);
  if (!res.ok) {
    let body: any = null;
    try { body = await res.json(); } catch {}
    const err = new Error(body?.detail?.message || `Request failed (${res.status})`) as ApiError;
    err.status = res.status;
    err.code = body?.detail?.code;
    err.detail = body?.detail;
    throw err;
  }
  return res.json();
}
```

- [ ] **Step 3: Write lib/posthog.ts**

Write `src/app/lib/posthog.ts`:

```typescript
'use client';
import posthog from 'posthog-js';
import { getAnonId } from './anon';

let initialized = false;

export function initPostHog(): void {
  if (initialized || typeof window === 'undefined') return;
  const key = process.env.NEXT_PUBLIC_POSTHOG_KEY;
  const host = process.env.NEXT_PUBLIC_POSTHOG_HOST || 'https://us.i.posthog.com';
  if (!key) return;

  posthog.init(key, {
    api_host: host,
    capture_pageview: true,
    autocapture: false,
    person_profiles: 'identified_only',
  });

  const anonId = getAnonId();
  if (anonId) {
    posthog.identify(anonId);
  }
  initialized = true;
}

export { posthog };
```

- [ ] **Step 4: Install posthog-js**

```bash
npm install posthog-js@^1.130.0
npm run build
```

Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add src/middleware.ts src/app/lib/anon.ts src/app/lib/api.ts src/app/lib/posthog.ts package.json package-lock.json
git commit -m "feat: anon cookie mirror + apiFetch wrapper + posthog-js init"
```

---

### Task 15: Initialize PostHog in app/layout.tsx

**Files:**
- Modify: `src/app/layout.tsx`

- [ ] **Step 1: Add PostHog initializer to layout**

Open `src/app/layout.tsx`. After the existing imports, add a client component that calls initPostHog on mount. Since `layout.tsx` is server-side by default, we need a small client wrapper.

Create `src/app/PostHogInit.tsx`:

```typescript
'use client';
import { useEffect } from 'react';
import { initPostHog } from './lib/posthog';

export default function PostHogInit() {
  useEffect(() => { initPostHog(); }, []);
  return null;
}
```

Modify `src/app/layout.tsx` — import and render the initializer once:

```typescript
import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import PostHogInit from './PostHogInit';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'ChartSage',
  description: 'Drop a CSV. Get a narrated data report with charts in seconds.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <PostHogInit />
        <main className="min-h-screen bg-stone-50">{children}</main>
      </body>
    </html>
  );
}
```

- [ ] **Step 2: Build check**

```bash
npm run build
```

Expected: success.

- [ ] **Step 3: Commit**

```bash
git add src/app/PostHogInit.tsx src/app/layout.tsx
git commit -m "feat: initialize PostHog SDK in app layout"
```

---

## Phase 5 — Frontend integration

### Task 16: Upload page uses apiFetch + redirect on 403

**Files:**
- Modify: `src/app/page.tsx`
- Create: `src/app/anon-limit/page.tsx`

- [ ] **Step 1: Create /anon-limit placeholder**

Write `src/app/anon-limit/page.tsx`:

```typescript
'use client';
import { useEffect } from 'react';
import { posthog } from '../lib/posthog';

export default function AnonLimitPage() {
  useEffect(() => {
    posthog.capture?.('anon_limit_page_viewed', { entryPoint: 'afterUpload' });
  }, []);

  return (
    <div className="min-h-screen bg-stone-50 flex items-center justify-center px-4">
      <div className="max-w-md text-center">
        <p className="text-xs uppercase tracking-widest text-stone-400 mb-2">Free tier</p>
        <h1 className="text-3xl font-semibold tracking-tight text-stone-900 mb-3">
          You've used your free report.
        </h1>
        <p className="text-stone-600 leading-relaxed mb-6">
          Sign in to do more — generate additional charts, save your reports, and
          come back to them later. Accounts are coming soon.
        </p>
        <button
          type="button"
          disabled
          className="px-5 py-2.5 bg-stone-300 text-stone-600 text-sm font-medium rounded-lg cursor-not-allowed"
          onClick={() => posthog.capture?.('signin_cta_clicked', { from: 'anonLimit' })}
        >
          Sign in · coming soon
        </button>
        <p className="mt-6 text-sm text-stone-400">
          <a href="/" className="hover:text-stone-700">← Back to home</a>
        </p>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Modify upload page to use apiFetch + handle 403**

Open `src/app/page.tsx`. In the `generate()` function, replace the fetch block with apiFetch and add the 403 → redirect logic. The full replacement of the `generate()` function:

```typescript
  async function generate() {
    if (!file) return;
    setIsProcessing(true);
    setError(null);
    setStep(0);
    try {
      setStep(1);
      const fd = new FormData();
      fd.append('file', file);
      const res = await apiFetch('/generate-report', { method: 'POST', body: fd });
      let body: any = null;
      try { body = await res.json(); } catch {}

      if (res.status === 403 && body?.detail?.code === 'ANON_LIMIT_REACHED') {
        router.push('/anon-limit');
        return;
      }
      if (res.status === 503 && body?.detail?.code === 'BUSY') {
        setError(body?.detail?.message ?? 'Service busy. Please retry in 30 seconds.');
        setIsProcessing(false);
        return;
      }
      if (!res.ok) {
        const detail = body?.detail;
        throw new Error(typeof detail === 'string' ? detail : detail?.message ?? 'Failed to generate report.');
      }
      setStep(2);
      router.push(`/report/${body.session_id}`);
    } catch (e: any) {
      setError(e.message || 'Generation failed.');
      setIsProcessing(false);
    }
  }
```

At the top of the file, add the import:

```typescript
import { apiFetch } from './lib/api';
```

- [ ] **Step 3: Build check**

```bash
npm run build
```

Expected: success.

- [ ] **Step 4: Commit**

```bash
git add src/app/page.tsx src/app/anon-limit/page.tsx
git commit -m "feat: upload page uses apiFetch; /anon-limit placeholder"
```

---

### Task 17: useReportLayout uses apiFetch

**Files:**
- Modify: `src/app/report/[id]/useReportLayout.ts`

- [ ] **Step 1: Replace the inline fetch with apiFetch**

Open `src/app/report/[id]/useReportLayout.ts`. At the top:

```typescript
import { apiFetch } from '../../lib/api';
```

Find the PATCH block:

```typescript
const res = await fetch(
  `${process.env.NEXT_PUBLIC_API_URL}/report/${sessionId}/layout`,
  {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(nextLayout),
  },
);
```

Replace with:

```typescript
const res = await apiFetch(`/report/${sessionId}/layout`, {
  method: 'PATCH',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(nextLayout),
});
```

- [ ] **Step 2: Build check**

```bash
npm run build
```

Expected: success.

- [ ] **Step 3: Commit**

```bash
git add src/app/report/'[id]'/useReportLayout.ts
git commit -m "feat: useReportLayout uses apiFetch (X-Anon-Id propagation)"
```

---

### Task 18: Toolbar uses apiFetch + fires PostHog events

**Files:**
- Modify: `src/app/report/[id]/Toolbar.tsx`

- [ ] **Step 1: Replace Toolbar.tsx**

Open `src/app/report/[id]/Toolbar.tsx`. Replace the entire file:

```typescript
'use client';
import { useState } from 'react';
import { apiFetch } from '../../lib/api';
import { posthog } from '../../lib/posthog';
import type { Report } from './useReportLayout';

interface Props {
  sessionId: string;
  onReportUpdated: (next: Report) => void;
}

export default function Toolbar({ sessionId, onReportUpdated }: Props) {
  const [generating, setGenerating] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleGenerateMore() {
    setGenerating(true);
    setError(null);
    posthog.capture?.('generate_more_clicked', { reportId: sessionId });
    try {
      const res = await apiFetch(`/report/${sessionId}/generate-more`, { method: 'POST' });
      if (res.status === 503) {
        setError('Claude is busy. Try again in 30 seconds.');
        return;
      }
      if (!res.ok) throw new Error(`Failed (${res.status})`);
      const updated: Report = await res.json();
      onReportUpdated(updated);
    } catch (e: any) {
      setError(e.message || 'Failed to generate more charts.');
    } finally {
      setGenerating(false);
    }
  }

  function handleExportPdf() {
    setExporting(true);
    setError(null);
    posthog.capture?.('export_pdf_clicked', { reportId: sessionId });
    const url = `${process.env.NEXT_PUBLIC_API_URL}/report/${sessionId}/export.pdf`;
    // X-Anon-Id can't be sent via window.open; the export endpoint requires it.
    // For now we redirect to the URL — the browser sends the chartsage_anon cookie
    // automatically, but our backend reads the header. So we issue an authenticated
    // fetch + create a blob URL.
    apiFetch(`/report/${sessionId}/export.pdf`)
      .then(async (r) => {
        if (!r.ok) throw new Error(`Export failed (${r.status})`);
        const blob = await r.blob();
        const blobUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = blobUrl;
        a.download = `chartsage-${sessionId.slice(0, 8)}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(blobUrl);
      })
      .catch((e) => setError(e.message || 'Export failed.'))
      .finally(() => setExporting(false));
  }

  return (
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
  );
}
```

- [ ] **Step 2: Build check**

```bash
npm run build
```

Expected: success.

- [ ] **Step 3: Commit**

```bash
git add src/app/report/'[id]'/Toolbar.tsx
git commit -m "feat: Toolbar uses apiFetch + fires PostHog events"
```

---

## Phase 6 — Deployment artifacts

### Task 19: Dockerfile + .dockerignore

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Write Dockerfile**

Write `Dockerfile` at repo root:

```dockerfile
FROM mcr.microsoft.com/playwright/python:v1.42.0-jammy
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/api ./src/api
ENV PYTHONPATH=/app/src/api
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --app-dir src/api"]
```

- [ ] **Step 2: Write .dockerignore**

Write `.dockerignore`:

```
.git
.gitignore
.env
.env.*
!.env.production.example
venv
node_modules
.next
__pycache__
*.pyc
tests
docs
README.md
*.md
src/app
src/api/logs/*.log
package*.json
postcss.config.js
tailwind.config.js
tsconfig.json
next-env.d.ts
```

- [ ] **Step 3: Local build smoke (optional but recommended)**

If Docker is installed locally:

```bash
docker build -t chartsage-backend:test .
docker run --rm -p 8080:8080 -e ANTHROPIC_API_KEY=test -e SUPABASE_URL=test -e SUPABASE_SERVICE_ROLE_KEY=test chartsage-backend:test &
sleep 5
curl -fsS http://127.0.0.1:8080/health
docker kill $(docker ps -q --filter ancestor=chartsage-backend:test)
```

Expected: `{"status":"ok"}`. If Docker isn't available locally, skip and verify in Cloud Build later.

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "chore: Dockerfile + .dockerignore for Cloud Run"
```

---

### Task 20: cloudbuild.yaml + vercel.json + .env.production.example

**Files:**
- Create: `cloudbuild.yaml`
- Create: `vercel.json`
- Create: `.env.production.example`

- [ ] **Step 1: Write cloudbuild.yaml**

Write `cloudbuild.yaml`:

```yaml
steps:
  - name: gcr.io/cloud-builders/docker
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/chartsage-backend:$SHORT_SHA', '.']
  - name: gcr.io/cloud-builders/docker
    args: ['push', 'gcr.io/$PROJECT_ID/chartsage-backend:$SHORT_SHA']
  - name: gcr.io/google.com/cloudsdktool/cloud-sdk
    entrypoint: gcloud
    args:
      - run
      - deploy
      - chartsage-backend
      - --image=gcr.io/$PROJECT_ID/chartsage-backend:$SHORT_SHA
      - --region=us-central1
      - --platform=managed
      - --allow-unauthenticated
      - --memory=1Gi
      - --cpu=1
      - --min-instances=0
      - --max-instances=10
      - --concurrency=4
      - --timeout=120s
      - --service-account=chartsage-runner@$PROJECT_ID.iam.gserviceaccount.com
      - --set-secrets=ANTHROPIC_API_KEY=anthropic-key:latest,SUPABASE_SERVICE_ROLE_KEY=supabase-srk:latest,POSTHOG_API_KEY=posthog-key:latest
      - --set-env-vars=SUPABASE_URL=$_SUPABASE_URL,FRONTEND_BASE_URL=$_FRONTEND_BASE_URL,CLAUDE_MODEL=haiku-4-5

substitutions:
  _SUPABASE_URL: 'https://YOUR_PROJECT.supabase.co'
  _FRONTEND_BASE_URL: 'https://chartsage.vercel.app'

images:
  - 'gcr.io/$PROJECT_ID/chartsage-backend:$SHORT_SHA'
```

- [ ] **Step 2: Write vercel.json**

Write `vercel.json`:

```json
{
  "framework": "nextjs",
  "buildCommand": "npm run build"
}
```

- [ ] **Step 3: Write .env.production.example**

Write `.env.production.example`:

```bash
# Required: Anthropic API key
ANTHROPIC_API_KEY=sk-ant-...

# Frontend → Backend
NEXT_PUBLIC_API_URL=https://chartsage-backend-xxxx-uc.a.run.app

# Supabase (backend uses service role; frontend uses anon)
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJh...                # Cloud Run only
NEXT_PUBLIC_SUPABASE_URL=https://xxxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJh...            # Vercel

# PostHog
POSTHOG_API_KEY=phc_...                          # Cloud Run
NEXT_PUBLIC_POSTHOG_KEY=phc_...                  # Vercel
NEXT_PUBLIC_POSTHOG_HOST=https://us.i.posthog.com

# Backend behavior
FRONTEND_BASE_URL=https://chartsage.vercel.app
CLAUDE_MODEL=haiku-4-5
# CLAUDE_MODEL_SELECTION=
# CLAUDE_MODEL_NARRATIVE=
```

- [ ] **Step 4: Commit**

```bash
git add cloudbuild.yaml vercel.json .env.production.example
git commit -m "chore: cloudbuild.yaml + vercel.json + .env.production.example"
```

---

### Task 21: Update README with Deploying section

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Append the Deploying section**

Open `README.md`. Append at the end:

````markdown

## Deploying

Production runs on Vercel (frontend) + Google Cloud Run (backend) + Supabase (Postgres + Storage + auth) + PostHog (analytics).

### One-time provisioning

1. **Supabase**
   - Create project at supabase.com (US-East recommended).
   - SQL editor → run the schema from [the SP1 design](docs/superpowers/specs/2026-05-24-sp1-foundation-design.md#data-model).
   - Storage → create a private bucket named `csv-inputs`.
   - Settings → copy `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`.

2. **PostHog**
   - Create a free account + project at posthog.com.
   - Copy the project API key (`phc_...`).

3. **Google Cloud**
   ```bash
   gcloud config set project YOUR_PROJECT_ID
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com
   gcloud iam service-accounts create chartsage-runner
   ```
   Push secrets:
   ```bash
   echo -n "$ANTHROPIC_API_KEY" | gcloud secrets create anthropic-key --data-file=-
   echo -n "$SUPABASE_SERVICE_ROLE_KEY" | gcloud secrets create supabase-srk --data-file=-
   echo -n "$POSTHOG_API_KEY" | gcloud secrets create posthog-key --data-file=-
   ```

4. **Vercel**
   - Import this repo at vercel.com.
   - Add env vars: `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`, `NEXT_PUBLIC_POSTHOG_KEY`, `NEXT_PUBLIC_POSTHOG_HOST`.

### Deploy

Backend:
```bash
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_SUPABASE_URL=https://YOUR.supabase.co,_FRONTEND_BASE_URL=https://chartsage.vercel.app
```

Frontend:
```bash
git push origin main   # Vercel auto-deploys
```

### Smoke test

After the first deploy, visit your Vercel URL. Drop a CSV. Verify the report renders, check the Supabase `reports` table has a row, and check the PostHog dashboard for `report_generation_succeeded` events.
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: deploying section for Vercel + Cloud Run + Supabase + PostHog"
```

---

## Phase 7 — Final verification + provisioning runbook

### Task 22: Final test sweep

**Files:** none (verification only)

- [ ] **Step 1: Run all tests**

```bash
make test
```

Expected: ~135 passed, 1 skipped (PDF opt-in).

- [ ] **Step 2: Run npm build**

```bash
npm run build
```

Expected: build succeeds with all routes including `/anon-limit` and `/report/[id]/print`.

- [ ] **Step 3: Local end-to-end smoke (against the production Supabase project)**

After provisioning Supabase, set local `.env`:

```bash
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...
POSTHOG_API_KEY=phc_...
ANTHROPIC_API_KEY=sk-ant-...
FRONTEND_BASE_URL=http://localhost:3000
```

Then:

```bash
make dev &       # backend on :8000
npm run dev &    # frontend on :3000
```

Open `http://localhost:3000`. Drop a CSV. Verify:
- Report generates and renders
- Supabase dashboard shows a row in `reports` table and a file in `csv-inputs` bucket
- PostHog dashboard receives `report_generation_succeeded` event
- Drop a second CSV in the same browser → redirects to `/anon-limit`
- Clear cookies → can upload again

- [ ] **Step 4: Commit any polish fixes** (only if you found bugs)

```bash
git status
# if anything outstanding, commit appropriately
```

---

### Task 23: Production provisioning runbook (USER-DRIVEN)

This task is not code — it's the sequence of commands the user runs once to actually deploy.

- [ ] **Step 1: Provision Supabase**
  1. Sign up at supabase.com, create a new project (US-East).
  2. SQL editor → paste and run the schema from `docs/superpowers/specs/2026-05-24-sp1-foundation-design.md` (Data Model section).
  3. Storage → create bucket `csv-inputs`, set to **private**.
  4. Settings → API → copy three values into your local `.env.local`: `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`.

- [ ] **Step 2: Provision PostHog**
  1. Sign up at posthog.com (free cloud).
  2. Create project. Copy the project key (`phc_...`).
  3. Project settings → Autocapture → OFF (we want explicit events only).

- [ ] **Step 3: Provision Google Cloud**
  ```bash
  gcloud config set project YOUR_PROJECT_ID
  gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com artifactregistry.googleapis.com
  gcloud iam service-accounts create chartsage-runner
  ```
  ```bash
  echo -n "YOUR_ANTHROPIC_KEY" | gcloud secrets create anthropic-key --data-file=-
  echo -n "YOUR_SUPABASE_SRK"   | gcloud secrets create supabase-srk --data-file=-
  echo -n "YOUR_POSTHOG_KEY"    | gcloud secrets create posthog-key --data-file=-
  ```
  ```bash
  PROJECT_ID=$(gcloud config get-value project)
  gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member=serviceAccount:chartsage-runner@$PROJECT_ID.iam.gserviceaccount.com \
    --role=roles/secretmanager.secretAccessor
  ```

- [ ] **Step 4: Deploy backend to Cloud Run**
  ```bash
  gcloud builds submit --config cloudbuild.yaml \
    --substitutions=_SUPABASE_URL=https://YOUR.supabase.co,_FRONTEND_BASE_URL=https://chartsage.vercel.app
  ```
  Copy the resulting Cloud Run URL (e.g., `https://chartsage-backend-xxxx-uc.a.run.app`).

- [ ] **Step 5: Provision Vercel**
  1. vercel.com → New Project → import this GitHub repo.
  2. Framework auto-detected as Next.js.
  3. Add Environment Variables (all environments):
     - `NEXT_PUBLIC_API_URL=<your Cloud Run URL>`
     - `NEXT_PUBLIC_SUPABASE_URL=https://...supabase.co`
     - `NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJh...`
     - `NEXT_PUBLIC_POSTHOG_KEY=phc_...`
     - `NEXT_PUBLIC_POSTHOG_HOST=https://us.i.posthog.com`
  4. Deploy. Vercel returns a URL.

- [ ] **Step 6: Production smoke test**
  - Visit the Vercel URL in a fresh browser.
  - DevTools → Application → Cookies → confirm `chartsage_anon` and `chartsage_anon_pub` are set.
  - Drop a CSV → report generates within ~15s (first call cold-starts Cloud Run).
  - Supabase dashboard → `reports` table has the row; Storage `csv-inputs` has the file.
  - PostHog dashboard → see `report_generation_succeeded` event with `estCostUsd`.
  - Drop a second CSV in same browser → redirects to `/anon-limit`.
  - PostHog → `anon_limit_blocked` + `anon_limit_page_viewed` events present.
  - Clear cookies, refresh → can upload again.
  - Click "Export PDF" → downloads a PDF; opens cleanly.

---

## Spec coverage check

| Spec requirement | Implemented in |
|---|---|
| Replace Redis with Supabase Postgres | Tasks 2, 3, 7-10 |
| Supabase Storage for CSV blobs | Tasks 4, 7, 9 |
| Anon UUID cookie (httpOnly + JS-readable mirror) | Tasks 13, 14 |
| 1-report-per-anon cap | Task 7 + test in Task 7 |
| PostHog server-side events with camelCase props | Tasks 5, 7, 9, 10, 12 |
| PostHog browser SDK initialization | Tasks 14, 15 |
| apiFetch wrapper with X-Anon-Id header | Tasks 14, 16, 17, 18 |
| /anon-limit placeholder page | Task 16 |
| Dockerfile (Playwright base) | Task 19 |
| cloudbuild.yaml | Task 20 |
| vercel.json | Task 20 |
| .env.production.example | Task 20 |
| Cost estimator (estimate_cost_usd) | Task 1 |
| Storage failure rollback | Task 11 |
| FakeDB / FakeStorage / FakePostHog | Tasks 2, 4, 5 |
| README Deploying section | Task 21 |
| Production smoke runbook | Task 23 |

No spec gaps.
