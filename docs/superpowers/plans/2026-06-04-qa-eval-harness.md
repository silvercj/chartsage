# ChartSage QA / Eval Harness Implementation Plan

> **For agentic workers: REQUIRED SUB-SKILL: superpowers:subagent-driven-development** — execute each task as an isolated subagent (write failing test → run → implement → run → commit). Subagents must NOT run any branch-mutating git command (`checkout`, `switch`, `reset`, `stash`, `worktree`, `rebase`, `merge`, `cherry-pick`). The controller owns all branch operations.

**Goal:** Catch data-handling regressions — broken/degenerate charts, type mishaps, weak/incoherent narrative, crashes — *before users hit them*, by running a corpus of edge-case + real datasets through the **real** ChartSage generation pipeline locally and validating every output deterministically **and** with an LLM judge. On-demand dev tooling; never deployed. Motivating bug: the year-as-int line chart that collapsed to a single point.

**Architecture:** A new top-level `qa/` package that imports the production modules unchanged (`profile.profile_dataframe`, `report_generator.ReportGenerator`, `chart_executor`, `llm_config`, `claude_client`, `schemas`) and exercises the exact endpoint sequence via one shared `qa/pipeline.run_report(df)` helper (the anti-drift guard). The runner feeds ~15 synthetic edge-case generators plus any drop-in `qa/datasets/*.csv` through the pipeline using the **real Anthropic Haiku** client, writes per-dataset JSON, then applies two validation layers — deterministic pure-function checks (`qa/validators.py`) and a Haiku LLM judge (`qa/judge.py`) — and emits a scannable `report.md` + machine-readable `summary.json`. Folded-in first: a test-suite-health fix (`pytest-timeout` + offline-safe mock of the one network-bound test) so runs can't hang.

**Tech Stack:** Python 3.11 (venv interpreter `venv/bin/python`), pandas 2.2, pydantic 2.13, anthropic 0.50 (real Haiku, key from `.env`), pytest 8 + pytest-timeout, argparse, dataclasses + stdlib `json`. No new heavy deps beyond `pytest-timeout`.

---

## Build constraints (read before any task)

- **Worktree, off `origin/main`.** The controller creates an isolated git worktree from `origin/main` for this branch. Subagents work only inside that worktree and **never** run `git checkout/switch/reset/stash/worktree/rebase/merge/cherry-pick`. Subagents may run `git add -A` and `git commit` (per-task) and `git status`/`git diff`/`git log`.
- **venv interpreter only.** Every Python/pytest invocation uses `venv/bin/python -m pytest ...` or `venv/bin/python qa/...`. Never bare `python`/`pytest`.
- **Real key from `.env`.** The runner needs `ANTHROPIC_API_KEY`. The app loads it via `from dotenv import load_dotenv; load_dotenv()` (see `src/api/main.py:16,38`). `qa/run_eval.py` and `qa/judge.py` do the same. The key is already in `.env` at repo root.
- **Local-tooling auto-background gotcha.** On this machine, long `python`/`git` shell commands tend to detach and run in the background. **Kick off a long command and WAIT for it to finish before touching the same worktree again** (don't fire a second git/pytest command into a still-running one). If a worktree's git state corrupts, clear stale locks: `rm -f .git/worktrees/<name>/*.lock`.
- **`pythonpath`.** Tests already resolve `src/api` via `pytest.ini` `pythonpath = src/api`. The `qa/` package imports the same way; `qa/run_eval.py` prepends `src/api` to `sys.path` at runtime (Task 9) so it runs from repo root without pytest.
- **Dev tooling — NO prod deploy.** No Cloud Run, no Vercel, no migrations. The only "ship" step is the controller merging the branch to `main` (user-authorized) in the final task.
- **Save this plan to:** `docs/superpowers/plans/2026-06-04-qa-eval-harness.md`.

---

## Anti-drift contract (why `qa/pipeline.run_report` must match `main.py`)

`qa/pipeline.run_report(df)` reproduces the `/generate-report` body in `src/api/main.py` **exactly**, so the harness can never validate a code path users don't hit. The production sequence (main.py lines 332–392 + report_generator `build_report`) is:

1. `df.columns = [str(c).lower() for c in df.columns]` (main.py:337). *(The runner lower-cases columns; generators emit already-lower names, but the helper applies this unconditionally to mirror prod.)*
2. `df, was_sampled, total_rows = sample_for_analysis(df)` (main.py:339; `MAX_ANALYSIS_ROWS=50000`, deterministic `random_state=0`).
3. Guards: `df.shape[1] >= 2`, `df.shape[0] >= 1` (main.py:343–346).
4. `profile = profile_dataframe(df)` (main.py:358).
5. `gen = ReportGenerator(profile=profile, df=df, claude=claude, model_selection=MODEL_SELECTION, model_narrative=MODEL_NARRATIVE, custom_prompt=custom_prompt)` (main.py:359–363).
6. `report = gen.build_report(deep=False)` (main.py:364) — which internally does `generate_charts()` → `generate_narrative()` → assembles `Report(generated_at, summary, data_quality, key_metrics=self._key_metrics, charts=[ChartWithCaption...], layout=[first 5 main / rest sidebar], metadata={...})` (report_generator.py:418–462).
7. If sampled, prepend the sampling note to `report.data_quality` (main.py:390–392).
8. The stored shape is `report.model_dump()` (main.py:412). `run_report` returns this dict.

The helper imports `sample_for_analysis`, `MAX_ANALYSIS_ROWS` from `main`, and `MODEL_SELECTION`/`MODEL_NARRATIVE` from `llm_config`, so any production change to sampling, models, or assembly flows into the harness automatically.

---

## File Structure

| Path | Created/Modified | One responsibility |
|---|---|---|
| `requirements.txt` | Modified | Add `pytest-timeout` dep. |
| `pytest.ini` | Modified | Add `timeout = 60` + `timeout_method = thread` so no test can hang a run. |
| `tests/<the-network-test>.py` | Modified | Mock the one network-bound call so the suite is offline-safe (test identified at runtime; see Task 3). |
| `qa/__init__.py` | Created | Marks `qa/` a package (empty). |
| `qa/generators.py` | Created | ~15 synthetic edge-case DataFrame generators `gen_<name>() -> (name, df)` + `SYNTHETIC` registry. |
| `qa/pipeline.py` | Created | `run_report(df, custom_prompt=None) -> dict` (fidelity helper mirroring main.py) + `RunResult` dataclass. |
| `qa/run_eval.py` | Created | CLI runner: iterate synthetic + discover `qa/datasets/*.csv`, call pipeline + validators + judge, write `qa/results/<ts>/`. |
| `qa/validators.py` | Created | Pure deterministic checks `(df, report) -> list[Issue]` + `Issue` dataclass. |
| `qa/judge.py` | Created | `judge_report(profile_text, report) -> JudgeVerdict` via real Haiku forced tool call + `JudgeVerdict`/`ChartVerdict` dataclasses. |
| `qa/report_writer.py` | Created | `write_report(run_dir, per_dataset_results)` → `report.md` + `summary.json` with PASS/WARN/FAIL roll-up. |
| `qa/datasets/.gitkeep` | Created | Keeps the drop-in real-CSV folder under version control. |
| `qa/results/.gitkeep` | Created | Keeps the results folder; run outputs are gitignored. |
| `qa/README.md` | Created | How to run `make qa`, flags, where results land, how to add datasets. |
| `qa/tests/__init__.py` | Created | Marks `qa/tests/` a package. |
| `qa/tests/test_generators.py` | Created | Unit tests: each generator returns intended shape/dtype. |
| `qa/tests/test_validators.py` | Created | Unit tests: each check fires on a crafted bad report, passes a good one (incl. year-as-int regression). |
| `qa/tests/test_pipeline.py` | Created | Light test: `run_report` returns the prod report shape using FakeClaude (no network). |
| `qa/tests/test_judge.py` | Created | Light test: `judge_report` parses a forced-tool-call response into `JudgeVerdict` (FakeClaude). |
| `qa/tests/test_report_writer.py` | Created | Unit test: PASS/WARN/FAIL roll-up + files written. |
| `.gitignore` | Modified | Ignore `qa/results/*` but keep `qa/results/.gitkeep`. |
| `Makefile` | Modified | Add `qa:` target → `venv/bin/python qa/run_eval.py`. |
| `docs/FUTURE-IMPROVEMENTS.md` | Modified | Log any baseline-run findings not fixed inline (Task 22). |

---

## Phase 1 — Test-suite health (offline-safe + hang-proof)

### Task 1 — Add `pytest-timeout` to requirements

**File:** `requirements.txt`

Under the `# Tests (dev; harmless in image)` block (currently ends with `pytest-asyncio==0.23.5`), add:

```
pytest-timeout==2.3.1  # hard per-test timeout so a hung network call can't stall a run
```

**Install + verify (WAIT for completion before the next command):**

```
venv/bin/python -m pip install pytest-timeout==2.3.1
venv/bin/python -m pytest --help 2>&1 | grep -- --timeout
```

Expected: `--timeout` appears in the help output (e.g. `--timeout=TIMEOUT ...`), confirming the plugin loaded.

**Commit:** `qa: add pytest-timeout dep`

---

### Task 2 — Wire the default timeout into pytest.ini

**File:** `pytest.ini`

Append two lines under `pythonpath = src/api`:

```
timeout = 60
timeout_method = thread
```

Full file after edit:

```
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -ra -q --strict-markers
markers =
    e2e: end-to-end tests that hit real Claude (opt-in via RUN_E2E=true)
pythonpath = src/api
timeout = 60
timeout_method = thread
```

`thread` is chosen over the default `signal` method so the timeout works even inside threads / when signals are unavailable, and produces a full traceback of where the hang occurred.

**Verify (WAIT):**

```
venv/bin/python -m pytest tests/unit/test_llm_config.py -q
```

Expected: passes (e.g. `... passed`), confirming the config still parses with the new keys.

**Commit:** `qa: default 60s per-test timeout (thread method)`

---

### Task 3 — Find and mock the network-bound straggler

**Goal:** Identify the one test that hangs on a live network call and patch it so the suite is fully offline-safe. The most likely culprit is the auth path: `src/api/auth.py` builds a real `PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json")` and `verify_token()` calls `get_signing_key_from_jwt(token)` over HTTP whenever `_public_key` is not injected. The auth **unit** tests (`tests/unit/test_auth.py`) all pass `_public_key=`, so they are safe — the straggler is an **integration** test that drives the real `deps.verify_token` → live JWKS fetch with a Bearer token (candidates: `tests/integration/test_auth_gating.py`, `test_credit_gating.py`, `test_claim_and_reports.py`).

**Step 3a — locate it (run with a short timeout so the hang surfaces as a traceback; WAIT for completion):**

```
venv/bin/python -m pytest -q --timeout=30
```

Read the failure block. A timed-out test prints `Failed: Timeout >30.0s` plus a traceback whose deepest frames point at the offending call. Note:
- the **test node id** (e.g. `tests/integration/test_auth_gating.py::test_authed_user_can_generate`), and
- the **call being made** in the traceback's lowest app frame.

**Step 3b — patch by the call you actually see.** Apply the matching case:

**Case A (expected) — JWKS / `verify_token` over HTTP.** The traceback bottoms out in `auth.py` `_get_jwks_client` / `get_signing_key_from_jwt` (or urllib/socket beneath it). Fix: in the identified test (or its shared `ctx`/fixture setup), force `deps.verify_token` to a pure function so no JWKS fetch happens. Add a fixture at the top of that test module:

```python
import pytest


@pytest.fixture(autouse=True)
def _no_live_jwks(monkeypatch):
    """Offline-safe: never fetch Supabase JWKS in tests. Authed cases inject a
    fixed user UUID; anything else verifies as unauthenticated (None)."""
    from uuid import UUID
    monkeypatch.setattr(
        "deps.verify_token",
        lambda token: UUID("00000000-0000-0000-0000-000000000001")
        if token == "good"
        else None,
    )
```

Then ensure the authed requests in that file send `Authorization: Bearer good` (matching the existing pattern in `tests/unit/test_get_identity.py::test_valid_bearer_is_authenticated`, which already does `monkeypatch.setattr("deps.verify_token", lambda t: uid)`). If the file already builds identities via `tests/helpers/fake_auth.py` and never calls `verify_token`, the hang is elsewhere — fall through to Case B/C.

**Case B — alerting / Sentry / Slack HTTP.** Traceback bottoms out in `alerting.report_alert` → `sentry_sdk.capture_message` doing a network transport (only if `SENTRY_DSN` is set in the test env). Fix: in the offending test, `monkeypatch.setattr("alerting.report_alert", lambda *a, **k: None)`.

**Case C — any other outbound HTTP (requests/httpx/urlopen).** Patch the specific client call named in the traceback to a stub returning a canned object, in the offending test only. Do not add a global socket-blocker (it would break the intentional real-Claude `e2e` tests, though those are already gated by `RUN_E2E`).

**Step 3c — verify the fix (WAIT):**

```
venv/bin/python -m pytest <the-test-node-id> -q --timeout=30
```

Expected: the previously-hanging test now passes within the timeout.

**Step 3d — full suite is green and hang-proof (WAIT; this is the longer run — do not fire other commands into it):**

```
venv/bin/python -m pytest -q --timeout=30
```

Expected: all tests pass (e2e tests skip because `RUN_E2E` is unset), zero `Timeout` failures, completes well under any single 30s test cap.

**Commit:** `qa: mock the network-bound straggler so the suite is offline-safe`

---

## Phase 2 — Generators, pipeline, runner

### Task 4 — Package skeleton + gitignore + folder keeps

**Files:** `qa/__init__.py`, `qa/tests/__init__.py`, `qa/datasets/.gitkeep`, `qa/results/.gitkeep`, `.gitignore`

- `qa/__init__.py`:

```python
"""ChartSage QA / Eval Harness — on-demand regression tooling (not deployed)."""
```

- `qa/tests/__init__.py`: empty file.
- `qa/datasets/.gitkeep`: empty file (drop real CSVs here; auto-discovered).
- `qa/results/.gitkeep`: empty file.
- `.gitignore` — add at the end:

```
# QA harness run outputs (keep the folder, ignore the runs)
qa/results/*
!qa/results/.gitkeep
```

**Verify (WAIT):**

```
venv/bin/python -c "import importlib.util; print(importlib.util.find_spec('qa') is not None)"
git -C . check-ignore qa/results/somerun || echo "NOT-IGNORED"
git -C . check-ignore qa/results/.gitkeep || echo "GITKEEP-TRACKED"
```

Run the find_spec check from the `qa/` parent (repo root). Expected: `True`; the second line prints `qa/results/somerun` (ignored); the third prints `GITKEEP-TRACKED` (the `!` negation keeps `.gitkeep`).

**Commit:** `qa: package skeleton, dataset/result folders, gitignore results`

---

### Task 5 — Synthetic generators (TDD)

**Files:** `qa/tests/test_generators.py` (failing test first), then `qa/generators.py`

**5a — write the failing test** `qa/tests/test_generators.py`:

```python
"""Each synthetic generator returns (name, DataFrame) with its intended edge-case shape."""
import pandas as pd
import pytest

from qa.generators import SYNTHETIC


def test_registry_has_at_least_15_generators():
    assert len(SYNTHETIC) >= 15


def test_every_generator_returns_name_and_dataframe():
    seen_names = set()
    for gen in SYNTHETIC:
        name, df = gen()
        assert isinstance(name, str) and name
        assert name not in seen_names, f"duplicate generator name {name!r}"
        seen_names.add(name)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] >= 1


def test_every_generator_is_chartable_shape_or_intentionally_tiny():
    # The pipeline requires >=2 cols and >=1 row; only the intentional 'single_row_tiny'
    # and 'duplicate_columns' edge cases may probe the guards. All others must be runnable.
    for gen in SYNTHETIC:
        name, df = gen()
        assert df.shape[0] >= 1, f"{name} produced 0 rows"


def test_year_as_int_is_integer_dtype():
    name, df = _by_name("year_as_int")
    assert "year" in df.columns
    assert pd.api.types.is_integer_dtype(df["year"])
    # Multiple distinct years so a working line chart has >=2 points.
    assert df["year"].nunique() >= 5


def test_tall_dataset_exceeds_sampling_threshold():
    name, df = _by_name("tall_100k")
    assert len(df) >= 100_000


def test_wide_dataset_has_50_plus_columns():
    name, df = _by_name("wide_50col")
    assert df.shape[1] >= 50


def test_high_cardinality_category_500_plus():
    name, df = _by_name("high_cardinality_cat")
    cat_col = [c for c in df.columns if df[c].dtype == object][0]
    assert df[cat_col].nunique() >= 500


def test_id_like_big_ints_are_unique_and_large():
    name, df = _by_name("id_like_bigints")
    id_col = [c for c in df.columns if c.endswith("_id")][0]
    assert df[id_col].nunique() == len(df)
    assert int(df[id_col].min()) > 10_000


def _by_name(target: str):
    for gen in SYNTHETIC:
        name, df = gen()
        if name == target:
            return name, df
    raise AssertionError(f"no generator named {target!r}")
```

Run it (WAIT) — expected: collection/import error or failures because `qa/generators.py` doesn't exist yet:

```
venv/bin/python -m pytest qa/tests/test_generators.py -q
```

**5b — implement** `qa/generators.py`:

```python
"""Synthetic edge-case dataset generators for the QA harness.

Each generator is a zero-arg function returning (name, DataFrame). `name` is a
stable slug used in result filenames and the report. Each targets one known
failure mode. SYNTHETIC is the registry the runner iterates.

Generators are deterministic (fixed seed) so reruns are comparable.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

_RNG = np.random.default_rng(1234)


def gen_year_as_int() -> tuple[str, pd.DataFrame]:
    """Bare integer years + a measure. The motivating bug: naive to_datetime
    turns 2019/2020/... into epoch-nanosecond timestamps (~1970), collapsing a
    multi-year line to one cluster. A correct pipeline yields a >=5-point line."""
    years = list(range(2015, 2024))
    rows = []
    for y in years:
        for _ in range(40):
            rows.append({"year": y, "revenue": float(_RNG.integers(1000, 9000)),
                         "region": _RNG.choice(["north", "south", "east", "west"])})
    return "year_as_int", pd.DataFrame(rows)


def gen_mixed_date_formats() -> tuple[str, pd.DataFrame]:
    """A date column whose cells use clashing formats incl. an Excel serial int."""
    raw = ["2020-01-15", "02/03/2020", "03-04-20", "44105", "Jan 2020",
           "2020-05-01", "06/07/2020", "07-08-20", "44200", "Feb 2020"] * 12
    return "mixed_date_formats", pd.DataFrame({
        "event_date": raw,
        "amount": [float(_RNG.integers(10, 500)) for _ in raw],
    })


def gen_nan_heavy() -> tuple[str, pd.DataFrame]:
    """Columns that are mostly null (one >95% null to trip the unusable anomaly)."""
    n = 200
    almost_all_null = [float(_RNG.integers(1, 100)) if i % 50 == 0 else np.nan for i in range(n)]
    half_null = [float(_RNG.integers(1, 100)) if i % 2 == 0 else np.nan for i in range(n)]
    cat = [_RNG.choice(["a", "b", "c"]) if i % 3 else None for i in range(n)]
    return "nan_heavy", pd.DataFrame({
        "mostly_null": almost_all_null, "half_null": half_null, "category": cat,
    })


def gen_high_cardinality_cat() -> tuple[str, pd.DataFrame]:
    """A categorical with 500+ distinct values (must roll into top-N + Other,
    not error out or chart every value)."""
    n = 3000
    cats = [f"sku_{_RNG.integers(0, 600)}" for _ in range(n)]
    return "high_cardinality_cat", pd.DataFrame({
        "sku": cats, "units": [float(_RNG.integers(1, 50)) for _ in range(n)],
    })


def gen_unicode_emoji() -> tuple[str, pd.DataFrame]:
    """Non-ASCII + emoji category labels (must survive JSON + chart x/labels)."""
    labels = ["café", "naïve", "Zürich", "東京", "🚀rocket", "Москва", "São Paulo", "δοκιμή"]
    n = 240
    return "unicode_emoji", pd.DataFrame({
        "place": [labels[i % len(labels)] for i in range(n)],
        "score": [float(_RNG.integers(1, 100)) for _ in range(n)],
    })


def gen_currency_strings() -> tuple[str, pd.DataFrame]:
    """Money stored as strings with symbols/grouping (£1,200 / $1.2k / 1.234,56) —
    must NOT be silently charted as a numeric measure."""
    vals = ["£1,200", "$1.2k", "1.234,56", "€980", "$3,400", "£500", "2.000,00", "$1.1k"]
    n = 200
    return "currency_strings", pd.DataFrame({
        "region": [_RNG.choice(["north", "south", "east", "west"]) for _ in range(n)],
        "revenue": [vals[i % len(vals)] for i in range(n)],
    })


def gen_single_row_tiny() -> tuple[str, pd.DataFrame]:
    """A 2-row frame: probes the >=1-row guard and degenerate aggregations."""
    return "single_row_tiny", pd.DataFrame({
        "category": ["a", "b"], "value": [10.0, 20.0],
    })


def gen_duplicate_columns() -> tuple[str, pd.DataFrame]:
    """Two columns named identically (post-lower-casing collision) — the loader
    lower-cases names; the harness must not crash on the duplicate."""
    df = pd.DataFrame(
        [[1, "x", 5.0], [2, "y", 6.0], [3, "z", 7.0]],
        columns=["Value", "label", "value"],
    )
    return "duplicate_columns", df


def gen_all_categorical() -> tuple[str, pd.DataFrame]:
    """No numeric columns at all — count-based charts only."""
    n = 300
    return "all_categorical", pd.DataFrame({
        "color": [_RNG.choice(["red", "green", "blue"]) for _ in range(n)],
        "size": [_RNG.choice(["s", "m", "l", "xl"]) for _ in range(n)],
        "shape": [_RNG.choice(["circle", "square", "tri"]) for _ in range(n)],
    })


def gen_all_numeric() -> tuple[str, pd.DataFrame]:
    """No categoricals/dates — distributions, scatter, correlation only."""
    n = 500
    a = _RNG.normal(50, 10, n)
    return "all_numeric", pd.DataFrame({
        "a": a, "b": a * 1.5 + _RNG.normal(0, 5, n), "c": _RNG.uniform(0, 100, n),
    })


def gen_wide_50col() -> tuple[str, pd.DataFrame]:
    """50+ columns — exercises column-handling at width."""
    n = 100
    data = {f"num_{i}": _RNG.normal(0, 1, n) for i in range(45)}
    for i in range(6):
        data[f"cat_{i}"] = [_RNG.choice(["p", "q", "r"]) for _ in range(n)]
    return "wide_50col", pd.DataFrame(data)


def gen_tall_100k() -> tuple[str, pd.DataFrame]:
    """100k+ rows — exercises sample_for_analysis (MAX_ANALYSIS_ROWS=50000)."""
    n = 120_000
    return "tall_100k", pd.DataFrame({
        "category": _RNG.choice(["a", "b", "c", "d", "e"], n),
        "value": _RNG.normal(100, 25, n),
    })


def gen_boolean_ish() -> tuple[str, pd.DataFrame]:
    """Yes/No, 0/1, True/False columns — low-cardinality, boolean-like."""
    n = 240
    return "boolean_ish", pd.DataFrame({
        "subscribed": [_RNG.choice(["Yes", "No"]) for _ in range(n)],
        "active": [int(_RNG.integers(0, 2)) for _ in range(n)],
        "flag": [bool(_RNG.integers(0, 2)) for _ in range(n)],
        "amount": [float(_RNG.integers(1, 100)) for _ in range(n)],
    })


def gen_mixed_type_column() -> tuple[str, pd.DataFrame]:
    """One column mixing ints and strings (object dtype) — type sniffing must
    not treat it as clean numeric."""
    n = 200
    col = [int(_RNG.integers(1, 100)) if i % 2 else f"n/a_{i}" for i in range(n)]
    return "mixed_type_column", pd.DataFrame({
        "region": [_RNG.choice(["north", "south"]) for _ in range(n)],
        "messy": col,
    })


def gen_id_like_bigints() -> tuple[str, pd.DataFrame]:
    """A unique big-integer id column (>=50 rows so the identifier heuristic
    fires) — must be classified identifier, never charted as a measure."""
    n = 300
    base = 1_000_000_000
    return "id_like_bigints", pd.DataFrame({
        "order_id": [base + i for i in range(n)],
        "region": [_RNG.choice(["north", "south", "east", "west"]) for _ in range(n)],
        "amount": [float(_RNG.integers(1, 500)) for _ in range(n)],
    })


def gen_negatives_outliers() -> tuple[str, pd.DataFrame]:
    """Negative values in a 'should-be-non-negative' column + an extreme outlier."""
    n = 300
    vals = _RNG.normal(50, 8, n)
    vals[0] = -200.0          # negative in a 'count'-named column -> anomaly
    vals[1] = 50_000.0        # extreme outlier -> anomaly
    return "negatives_outliers", pd.DataFrame({
        "region": [_RNG.choice(["north", "south", "east", "west"]) for _ in range(n)],
        "count": vals,
    })


SYNTHETIC: list[Callable[[], tuple[str, pd.DataFrame]]] = [
    gen_year_as_int,
    gen_mixed_date_formats,
    gen_nan_heavy,
    gen_high_cardinality_cat,
    gen_unicode_emoji,
    gen_currency_strings,
    gen_single_row_tiny,
    gen_duplicate_columns,
    gen_all_categorical,
    gen_all_numeric,
    gen_wide_50col,
    gen_tall_100k,
    gen_boolean_ish,
    gen_mixed_type_column,
    gen_id_like_bigints,
    gen_negatives_outliers,
]
```

**Verify (WAIT):**

```
venv/bin/python -m pytest qa/tests/test_generators.py -q
```

Expected: all tests pass (16 generators ≥ 15, year-as-int integer dtype + ≥5 distinct years, tall ≥100k, wide ≥50 cols, high-card ≥500, id-like unique+large).

**Commit:** `qa: 16 synthetic edge-case generators + registry`

---

### Task 6 — Pipeline fidelity helper + RunResult (light test)

**Files:** `qa/tests/test_pipeline.py` (failing test first), then `qa/pipeline.py`

This is an integration piece (it imports the real generator), but Task 6's test uses FakeClaude so it stays offline. The real-Haiku exercise happens in the runner + baseline (Tasks 9, 21).

**6a — failing test** `qa/tests/test_pipeline.py`:

```python
"""run_report mirrors the production endpoint assembly. Uses FakeClaude (offline)."""
import pandas as pd
import types

from qa.pipeline import run_report, RunResult
from tests.helpers.fake_claude import FakeClaude, tool_use


def _fake_claude_for_sales():
    return FakeClaude([
        {"tool_calls": [
            tool_use("frequency_bar_chart", {"column": "region", "title": "By region", "intent": "mix"}),
            tool_use("aggregation_bar_chart", {"value_col": "revenue", "group_col": "region",
                                               "agg": "sum", "title": "Rev by region", "intent": "compare"}),
            tool_use("line_chart", {"date_col": "order_date", "value_col": "revenue", "agg": "sum",
                                    "granularity": "month", "title": "Rev over time", "intent": "trend"}),
        ]},
        {"tool_calls": []},  # reach-for-more proposes nothing
        {"tool_calls": [tool_use("submit_narrative", {
            "summary": "Revenue rises over time.",
            "captions": ["A.", "B.", "C."], "data_quality": []})]},
    ])


def _sales_df():
    return pd.DataFrame({
        "order_id": list(range(1, 16)),
        "region": ["north", "south", "east", "west", "north", "south", "east", "west",
                   "north", "south", "east", "west", "north", "south", "east"],
        "revenue": [1200.0, 850.0, 2100.0, 1500.0, 1800.0, 900.0, 2400.0, 1750.0,
                    1300.0, 950.0, 2200.0, 1650.0, 1400.0, 1000.0, 2300.0],
        "order_date": pd.date_range("2024-01-01", periods=15, freq="3D"),
    })


def test_run_report_returns_prod_report_shape(monkeypatch):
    # Inject FakeClaude as the shared client so no network call happens.
    fake = _fake_claude_for_sales()
    monkeypatch.setattr(
        "qa.pipeline._build_claude",
        lambda: types.SimpleNamespace(messages_create=fake),
    )
    result = run_report(_sales_df())
    assert isinstance(result, RunResult)
    assert result.error is None
    rep = result.report
    # Exact production keys from report_generator.build_report / Report model.
    for key in ("generated_at", "summary", "data_quality", "key_metrics",
                "charts", "layout", "metadata"):
        assert key in rep, f"missing report key {key!r}"
    assert rep["summary"] == "Revenue rises over time."
    assert len(rep["charts"]) == 3
    # Layout: first 5 main / rest sidebar (here all 3 are main).
    assert all(e["position"] == "main" for e in rep["layout"])
    assert result.elapsed_ms >= 0


def test_run_report_captures_exception(monkeypatch):
    def boom():
        raise RuntimeError("claude down")
    monkeypatch.setattr("qa.pipeline._build_claude", boom)
    result = run_report(_sales_df())
    assert result.report is None
    assert result.error is not None
    assert "claude down" in result.error
```

Run (WAIT) — expected import/collection failure (`qa/pipeline.py` missing):

```
venv/bin/python -m pytest qa/tests/test_pipeline.py -q
```

**6b — implement** `qa/pipeline.py`. It imports the production sequence pieces directly so it can't drift:

```python
"""Fidelity helper: run a DataFrame through the EXACT production generation
sequence used by POST /generate-report, returning the stored report dict (or a
captured exception). This is the harness's anti-drift guard — it imports the
real endpoint helpers and the real ReportGenerator, so any production change in
sampling, models, or report assembly flows through automatically.

Mirrors src/api/main.py generate_report() lines ~337-392 and
report_generator.ReportGenerator.build_report().
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv

# Make the production package importable when run outside pytest (from repo root).
_API_DIR = Path(__file__).resolve().parent.parent / "src" / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

load_dotenv()  # same as main.py:38 — pulls ANTHROPIC_API_KEY etc. from .env

from claude_client import ClaudeClient                       # noqa: E402
from llm_config import MODEL_NARRATIVE, MODEL_SELECTION      # noqa: E402
from main import MAX_ANALYSIS_ROWS, sample_for_analysis      # noqa: E402  (reuse prod sampling)
from profile import profile_dataframe                        # noqa: E402
from report_generator import ReportGenerator                 # noqa: E402


@dataclass
class RunResult:
    """Outcome of one dataset run: the stored report dict OR an error string,
    plus timing and the post-sampling row/col counts actually analyzed."""
    name: str
    report: Optional[dict]
    error: Optional[str]
    elapsed_ms: int
    rows_analyzed: int
    cols_analyzed: int
    was_sampled: bool
    original_rows: int


def _build_claude() -> ClaudeClient:
    """Construct the REAL Anthropic client exactly like main.get_claude_client()."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (expected in .env)")
    return ClaudeClient(api_key=api_key)


def run_report(df: pd.DataFrame, custom_prompt: str | None = None, name: str = "") -> RunResult:
    """Reproduce the /generate-report body and return a RunResult.

    Sequence (main.py): lower-case columns -> sample_for_analysis -> 2-col/1-row
    guards -> profile_dataframe -> ReportGenerator(...) -> build_report(deep=False)
    -> prepend sampling note -> return report.model_dump().
    """
    started = time.perf_counter()
    original_rows = int(df.shape[0])
    try:
        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]          # main.py:337
        df, was_sampled, total_rows = sample_for_analysis(df)       # main.py:339

        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns to chart.")   # main.py:343-344
        if df.shape[0] < 1:
            raise ValueError("File has no data rows.")                        # main.py:345-346

        profile = profile_dataframe(df)                            # main.py:358
        claude = _build_claude()
        gen = ReportGenerator(
            profile=profile, df=df, claude=claude,
            model_selection=MODEL_SELECTION, model_narrative=MODEL_NARRATIVE,
            custom_prompt=custom_prompt,
        )                                                          # main.py:359-363
        report = gen.build_report(deep=False)                      # main.py:364

        if was_sampled:                                            # main.py:390-392
            note = (f"Analyzed a representative random sample of "
                    f"{MAX_ANALYSIS_ROWS:,} of {total_rows:,} rows.")
            report.data_quality = [note] + list(report.data_quality or [])

        report_dict: dict[str, Any] = report.model_dump()         # main.py:412 (stored shape)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return RunResult(
            name=name, report=report_dict, error=None, elapsed_ms=elapsed_ms,
            rows_analyzed=int(df.shape[0]), cols_analyzed=int(df.shape[1]),
            was_sampled=was_sampled, original_rows=total_rows,
        )
    except Exception as e:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return RunResult(
            name=name, report=None,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            elapsed_ms=elapsed_ms, rows_analyzed=0, cols_analyzed=0,
            was_sampled=False, original_rows=original_rows,
        )
```

**Verify (WAIT):**

```
venv/bin/python -m pytest qa/tests/test_pipeline.py -q
```

Expected: both tests pass.

**Commit:** `qa: pipeline.run_report fidelity helper + RunResult`

---

### Task 7 — Runner CLI (write JSON per dataset)

**File:** `qa/run_eval.py`

No new failing unit test (it's CLI glue exercised by the smoke in Task 21). Implement directly; validators/judge/report_writer hooks are wired here and become live as those modules land (Tasks 12–20). Until then, the runner imports them lazily so this task runs standalone for a JSON-only smoke.

```python
"""QA harness runner: feed synthetic generators + dropped-in CSVs through the
real production pipeline, validate (deterministic + judge), and write results.

Usage (from repo root):
    venv/bin/python qa/run_eval.py [--only synthetic|real|<name>] [--no-judge] [--limit N]

Outputs land in qa/results/<timestamp>/:
    <dataset>.json   per-dataset: run meta, report, deterministic issues, judge verdict
    report.md        scannable PASS/WARN/FAIL roll-up
    summary.json     machine-readable roll-up
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_QA_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _QA_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))   # so `import qa.*` works when run as a script

from qa.generators import SYNTHETIC          # noqa: E402
from qa.pipeline import RunResult, run_report  # noqa: E402


def _discover_real() -> list[tuple[str, pd.DataFrame]]:
    """Auto-discover qa/datasets/*.csv (name = filename stem)."""
    out: list[tuple[str, pd.DataFrame]] = []
    for csv in sorted((_QA_DIR / "datasets").glob("*.csv")):
        try:
            df = pd.read_csv(csv)
        except UnicodeDecodeError:
            df = pd.read_csv(csv, encoding="latin1")
        out.append((csv.stem, df))
    return out


def _collect(only: str | None) -> list[tuple[str, pd.DataFrame]]:
    synthetic = [gen() for gen in SYNTHETIC]
    real = _discover_real()
    if only == "synthetic":
        return synthetic
    if only == "real":
        return real
    if only:  # a specific dataset name
        both = synthetic + real
        picked = [(n, d) for (n, d) in both if n == only]
        if not picked:
            raise SystemExit(f"No dataset named {only!r} (synthetic or in qa/datasets/).")
        return picked
    return synthetic + real


def _to_jsonable(result: RunResult, det_issues, verdict) -> dict:
    return {
        "name": result.name,
        "error": result.error,
        "elapsed_ms": result.elapsed_ms,
        "rows_analyzed": result.rows_analyzed,
        "cols_analyzed": result.cols_analyzed,
        "was_sampled": result.was_sampled,
        "original_rows": result.original_rows,
        "report": result.report,
        "deterministic_issues": [dataclasses.asdict(i) for i in det_issues],
        "judge": dataclasses.asdict(verdict) if verdict is not None else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ChartSage QA / Eval Harness")
    parser.add_argument("--only", default=None,
                        help="synthetic | real | <dataset-name>")
    parser.add_argument("--no-judge", action="store_true",
                        help="deterministic checks only (free, no Haiku judge calls)")
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N datasets")
    args = parser.parse_args(argv)

    # Lazy imports so a JSON-only smoke works before these modules exist.
    from qa.validators import validate            # noqa: E402
    judge_report = None
    if not args.no_judge:
        from qa.judge import judge_report         # noqa: E402
    from qa.report_writer import write_report     # noqa: E402

    datasets = _collect(args.only)
    if args.limit is not None:
        datasets = datasets[: args.limit]

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = _QA_DIR / "results" / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    per_dataset: list[dict] = []
    for name, df in datasets:
        print(f"[qa] running {name} ({df.shape[0]} rows x {df.shape[1]} cols) ...", flush=True)
        result = run_report(df, name=name)
        det_issues = validate(df, result.report) if result.report is not None else \
            validate(df, None)
        verdict = None
        if judge_report is not None and result.report is not None:
            try:
                from profile import profile_dataframe
                profile_text = profile_dataframe(df).to_text()
                verdict = judge_report(profile_text, result.report)
            except Exception as e:  # judge must never sink the whole run
                print(f"[qa] judge failed for {name}: {e}", flush=True)
        payload = _to_jsonable(result, det_issues, verdict)
        (run_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))
        per_dataset.append(payload)

    write_report(run_dir, per_dataset)
    print(f"[qa] done -> {run_dir}/report.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

**Verify — JSON-only smoke on the two cheapest synthetic datasets (no judge; WAIT, this calls real Haiku for selection+narrative so it takes ~10-30s/dataset):**

```
venv/bin/python qa/run_eval.py --only single_row_tiny --no-judge
```

Expected: prints `[qa] running single_row_tiny ...` then `[qa] done -> qa/results/<ts>/report.md`; a `qa/results/<ts>/single_row_tiny.json` exists with a populated `report` (or a captured `error`). (`report.md` is written by report_writer, which lands in Task 20; if running this before Task 20, temporarily expect an ImportError on `write_report` and re-run after Task 20. Sequence the tasks in order to avoid this.)

**Commit:** `qa: run_eval CLI — pipeline runner + per-dataset JSON`

---

## Phase 3 — Deterministic validators (TDD each check)

### Task 8 — Issue dataclass + generation-error + degenerate-chart checks (TDD)

**Files:** `qa/tests/test_validators.py` (failing test first), then `qa/validators.py`

The validators are pure functions and are the heart of the regression gate — TDD every check with a crafted good + bad report. Task 8 establishes the module + the first two checks; Tasks 9–11 add the rest to the same files.

**8a — failing test (start `qa/tests/test_validators.py`):**

```python
"""Deterministic validators: each check fires on a crafted bad report and passes a good one."""
import pandas as pd

from qa.validators import validate, Issue


# ---- helpers ---------------------------------------------------------------

def _df_sales():
    return pd.DataFrame({
        "region": ["north", "south", "east", "north", "south", "east"],
        "revenue": [100.0, 200.0, 300.0, 150.0, 250.0, 350.0],
    })


def _good_bar_report():
    """A clean frequency bar over region with a real groupby-consistent y."""
    return {
        "generated_at": "2026-06-04T00:00:00",
        "summary": "Region counts are even across the three regions.",
        "data_quality": [],
        "key_metrics": [{"label": "Total revenue", "value": 1350.0, "format": "currency"}],
        "charts": [{
            "chart_id": "c1",
            "caption": "Counts by region.",
            "spec": {
                "kind": "bar", "title": "By region", "intent": "mix",
                "x": ["north", "south", "east"], "y": [2, 2, 2],
                "x_label": "region", "y_label": "Count",
                "x_display_type": "category", "y_display_type": "count",
                "source_columns": ["region"], "data_point_count": 6,
            },
        }],
        "layout": [{"chart_id": "c1", "position": "main", "order": 0}],
        "metadata": {},
    }


def _codes(issues: list[Issue]) -> set[str]:
    return {i.code for i in issues}


# ---- generation error ------------------------------------------------------

def test_generation_error_when_report_none():
    issues = validate(_df_sales(), None)
    assert any(i.code == "generation_error" and i.severity == "fail" for i in issues)


def test_good_report_has_no_fail_issues():
    issues = validate(_df_sales(), _good_bar_report())
    fails = [i for i in issues if i.severity == "fail"]
    assert fails == [], f"unexpected fails: {fails}"


# ---- degenerate chart ------------------------------------------------------

def test_empty_xy_flagged():
    rep = _good_bar_report()
    rep["charts"][0]["spec"]["x"] = []
    rep["charts"][0]["spec"]["y"] = []
    issues = validate(_df_sales(), rep)
    assert "degenerate_chart" in _codes(issues)


def test_line_with_one_distinct_x_flagged():
    rep = _good_bar_report()
    spec = rep["charts"][0]["spec"]
    spec["kind"] = "line"
    spec["x"] = ["2020", "2020", "2020"]
    spec["y"] = [1.0, 2.0, 3.0]
    issues = validate(_df_sales(), rep)
    assert any(i.code == "degenerate_chart" and i.chart_id == "c1" for i in issues)


def test_all_identical_y_flagged():
    rep = _good_bar_report()
    rep["charts"][0]["spec"]["y"] = [5, 5, 5]
    issues = validate(_df_sales(), rep)
    assert "degenerate_chart" in _codes(issues)


def test_all_nan_y_flagged():
    rep = _good_bar_report()
    rep["charts"][0]["spec"]["y"] = [None, None, None]
    issues = validate(_df_sales(), rep)
    assert "degenerate_chart" in _codes(issues)


def test_all_zero_y_flagged():
    rep = _good_bar_report()
    rep["charts"][0]["spec"]["y"] = [0, 0, 0]
    issues = validate(_df_sales(), rep)
    assert "degenerate_chart" in _codes(issues)
```

Run (WAIT) — expected import failure (`qa/validators.py` missing):

```
venv/bin/python -m pytest qa/tests/test_validators.py -q
```

**8b — implement** `qa/validators.py` (Issue + the first two checks; later tasks append more checks to the `validate` aggregator):

```python
"""Deterministic validators: pure functions over (df, report_dict) returning a
flat list of Issue. No network, no LLM. These are the harness's hard gate.

`validate(df, report)` runs every check and concatenates their issues. A report
of None means generation crashed (the pipeline captured an exception).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from report_generator import MIN_CHARTS_FOR_NO_FALLBACK  # the real selection floor (==3)


@dataclass
class Issue:
    severity: str            # 'fail' | 'warn'
    code: str
    message: str
    chart_id: Optional[str] = None


# ---- small helpers ---------------------------------------------------------

def _charts(report: dict) -> list[dict]:
    return list(report.get("charts") or [])


def _spec(chart: dict) -> dict:
    return chart.get("spec") or {}


def _numeric_ys(spec: dict) -> list[float]:
    """Flatten a chart's y values to floats (drop non-finite/non-numeric).
    Handles flat y, series-with-'y', series-with-'data', and treemap nodes."""
    out: list[float] = []

    def _push(v: Any):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return
        if math.isfinite(f):
            out.append(f)

    if spec.get("y"):
        for v in spec["y"]:
            _push(v)
    for s in (spec.get("series") or []):
        for key in ("y", "data"):
            for v in (s.get(key) or []):
                _push(v)
        if "value" in s:
            _push(s.get("value"))
    for n in (spec.get("nodes") or []):
        _push(n.get("value"))
        for child in (n.get("children") or []):
            _push(child.get("value"))
    return out


# ---- checks ----------------------------------------------------------------

def check_generation_error(df: pd.DataFrame, report: Optional[dict]) -> list[Issue]:
    if report is None:
        return [Issue("fail", "generation_error",
                      "Report generation raised an exception (no report produced).")]
    return []


def check_degenerate_charts(df: pd.DataFrame, report: dict) -> list[Issue]:
    issues: list[Issue] = []
    for chart in _charts(report):
        spec = _spec(chart)
        cid = chart.get("chart_id")
        kind = spec.get("kind")
        has_xy = bool(spec.get("x")) and bool(spec.get("y"))
        has_series = bool(spec.get("series"))
        has_nodes = bool(spec.get("nodes"))
        if not (has_xy or has_series or has_nodes):
            issues.append(Issue("fail", "degenerate_chart",
                                f"[{kind}] '{spec.get('title')}' has empty x/y, series and nodes.",
                                cid))
            continue
        # Line charts need >=2 distinct x points (the year-as-int bug class).
        if kind == "line":
            xs = spec.get("x") or []
            if xs and len({str(v) for v in xs}) < 2:
                issues.append(Issue("fail", "degenerate_chart",
                                    f"[line] '{spec.get('title')}' has <2 distinct x points "
                                    f"({len(xs)} points, {len(set(map(str, xs)))} distinct).", cid))
            # series-form line: every series collapsed to one distinct period
            for s in (spec.get("series") or []):
                sx = s.get("x") or []
                if sx and len({str(v) for v in sx}) < 2:
                    issues.append(Issue("fail", "degenerate_chart",
                                        f"[line] '{spec.get('title')}' series "
                                        f"'{s.get('name')}' has <2 distinct x points.", cid))
        # Numeric y degeneracy: all-NaN / all-zero / all-identical.
        ys = _numeric_ys(spec)
        raw_count = len(spec.get("y") or []) + sum(
            len(s.get("y") or s.get("data") or []) for s in (spec.get("series") or []))
        if raw_count > 0 and len(ys) == 0:
            issues.append(Issue("fail", "degenerate_chart",
                                f"[{kind}] '{spec.get('title')}' has all-NaN y values.", cid))
        elif len(ys) >= 2:
            if all(v == 0 for v in ys):
                issues.append(Issue("fail", "degenerate_chart",
                                    f"[{kind}] '{spec.get('title')}' has all-zero y values.", cid))
            elif len(set(ys)) == 1:
                issues.append(Issue("warn", "degenerate_chart",
                                    f"[{kind}] '{spec.get('title')}' has all-identical y "
                                    f"values ({ys[0]}).", cid))
    return issues


def validate(df: pd.DataFrame, report: Optional[dict]) -> list[Issue]:
    """Run every deterministic check. If generation failed, return only that."""
    err = check_generation_error(df, report)
    if err:
        return err
    issues: list[Issue] = []
    issues += check_degenerate_charts(df, report)
    # Tasks 9-11 append: consistency, KPI sanity, narrative/floor/labels.
    return issues
```

**Verify (WAIT):**

```
venv/bin/python -m pytest qa/tests/test_validators.py -q
```

Expected: all Task-8 tests pass.

**Commit:** `qa: validators — Issue + generation-error + degenerate-chart checks`

---

### Task 9 — Chart-data consistency check (TDD)

**Files:** append to `qa/tests/test_validators.py`, then `qa/validators.py`

Recompute a count/sum groupby from the source df and compare against the chart's reported y for the cases where it's feasible: a **frequency bar** (`y_display_type == "count"`, single `source_columns`) and an **aggregation/line sum** over one group/date column.

**9a — append failing tests:**

```python
# ---- chart-data consistency ------------------------------------------------

def test_frequency_bar_consistent_passes():
    # _good_bar_report's y=[2,2,2] matches value_counts of region.
    issues = validate(_df_sales(), _good_bar_report())
    assert "chart_data_mismatch" not in _codes(issues)


def test_frequency_bar_inconsistent_flagged():
    rep = _good_bar_report()
    rep["charts"][0]["spec"]["y"] = [2, 2, 99]   # east is really 2, not 99
    issues = validate(_df_sales(), rep)
    assert "chart_data_mismatch" in _codes(issues)


def test_aggregation_sum_bar_consistent_passes():
    rep = _good_bar_report()
    spec = rep["charts"][0]["spec"]
    spec["title"] = "Revenue by region"
    spec["y_display_type"] = "currency"
    spec["y_label"] = "Sum of revenue"
    spec["source_columns"] = ["revenue", "region"]
    # real sums: north=250, south=450, east=650
    spec["x"] = ["east", "south", "north"]
    spec["y"] = [650.0, 450.0, 250.0]
    issues = validate(_df_sales(), rep)
    assert "chart_data_mismatch" not in _codes(issues)


def test_aggregation_sum_bar_inconsistent_flagged():
    rep = _good_bar_report()
    spec = rep["charts"][0]["spec"]
    spec["title"] = "Revenue by region"
    spec["y_label"] = "Sum of revenue"
    spec["source_columns"] = ["revenue", "region"]
    spec["x"] = ["east", "south", "north"]
    spec["y"] = [650.0, 450.0, 999.0]    # north sum is 250, not 999
    issues = validate(_df_sales(), rep)
    assert "chart_data_mismatch" in _codes(issues)
```

Run (WAIT) — expected: the new tests fail (check not implemented), older ones still pass.

**9b — implement: add `check_chart_data_consistency` and wire it into `validate`.**

Insert this function above `validate` in `qa/validators.py`:

```python
def _approx(a: float, b: float, *, rel: float = 0.02, abs_: float = 1e-6) -> bool:
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def check_chart_data_consistency(df: pd.DataFrame, report: dict) -> list[Issue]:
    """Spot-check that bar/line values are derivable from the source df.

    Only the unambiguous cases are checked (everything else is skipped, not failed):
      - frequency bar: y_display_type == 'count' and exactly one source column ->
        compare reported (x->y) to df[col].value_counts().
      - sum aggregation/line: y_label starts with 'Sum of <col>' and there is a
        clear group/date column -> compare reported (x->y) to a groupby-sum.
    A mismatch on a checkable case is a 'fail'.
    """
    issues: list[Issue] = []
    cols = set(df.columns)
    for chart in _charts(report):
        spec = _spec(chart)
        cid = chart.get("chart_id")
        kind = spec.get("kind")
        x = spec.get("x")
        y = spec.get("y")
        srcs = [c for c in (spec.get("source_columns") or []) if c in cols]
        if kind not in ("bar", "line") or not x or not y or len(x) != len(y):
            continue
        reported = {}
        for xv, yv in zip(x, y):
            try:
                reported[str(xv)] = float(yv)
            except (TypeError, ValueError):
                reported = {}
                break
        if not reported:
            continue

        # Case 1: frequency count bar over a single categorical column.
        if (kind == "bar" and spec.get("y_display_type") == "count" and len(srcs) == 1):
            col = srcs[0]
            vc = df[col].dropna().astype(str).value_counts()
            mism = [k for k, v in reported.items()
                    if k in vc.index and not _approx(v, float(vc[k]))]
            if mism:
                issues.append(Issue(
                    "fail", "chart_data_mismatch",
                    f"[bar] '{spec.get('title')}' count(s) for {mism[:3]} "
                    f"don't match df['{col}'].value_counts().", cid))
            continue

        # Case 2: sum aggregation (bar or line) of <value_col> by a group/date col.
        m = re.match(r"sum of (.+)", str(spec.get("y_label", "")).strip(), re.IGNORECASE)
        if m:
            value_col = m.group(1).strip().lower()
            group_candidates = [c for c in srcs if c != value_col]
            if value_col in cols and len(group_candidates) == 1:
                gcol = group_candidates[0]
                work = df[[gcol, value_col]].copy()
                work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
                work = work.dropna()
                grouped = work.groupby(work[gcol].astype(str))[value_col].sum()
                mism = [k for k, v in reported.items()
                        if k in grouped.index and not _approx(v, float(grouped[k]))]
                if mism:
                    issues.append(Issue(
                        "fail", "chart_data_mismatch",
                        f"[{kind}] '{spec.get('title')}' sum(s) for {mism[:3]} "
                        f"don't match a groupby-sum of '{value_col}' by '{gcol}'.", cid))
    return issues
```

Then in `validate`, add after the degenerate check:

```python
    issues += check_chart_data_consistency(df, report)
```

**Verify (WAIT):**

```
venv/bin/python -m pytest qa/tests/test_validators.py -q
```

Expected: all tests pass.

**Commit:** `qa: validators — chart-data consistency (count/sum recompute)`

---

### Task 10 — KPI sanity check (TDD)

**Files:** append to `qa/tests/test_validators.py`, then `qa/validators.py`

Flag non-finite KPI values (fail) and the specific "year-as-a-metric" slip — a metric whose label contains year/years/count/total whose value parses to a 4-digit calendar year 1900–2100 (warn). This is the "2,022 years covered" guard.

**10a — append failing tests:**

```python
# ---- KPI sanity ------------------------------------------------------------

def test_kpi_non_finite_flagged():
    rep = _good_bar_report()
    rep["key_metrics"] = [{"label": "Avg", "value": float("inf"), "format": "number"}]
    issues = validate(_df_sales(), rep)
    assert any(i.code == "kpi_non_finite" and i.severity == "fail" for i in issues)


def test_kpi_year_lookalike_warns():
    rep = _good_bar_report()
    rep["key_metrics"] = [{"label": "Years covered", "value": 2022.0, "format": "number"}]
    issues = validate(_df_sales(), rep)
    assert any(i.code == "kpi_year_lookalike" and i.severity == "warn" for i in issues)


def test_kpi_total_2020_warns():
    rep = _good_bar_report()
    rep["key_metrics"] = [{"label": "Total", "value": 2020.0, "format": "number"}]
    issues = validate(_df_sales(), rep)
    assert "kpi_year_lookalike" in {i.code for i in issues}


def test_kpi_legit_count_not_a_year_passes():
    rep = _good_bar_report()
    rep["key_metrics"] = [{"label": "Total rows", "value": 1500.0, "format": "number"}]
    issues = validate(_df_sales(), rep)
    assert "kpi_year_lookalike" not in {i.code for i in issues}


def test_kpi_revenue_2020_not_flagged_label_mismatch():
    # 'revenue' isn't a year/count/total label, so 2020.0 is fine.
    rep = _good_bar_report()
    rep["key_metrics"] = [{"label": "Revenue", "value": 2020.0, "format": "currency"}]
    issues = validate(_df_sales(), rep)
    assert "kpi_year_lookalike" not in {i.code for i in issues}
```

Run (WAIT) — expected: the new tests fail.

**10b — implement: add `check_kpi_sanity` and wire it in.**

```python
_YEAR_LABEL_RE = re.compile(r"\b(year|years|count|total)\b", re.IGNORECASE)


def check_kpi_sanity(df: pd.DataFrame, report: dict) -> list[Issue]:
    issues: list[Issue] = []
    for m in (report.get("key_metrics") or []):
        label = str(m.get("label", ""))
        raw = m.get("value")
        try:
            val = float(raw)
        except (TypeError, ValueError):
            issues.append(Issue("fail", "kpi_non_finite",
                                f"KPI '{label}' has a non-numeric value ({raw!r})."))
            continue
        if not math.isfinite(val):
            issues.append(Issue("fail", "kpi_non_finite",
                                f"KPI '{label}' is non-finite ({val})."))
            continue
        # Year-look-alike: a year/count/total metric whose value IS a calendar year.
        if _YEAR_LABEL_RE.search(label) and float(val).is_integer() and 1900 <= val <= 2100:
            issues.append(Issue("warn", "kpi_year_lookalike",
                                f"KPI '{label}' = {int(val)} looks like a calendar year, "
                                f"not a {label.lower()} (e.g. the '2,022 years covered' slip)."))
    return issues
```

Wire into `validate`:

```python
    issues += check_kpi_sanity(df, report)
```

**Verify (WAIT):**

```
venv/bin/python -m pytest qa/tests/test_validators.py -q
```

Expected: all tests pass.

**Commit:** `qa: validators — KPI sanity (non-finite + year-look-alike)`

---

### Task 11 — Narrative + chart-count floor + axis-label checks, and the year-as-int regression (TDD)

**Files:** append to `qa/tests/test_validators.py`, then `qa/validators.py`

**11a — append failing tests (incl. the named regression):**

```python
# ---- narrative / floor / labels --------------------------------------------

def test_empty_summary_flagged():
    rep = _good_bar_report()
    rep["summary"] = "   "
    issues = validate(_df_sales(), rep)
    assert any(i.code == "narrative_missing" and i.severity == "fail" for i in issues)


def test_chart_count_below_floor_flagged():
    rep = _good_bar_report()   # only 1 chart; floor is MIN_CHARTS_FOR_NO_FALLBACK (3)
    issues = validate(_df_sales(), rep)
    assert any(i.code == "chart_count_below_floor" for i in issues)


def test_missing_axis_labels_warn():
    rep = _good_bar_report()
    rep["charts"][0]["spec"]["x_label"] = ""
    rep["charts"][0]["spec"]["y_label"] = ""
    issues = validate(_df_sales(), rep)
    assert any(i.code == "missing_axis_label" and i.severity == "warn" for i in issues)


# ---- the motivating regression ---------------------------------------------

def test_year_as_int_dataset_flags_if_line_regresses():
    """Build the year-as-int generator df and feed a DEGENERATE line report (what a
    broken _to_datetime would yield: every year collapsed to one period). The
    degenerate-chart check must fire. This is the harness's proof-of-catch."""
    from qa.generators import gen_year_as_int
    _name, df = gen_year_as_int()
    df.columns = [c.lower() for c in df.columns]
    degenerate = {
        "generated_at": "2026-06-04T00:00:00",
        "summary": "Revenue over time.",
        "data_quality": [],
        "key_metrics": [],
        "charts": [
            {"chart_id": "L", "caption": "trend", "spec": {
                "kind": "line", "title": "Revenue over time", "intent": "trend",
                # broken: every point lands on the SAME period label
                "x": ["1970", "1970", "1970", "1970", "1970"],
                "y": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
                "x_label": "Year (year)", "y_label": "Sum of revenue",
                "x_display_type": "date", "y_display_type": "currency",
                "source_columns": ["year", "revenue"], "data_point_count": 360,
            }},
            # pad to clear the floor so we isolate the degenerate-line signal
            {"chart_id": "B1", "caption": "c", "spec": {
                "kind": "bar", "title": "Region mix", "intent": "mix",
                "x": ["north", "south"], "y": [180, 180],
                "x_label": "region", "y_label": "Count",
                "x_display_type": "category", "y_display_type": "count",
                "source_columns": ["region"], "data_point_count": 360}},
            {"chart_id": "B2", "caption": "c", "spec": {
                "kind": "bar", "title": "Rev by region", "intent": "x",
                "x": ["north", "south"], "y": [50000.0, 49000.0],
                "x_label": "region", "y_label": "Sum of revenue",
                "x_display_type": "category", "y_display_type": "currency",
                "source_columns": ["revenue", "region"], "data_point_count": 360}},
        ],
        "layout": [{"chart_id": "L", "position": "main", "order": 0}],
        "metadata": {},
    }
    issues = validate(df, degenerate)
    assert any(i.code == "degenerate_chart" and i.chart_id == "L" for i in issues), \
        "the collapsed year-as-int line must be flagged degenerate"
```

Run (WAIT) — expected: the new tests fail (narrative/floor/label checks not yet implemented; the regression test should already PASS because the degenerate-line check from Task 8 catches it — that is intentional, it pins the behavior).

**11b — implement the remaining checks and wire them in:**

```python
def check_narrative(df: pd.DataFrame, report: dict) -> list[Issue]:
    summary = str(report.get("summary") or "").strip()
    if not summary:
        return [Issue("fail", "narrative_missing", "Report summary is empty.")]
    return []


def check_chart_count_floor(df: pd.DataFrame, report: dict) -> list[Issue]:
    n = len(_charts(report))
    if n < MIN_CHARTS_FOR_NO_FALLBACK:
        return [Issue("fail", "chart_count_below_floor",
                      f"Only {n} chart(s); the selection floor is "
                      f"{MIN_CHARTS_FOR_NO_FALLBACK} (fallback should guarantee it).")]
    return []


def check_axis_labels(df: pd.DataFrame, report: dict) -> list[Issue]:
    """x_label/y_label should be present. Some kinds (pie/treemap/box/heatmap)
    legitimately leave an axis blank, so a missing label is a 'warn', not a 'fail'."""
    issues: list[Issue] = []
    for chart in _charts(report):
        spec = _spec(chart)
        cid = chart.get("chart_id")
        if not str(spec.get("x_label") or "").strip():
            issues.append(Issue("warn", "missing_axis_label",
                                f"[{spec.get('kind')}] '{spec.get('title')}' has no x_label.", cid))
        if not str(spec.get("y_label") or "").strip():
            issues.append(Issue("warn", "missing_axis_label",
                                f"[{spec.get('kind')}] '{spec.get('title')}' has no y_label.", cid))
    return issues
```

Wire into `validate` (final form of the aggregator):

```python
    issues += check_chart_data_consistency(df, report)
    issues += check_kpi_sanity(df, report)
    issues += check_narrative(df, report)
    issues += check_chart_count_floor(df, report)
    issues += check_axis_labels(df, report)
    return issues
```

**Verify (WAIT):**

```
venv/bin/python -m pytest qa/tests/test_validators.py -q
```

Expected: all validator tests pass, including `test_year_as_int_dataset_flags_if_line_regresses`.

**Commit:** `qa: validators — narrative + chart-count floor + axis labels + year regression test`

---

## Phase 4 — LLM judge + results report

### Task 12 — Haiku judge (light test + structured tool call)

**Files:** `qa/tests/test_judge.py` (failing test first), then `qa/judge.py`

The judge mirrors how `report_generator.generate_narrative` forces a single tool call (`tool_choice={"type":"tool","name":...}`, `cache_static=True`), but with its own schema. Task 12's test uses FakeClaude (offline); the real-Haiku path runs in the baseline (Task 21).

**12a — failing test** `qa/tests/test_judge.py`:

```python
"""judge_report parses a forced-tool-call response into a JudgeVerdict (offline)."""
import types

from qa.judge import judge_report, JudgeVerdict
from tests.helpers.fake_claude import FakeClaude, tool_use


def _report():
    return {
        "summary": "Revenue rises over the year.",
        "charts": [
            {"chart_id": "c1", "caption": "trend", "spec": {
                "kind": "line", "title": "Revenue over time", "intent": "trend",
                "x": ["2019", "2020", "2021"], "y": [1.0, 2.0, 3.0],
                "x_label": "Year", "y_label": "Sum of revenue"}},
        ],
    }


def test_judge_parses_structured_verdict(monkeypatch):
    fake = FakeClaude([{"tool_calls": [tool_use("submit_judgement", {
        "charts": [{"chart_id": "c1", "makes_sense": True, "issue": None, "severity": "none"}],
        "narrative_matches": True,
    })]}])
    monkeypatch.setattr(
        "qa.judge._build_claude",
        lambda: types.SimpleNamespace(messages_create=fake),
    )
    verdict = judge_report("Rows: 3\nColumns ...", _report())
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.narrative_matches is True
    assert len(verdict.charts) == 1
    assert verdict.charts[0].chart_id == "c1"
    assert verdict.charts[0].makes_sense is True
    assert verdict.any_chart_fails is False


def test_judge_flags_nonsense_chart(monkeypatch):
    fake = FakeClaude([{"tool_calls": [tool_use("submit_judgement", {
        "charts": [{"chart_id": "c1", "makes_sense": False,
                    "issue": "line collapses to one point", "severity": "fail"}],
        "narrative_matches": False,
    })]}])
    monkeypatch.setattr(
        "qa.judge._build_claude",
        lambda: types.SimpleNamespace(messages_create=fake),
    )
    verdict = judge_report("profile", _report())
    assert verdict.any_chart_fails is True
    assert verdict.charts[0].severity == "fail"
```

Run (WAIT) — expected import failure (`qa/judge.py` missing).

**12b — implement** `qa/judge.py`:

```python
"""LLM judge: ask Haiku whether each chart makes sense for the data and whether
the narrative matches the charts. Uses the REAL Anthropic client (key from .env)
and a single forced tool call, mirroring report_generator.generate_narrative.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

_API_DIR = Path(__file__).resolve().parent.parent / "src" / "api"
if str(_API_DIR) not in sys.path:
    sys.path.insert(0, str(_API_DIR))

load_dotenv()

from claude_client import ClaudeClient            # noqa: E402
from llm_config import MODEL_NARRATIVE            # noqa: E402  (cheap Haiku alias)


JUDGE_SYSTEM = (
    "You are a data-visualization QA reviewer. You are given a dataset PROFILE "
    "and a generated REPORT (chart specs + narrative). For EACH chart, decide if "
    "it makes sense for this data: is it degenerate (a line with one point, an "
    "all-zero/identical series), misleading (an identifier charted as a measure, a "
    "year treated as a quantity), or redundant? Then decide whether the narrative "
    "summary is supported by the charts (no claims a chart doesn't back). Be strict "
    "but fair; only mark makes_sense=false when there's a real problem. Always call "
    "submit_judgement exactly once."
)

JUDGE_TOOL: dict = {
    "name": "submit_judgement",
    "description": "Submit the QA judgement for this report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "charts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chart_id": {"type": "string"},
                        "makes_sense": {"type": "boolean"},
                        "issue": {"type": ["string", "null"],
                                  "description": "Short problem description, or null."},
                        "severity": {"type": "string",
                                     "enum": ["none", "warn", "fail"]},
                    },
                    "required": ["chart_id", "makes_sense", "severity"],
                },
            },
            "narrative_matches": {"type": "boolean"},
        },
        "required": ["charts", "narrative_matches"],
    },
}


@dataclass
class ChartVerdict:
    chart_id: str
    makes_sense: bool
    severity: str            # 'none' | 'warn' | 'fail'
    issue: Optional[str] = None


@dataclass
class JudgeVerdict:
    charts: list[ChartVerdict] = field(default_factory=list)
    narrative_matches: bool = True

    @property
    def any_chart_fails(self) -> bool:
        return any((not c.makes_sense) or c.severity == "fail" for c in self.charts)


def _build_claude() -> ClaudeClient:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (expected in .env)")
    return ClaudeClient(api_key=api_key)


def _compact_report(report: dict) -> str:
    """A token-lean view of the report for the judge: one line per chart + summary."""
    lines = [f"SUMMARY: {report.get('summary', '')}", "", "CHARTS:"]
    for chart in (report.get("charts") or []):
        spec = chart.get("spec") or {}
        x = spec.get("x") or []
        y = spec.get("y") or []
        sample = ""
        if x and y:
            sample = f" x[:5]={list(x)[:5]} y[:5]={list(y)[:5]}"
        elif spec.get("series"):
            sample = f" series={len(spec['series'])} e.g. {str(spec['series'][0])[:120]}"
        elif spec.get("nodes"):
            sample = f" nodes={len(spec['nodes'])}"
        lines.append(
            f"- id={chart.get('chart_id')} [{spec.get('kind')}] {spec.get('title')} "
            f"| x_label={spec.get('x_label')} y_label={spec.get('y_label')} "
            f"y_display={spec.get('y_display_type')}{sample}")
    return "\n".join(lines)


def judge_report(profile_text: str, report: dict) -> JudgeVerdict:
    claude = _build_claude()
    user = (f"DATASET PROFILE:\n{profile_text}\n\n"
            f"GENERATED REPORT:\n{_compact_report(report)}\n\n"
            f"Judge every chart and the narrative. Call submit_judgement once.")
    response = claude.messages_create(
        model=MODEL_NARRATIVE,
        max_tokens=1500,
        system=JUDGE_SYSTEM,
        tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "submit_judgement"},
        messages=[{"role": "user", "content": user}],
        cache_static=True,
    )
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_judgement":
            data = block.input
            charts = [
                ChartVerdict(
                    chart_id=str(c.get("chart_id", "")),
                    makes_sense=bool(c.get("makes_sense", True)),
                    severity=str(c.get("severity", "none")),
                    issue=c.get("issue"),
                )
                for c in (data.get("charts") or [])
            ]
            return JudgeVerdict(
                charts=charts,
                narrative_matches=bool(data.get("narrative_matches", True)),
            )
    # No tool call -> treat as inconclusive-pass (deterministic gate still applies).
    return JudgeVerdict(charts=[], narrative_matches=True)
```

**Verify (WAIT):**

```
venv/bin/python -m pytest qa/tests/test_judge.py -q
```

Expected: both tests pass.

**Commit:** `qa: judge — Haiku structured verdict (per-chart + narrative)`

---

### Task 13 — Results report writer (TDD)

**Files:** `qa/tests/test_report_writer.py` (failing test first), then `qa/report_writer.py`

**Verdict rule (spec):** a dataset is **FAIL** if it has any deterministic `fail` issue OR the judge says any chart doesn't make sense (`any_chart_fails`); **WARN** if it has any `warn` issue (or `narrative_matches` is false) but no FAIL; else **PASS**.

**13a — failing test** `qa/tests/test_report_writer.py`:

```python
"""report_writer rolls up PASS/WARN/FAIL and writes report.md + summary.json."""
import json

from qa.report_writer import classify, write_report


def _payload(name, det_issues=None, judge=None, error=None, report=True):
    return {
        "name": name,
        "error": error,
        "elapsed_ms": 10,
        "rows_analyzed": 5, "cols_analyzed": 2,
        "was_sampled": False, "original_rows": 5,
        "report": {"summary": "s", "charts": []} if report else None,
        "deterministic_issues": det_issues or [],
        "judge": judge,
    }


def test_classify_fail_on_deterministic_fail():
    p = _payload("d", det_issues=[{"severity": "fail", "code": "x", "message": "m", "chart_id": None}])
    assert classify(p) == "FAIL"


def test_classify_fail_on_judge_nonsense():
    judge = {"charts": [{"chart_id": "c1", "makes_sense": False, "severity": "fail", "issue": "bad"}],
             "narrative_matches": True}
    assert classify(_payload("d", judge=judge)) == "FAIL"


def test_classify_warn_on_warn_only():
    p = _payload("d", det_issues=[{"severity": "warn", "code": "x", "message": "m", "chart_id": None}])
    assert classify(p) == "WARN"


def test_classify_warn_on_narrative_mismatch():
    judge = {"charts": [], "narrative_matches": False}
    assert classify(_payload("d", judge=judge)) == "WARN"


def test_classify_pass_when_clean():
    assert classify(_payload("d")) == "PASS"


def test_write_report_emits_files_and_table(tmp_path):
    payloads = [
        _payload("good"),
        _payload("bad", det_issues=[{"severity": "fail", "code": "degenerate_chart",
                                     "message": "boom", "chart_id": "c1"}]),
    ]
    write_report(tmp_path, payloads)
    md = (tmp_path / "report.md").read_text()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert "| good | PASS |" in md
    assert "| bad | FAIL |" in md
    assert summary["totals"]["FAIL"] == 1
    assert summary["totals"]["PASS"] == 1
    assert summary["datasets"]["bad"]["status"] == "FAIL"
```

Run (WAIT) — expected import failure (`qa/report_writer.py` missing).

**13b — implement** `qa/report_writer.py`:

```python
"""Render a QA run into report.md (scannable) + summary.json (machine-readable)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _det_issues(payload: dict) -> list[dict]:
    return payload.get("deterministic_issues") or []


def _judge(payload: dict) -> dict | None:
    return payload.get("judge")


def classify(payload: dict) -> str:
    """PASS / WARN / FAIL per the spec's rule."""
    issues = _det_issues(payload)
    if any(i.get("severity") == "fail" for i in issues):
        return "FAIL"
    judge = _judge(payload)
    if judge:
        charts = judge.get("charts") or []
        if any((not c.get("makes_sense", True)) or c.get("severity") == "fail" for c in charts):
            return "FAIL"
    if any(i.get("severity") == "warn" for i in issues):
        return "WARN"
    if judge:
        if any(c.get("severity") == "warn" for c in (judge.get("charts") or [])):
            return "WARN"
        if judge.get("narrative_matches") is False:
            return "WARN"
    return "WARN" if (payload.get("report") is None and not issues) else "PASS"


def _counts(payload: dict) -> tuple[int, int]:
    issues = _det_issues(payload)
    fails = sum(1 for i in issues if i.get("severity") == "fail")
    warns = sum(1 for i in issues if i.get("severity") == "warn")
    judge = _judge(payload)
    if judge:
        for c in (judge.get("charts") or []):
            if (not c.get("makes_sense", True)) or c.get("severity") == "fail":
                fails += 1
            elif c.get("severity") == "warn":
                warns += 1
    return fails, warns


def write_report(run_dir: str | Path, per_dataset_results: list[dict]) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    statuses = {p["name"]: classify(p) for p in per_dataset_results}
    totals = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for s in statuses.values():
        totals[s] += 1

    # ---- report.md ----
    lines = [f"# ChartSage QA run — {run_dir.name}", "",
             f"PASS {totals['PASS']} · WARN {totals['WARN']} · FAIL {totals['FAIL']} "
             f"(of {len(per_dataset_results)} datasets)", "",
             "| dataset | status | fails | warns | charts | ms |",
             "|---|---|---|---|---|---|"]
    for p in per_dataset_results:
        fails, warns = _counts(p)
        n_charts = len((p.get("report") or {}).get("charts") or []) if p.get("report") else 0
        lines.append(f"| {p['name']} | {statuses[p['name']]} | {fails} | {warns} | "
                     f"{n_charts} | {p.get('elapsed_ms', 0)} |")
    lines.append("")

    # Per-dataset detail (FAIL/WARN first).
    order = sorted(per_dataset_results,
                   key=lambda p: {"FAIL": 0, "WARN": 1, "PASS": 2}[statuses[p["name"]]])
    for p in order:
        name = p["name"]
        lines.append(f"## {name} — {statuses[name]}")
        if p.get("error"):
            first = p["error"].splitlines()[0] if p["error"] else ""
            lines.append(f"- generation error: `{first}`")
        for i in _det_issues(p):
            tag = f" (chart {i['chart_id']})" if i.get("chart_id") else ""
            lines.append(f"- [{i['severity']}] {i['code']}: {i['message']}{tag}")
        judge = _judge(p)
        if judge:
            if judge.get("narrative_matches") is False:
                lines.append("- [judge] narrative does not match the charts")
            for c in (judge.get("charts") or []):
                if (not c.get("makes_sense", True)) or c.get("severity") in ("warn", "fail"):
                    lines.append(f"- [judge:{c.get('severity')}] chart {c.get('chart_id')}: "
                                 f"{c.get('issue')}")
        lines.append("")
    (run_dir / "report.md").write_text("\n".join(lines))

    # ---- summary.json ----
    summary: dict[str, Any] = {
        "run": run_dir.name,
        "totals": totals,
        "datasets": {
            p["name"]: {
                "status": statuses[p["name"]],
                "fails": _counts(p)[0],
                "warns": _counts(p)[1],
                "elapsed_ms": p.get("elapsed_ms", 0),
                "was_sampled": p.get("was_sampled", False),
            }
            for p in per_dataset_results
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
```

**Verify (WAIT):**

```
venv/bin/python -m pytest qa/tests/test_report_writer.py -q
```

Expected: all tests pass.

**Commit:** `qa: report_writer — PASS/WARN/FAIL roll-up + report.md/summary.json`

---

## Phase 5 — Wire-up, baseline, self-test

### Task 14 — `make qa` target + README

**Files:** `Makefile`, `qa/README.md`

**Makefile** — add `qa` to `.PHONY` and a target. Change line 1 to:

```
.PHONY: dev test test-e2e test-pdf smoke clean qa
```

Add at the end:

```
# QA / Eval Harness — run the corpus through the real pipeline + validators + judge.
# Flags pass through, e.g.: make qa ARGS="--only synthetic --no-judge --limit 3"
qa:
	venv/bin/python qa/run_eval.py $(ARGS)
```

**qa/README.md:**

```markdown
# ChartSage QA / Eval Harness

On-demand regression tooling. Runs a corpus of edge-case + real datasets through
the **real** generation pipeline (the same code `/generate-report` uses), then
validates every output deterministically and with a Haiku LLM judge. **Not deployed.**

## Run

```
make qa                              # full corpus (synthetic + qa/datasets/*.csv)
make qa ARGS="--only synthetic"      # only the synthetic edge cases
make qa ARGS="--only real"           # only dropped-in CSVs
make qa ARGS="--only year_as_int"    # a single dataset by name
make qa ARGS="--no-judge"            # deterministic checks only (free, no Haiku)
make qa ARGS="--limit 5"             # first 5 datasets
```

Requires `ANTHROPIC_API_KEY` in `.env` (already present). The runner uses real
Haiku for chart selection, narrative, and the judge.

## Add a real dataset

Drop any `*.csv` into `qa/datasets/`. It's auto-discovered (name = filename stem).

## Output

Each run writes `qa/results/<timestamp>/`:
- `report.md` — top table (dataset → PASS/WARN/FAIL + issue counts), then per-dataset detail.
- `summary.json` — machine-readable roll-up.
- `<dataset>.json` — full per-dataset record (report, deterministic issues, judge verdict).

`qa/results/*` is gitignored.

## Verdict rule

- **FAIL** — any deterministic `fail` issue, or the judge marks any chart as not making sense.
- **WARN** — any `warn` issue (or the judge says the narrative doesn't match), no FAIL.
- **PASS** — clean.

## What it checks

Deterministic (`qa/validators.py`): generation crash; degenerate charts (empty x/y,
<2-distinct-x lines, all-NaN/zero/identical y); chart-data consistency (recomputed
count/sum groupby vs the chart's y); KPI sanity (non-finite; year-look-alike metrics);
empty narrative; chart count below the selection floor; missing axis labels.

Judge (`qa/judge.py`): does each chart make sense for the data, is any chart
misleading/degenerate/redundant, does the narrative match the charts.
```

**Verify (WAIT) — the deterministic-only path end-to-end on synthetic, capped (this calls real Haiku for selection+narrative; allow a few minutes):**

```
make qa ARGS="--only synthetic --no-judge --limit 2"
```

Expected: prints two `[qa] running ...` lines and `[qa] done -> qa/results/<ts>/report.md`; that `report.md` and `summary.json` exist.

**Commit:** `qa: make qa target + README`

---

### Task 15 — Full test-suite green gate (qa/ included)

**File:** none (verification gate).

Run the entire suite plus the qa package tests to confirm nothing regressed and everything is offline-safe and hang-proof (WAIT; longer run — do not fire other commands into it):

```
venv/bin/python -m pytest -q --timeout=30
venv/bin/python -m pytest qa/tests -q --timeout=30
```

Expected: both green, zero `Timeout` failures, e2e tests skipped (no `RUN_E2E`). If `qa/tests` isn't collected by the first command (testpaths is `tests`), the second explicit run covers it.

**Commit:** none (gate only). If anything fails, fix under systematic-debugging before proceeding.

---

### Task 16 — Baseline full run + triage

**Files:** `docs/FUTURE-IMPROVEMENTS.md` (append findings only).

Run the **full** corpus with the judge once (WAIT; this is the long one — synthetic + any real CSVs through real Haiku selection+narrative+judge; can take many minutes):

```
make qa
```

Then read `qa/results/<ts>/report.md`. Triage each FAIL/WARN:

- **Real harness bug** (validator false positive, pipeline drift, judge prompt issue): fix it inline in a focused commit, re-run the affected dataset with `make qa ARGS="--only <name>"`, confirm it clears.
- **Real product bug surfaced by a synthetic case** (e.g. currency-strings charted as a measure, id-like ints charted): if it's a small, safe fix in the production code, the controller decides whether to fix here or defer. Otherwise **log it** to `docs/FUTURE-IMPROVEMENTS.md` under a new dated heading, e.g.:

```markdown
## 2026-06-04 — QA harness baseline findings

- [<dataset>] <PASS/WARN/FAIL>: <one-line description + the chart_id / code>. <decision: fixed inline / deferred + why>.
```

Do not silence a check just to make the table green — only adjust a check if it's genuinely a false positive (with a new/updated unit test in `qa/tests/test_validators.py` proving the corrected behavior, plus a commit).

**Commit:** `qa: baseline run triage — <short summary>` (and any inline fix commits).

---

### Task 17 — Self-test: prove the harness catches the year bug

**File:** `src/api/chart_executor.py` (temporary revert, then restore — net-zero diff).

This proves the harness FAILs the year-as-int dataset when the year-handling regresses. The controller's worktree (off `origin/main`) contains the year-aware `_to_datetime`; this task temporarily reverts it to the naive form and confirms the FAIL, then restores it.

**17a — record the current `_to_datetime` body** (so it can be restored verbatim). Read `src/api/chart_executor.py` around the `_to_datetime` definition (search for `def _to_datetime`). Copy its exact current lines.

**17b — temporarily revert** `_to_datetime` to the naive two-line form (this is the pre-fix behavior that collapses bare integer years to epoch-nanosecond timestamps):

```python
def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")
```

> If the worktree's `_to_datetime` is **already** this naive form (i.e. the year fix isn't in this branch yet), do NOT proceed with the revert — instead note in the run summary that the self-test can't demonstrate a regression because the fix isn't present, and skip to 17d after confirming the harness still flags the constructed degenerate report via `test_year_as_int_dataset_flags_if_line_regresses` (already green from Task 11). Flag this to the controller.

**17c — run the year dataset and confirm FAIL** (WAIT; real Haiku):

```
make qa ARGS="--only year_as_int"
```

Read the resulting `qa/results/<ts>/report.md`. Expected: the `year_as_int` row is **FAIL** with a `degenerate_chart` issue on the line chart (and/or the judge flagging the collapsed line). If it does NOT fail, the harness isn't catching the bug class — stop and debug (the line chart may not have been selected; inspect `year_as_int.json` to confirm a line chart was produced and that its x collapsed).

**17d — restore** `_to_datetime` to the exact body recorded in 17a. Confirm a clean tree for that file:

```
git -C . diff --stat src/api/chart_executor.py
```

Expected: empty output (no net change). Re-run to confirm the fix makes it PASS/WARN (not FAIL on degeneracy) (WAIT):

```
make qa ARGS="--only year_as_int"
```

Expected: the `year_as_int` line chart is no longer flagged `degenerate_chart` (the year-aware parse yields ≥5 distinct periods).

**Commit:** none if the tree is clean (the revert/restore is net-zero). If 17a/17d leaves any incidental change, revert it so `git diff` is empty. Record the self-test outcome in the final summary.

---

### Task 18 — Final verification + hand-off to controller

**File:** none (final gate + branch hand-off note).

Final gate (WAIT for each; don't overlap):

```
venv/bin/python -m pytest -q --timeout=30
venv/bin/python -m pytest qa/tests -q --timeout=30
git -C . status --short
```

Expected: both suites green; `git status` shows only intended, already-committed work (working tree clean). Confirm `qa/results/` run folders are untracked/ignored (only `.gitkeep` tracked):

```
git -C . status --short qa/results
```

Expected: empty (runs ignored).

**Hand-off:** This is **dev tooling — there is NO production deploy** (no Cloud Run, no Vercel, no migration). The only remaining step is the **controller** merging this branch to `main` (user-authorized). State in the final summary:
- the self-test result (harness FAILs year-as-int when `_to_datetime` regresses, PASSes when fixed),
- the baseline `report.md` roll-up (PASS/WARN/FAIL counts),
- any items logged to `docs/FUTURE-IMPROVEMENTS.md`,
- that subagents performed no branch operations; the controller owns the merge.

**Commit:** none (gate only).

---

## Verification summary (maps to the spec)

- **Test-suite health:** `pytest-timeout` + `timeout=60`/`timeout_method=thread` in `pytest.ini`; the network straggler mocked; `venv/bin/python -m pytest -q --timeout=30` green with zero timeouts (Tasks 1–3, 15, 18).
- **Generators:** 16 generators, each shape-tested (Task 5).
- **Pipeline fidelity:** `run_report` imports the real `sample_for_analysis`/`MODEL_*`/`ReportGenerator` and reproduces the endpoint assembly; light test pins the prod report shape (Task 6).
- **Deterministic validators:** every check TDD'd with a good + bad report, incl. `test_year_as_int_dataset_flags_if_line_regresses` (Tasks 8–11).
- **Judge + report:** structured Haiku verdict (Task 12); `report.md` + `summary.json` with the PASS/WARN/FAIL rule (Task 13).
- **Wire-up + baseline + self-test:** `make qa` with `--only/--no-judge/--limit`; full baseline run triaged; the revert-the-fix self-test proves the catch (Tasks 14, 16, 17).
- **Scope:** dev tooling only — controller merges to `main`; no prod deploy (Task 18).

---

### Critical Files for Implementation
- /Users/chrissilver/Documents/ChartSage/src/api/main.py (the exact `/generate-report` sequence `qa/pipeline.run_report` must mirror: `_load_dataframe`/lower-case → `sample_for_analysis` → guards → `profile_dataframe` → `ReportGenerator(...)` → `build_report` → sampling-note → `model_dump`)
- /Users/chrissilver/Documents/ChartSage/src/api/report_generator.py (`ReportGenerator.__init__`/`build_report` assembly + `MIN_CHARTS_FOR_NO_FALLBACK` selection floor the validator imports)
- /Users/chrissilver/Documents/ChartSage/src/api/chart_executor.py (executors + `_to_datetime` at the `def _to_datetime` definition — the Task 17 self-test revert target)
- /Users/chrissilver/Documents/ChartSage/src/api/schemas.py (`ChartSpec`/`KeyMetric`/`Report` field names the validators + judge read)
- /Users/chrissilver/Documents/ChartSage/src/api/claude_client.py + /Users/chrissilver/Documents/ChartSage/src/api/llm_config.py (real Haiku client construction + `MODEL_SELECTION`/`MODEL_NARRATIVE` aliases the pipeline and judge use)
