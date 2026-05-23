# ChartSage v2 Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild ChartSage's chart pipeline on Anthropic tool use, replace the dashboard with a narrative report, and deliver a working CSV → report flow that costs ~$0.01 on Haiku 4.5 and never silently produces wrong charts.

**Architecture:** FastAPI backend with a typed two-pass flow — pass #1 uses parallel tool calls for chart selection, pass #2 uses a single forced tool call to produce a structured narrative. Each chart "kind" is a tool with a strict schema; executors validate semantics in Python and surface errors back to Claude for one retry round before falling back to heuristic chart picking. Frontend renders the structured report as summary + chart grid + data-quality callout.

**Tech Stack:** Python 3.11+, FastAPI, pandas, Pydantic v2, anthropic SDK, pytest, Next.js 14, React 18, ECharts, Redis. See [the design spec](../specs/2026-05-23-chartsage-rebuild-design.md) (commit `872256b`) for full context.

---

## File Structure

### Backend (new / modified)

```
src/api/
├── main.py                       # FastAPI app, only /generate-report + /report/{id}
├── schemas.py                    # All Pydantic models (ChartSpec, Report, etc.)
├── llm_config.py                 # Model alias resolution (env-driven)
├── claude_client.py              # anthropic SDK wrapper: retries, caching, cost log
├── profile.py                    # profile_dataframe(df) → DataProfile
├── chart_tools.py                # 8 Anthropic tool definitions
├── chart_executor.py             # 8 executor functions + TOOL_EXECUTORS dispatch
├── report_generator.py           # Two-pass orchestrator
├── fallback.py                   # Heuristic chart picker (used when pass #1 fails)
├── data_processing_utils.py      # compute_group_count + new histogram binning
└── prompts/
    ├── selection_system.txt
    └── narrative_system.txt
```

### Frontend (new)

```
src/app/
├── page.tsx                                  # Upload, navigates to /report/[id]
├── lib/format.ts                             # Number / currency / count formatters
└── report/[id]/
    ├── page.tsx                              # Fetch + render report
    ├── ReportSummary.tsx
    ├── DataQualityCallout.tsx
    ├── ChartCard.tsx                         # Dispatches by spec.kind
    └── charts/
        ├── BarChart.tsx
        ├── HistogramChart.tsx
        ├── ScatterChart.tsx
        ├── LineChart.tsx
        ├── PieChart.tsx
        ├── BoxPlot.tsx
        └── Heatmap.tsx
```

### Tests

```
tests/
├── conftest.py                               # Shared DataFrame fixtures
├── unit/
│   ├── test_profile.py
│   ├── test_executors_frequency.py
│   ├── test_executors_aggregation.py
│   ├── test_executors_histogram.py
│   ├── test_executors_scatter.py
│   ├── test_executors_line.py
│   ├── test_executors_pie.py
│   ├── test_executors_box.py
│   └── test_executors_heatmap.py
├── integration/
│   ├── test_pipeline_happy.py
│   ├── test_pipeline_retry.py
│   ├── test_pipeline_fallback.py
│   └── test_api_errors.py
├── e2e/
│   ├── test_real_claude_smoke.py
│   └── fixtures/
│       ├── activities.csv
│       ├── sales.csv
│       ├── signups.csv
│       ├── survey.csv
│       └── degenerate.csv
└── helpers/
    ├── fake_claude.py
    └── builders.py
```

### Top-level

```
Makefile                          # test / test-e2e / smoke / dev
requirements.txt                  # +pytest, +httpx (for TestClient)
.env.example                      # Stripe/NextAuth gone; CLAUDE_MODEL added
README.md                         # Honest description
ChartSage.md                      # Mirrored against reality
```

### Deletions (executed in Phase 8)

`src/api/bar_chart_processor.py`, `src/api/chart_processing.py`, `src/api/insight_prompt.txt`, `src/api/bar_chart_prompt.txt`, `src/api/log_viewer.py`, `src/api/field_type_utils.py`, `src/api/derived_fields.py`, `src/api/temp/`, `src/api/logs/*.log` (kept folder), `src/app/visualizations/`, `src/app/dashboard/`, `src/app/workers/`, `BAR_CHART_SYSTEM.md`, `REFACTORING_SUMMARY.md`, `src/api/CHART_GENERATION_FIXES.md`, `src/api/LOGGING_SYSTEM.md`, `cursor_rule_prompt_json_formatting.mdc`.

---

## Phase 0 — Project setup

### Task 1: pytest scaffolding

**Files:**
- Create: `pytest.ini`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/helpers/__init__.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add pytest + httpx to requirements**

Edit `requirements.txt` to add at the bottom:

```
pytest==8.0.0
pytest-asyncio==0.23.5
httpx==0.27.0
```

- [ ] **Step 2: Create pytest.ini**

Write `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -ra -q --strict-markers
markers =
    e2e: end-to-end tests that hit real Claude (opt-in via RUN_E2E=true)
pythonpath = src/api
```

- [ ] **Step 3: Create empty __init__.py files**

Create empty files at:
- `tests/__init__.py`
- `tests/unit/__init__.py`
- `tests/integration/__init__.py`
- `tests/helpers/__init__.py`

- [ ] **Step 4: Install and verify**

Run:

```bash
pip install -r requirements.txt
pytest --collect-only
```

Expected: "no tests ran" with no errors.

- [ ] **Step 5: Commit**

```bash
git add pytest.ini tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py tests/helpers/__init__.py requirements.txt
git commit -m "chore: pytest scaffolding"
```

---

### Task 2: Shared test fixtures

**Files:**
- Create: `tests/conftest.py`
- Create: `tests/helpers/builders.py`

- [ ] **Step 1: Write the DataFrame factory in builders.py**

Write `tests/helpers/builders.py`:

```python
"""Small DataFrame factories for tests."""
from datetime import datetime, timedelta
import pandas as pd


def activities_df() -> pd.DataFrame:
    """The canonical example: categorical type + numeric duration + date + identifier."""
    return pd.DataFrame({
        "activity_id": list(range(1, 21)),
        "patient_id": [1001, 1002, 1003, 1001, 1002, 1003, 1004, 1005,
                       1001, 1006, 1007, 1008, 1009, 1010, 1011, 1012,
                       1013, 1014, 1015, 1016],
        "activity_type": ["consultation", "consultation", "lab_test", "intro_call",
                          "consultation", "lab_test", "lab_test", "intro_call",
                          "consultation", "lab_test", "intro_call", "consultation",
                          "consultation", "lab_test", "intro_call", "consultation",
                          "lab_test", "intro_call", "consultation", "lab_test"],
        "activity_date": [datetime(2024, 1, 1) + timedelta(days=i * 3) for i in range(20)],
        "duration_minutes": [30.0, 45.0, 15.0, 20.0, 60.0, 25.0, 30.0, 15.0,
                             40.0, 20.0, 25.0, 50.0, 35.0, 30.0, 18.0, 55.0,
                             22.0, 19.0, 42.0, 28.0],
    })


def sales_df() -> pd.DataFrame:
    """Revenue / region / date — classic BI shape."""
    return pd.DataFrame({
        "order_id": list(range(1, 16)),
        "region": ["north", "south", "east", "west", "north",
                   "south", "east", "west", "north", "south",
                   "east", "west", "north", "south", "east"],
        "revenue": [1200.0, 850.0, 2100.0, 1500.0, 1800.0,
                    900.0, 2400.0, 1750.0, 1300.0, 950.0,
                    2200.0, 1650.0, 1400.0, 1000.0, 2300.0],
        "order_date": pd.date_range("2024-01-01", periods=15, freq="3D"),
    })


def degenerate_df() -> pd.DataFrame:
    """Single column, mostly null — stress test."""
    return pd.DataFrame({"x": [1.0, None, None, 2.0, None, None, None, 3.0]})


def negative_duration_df() -> pd.DataFrame:
    """Triggers the 'negative values in duration column' anomaly."""
    return pd.DataFrame({
        "activity_type": ["a", "b", "c", "a", "b"],
        "duration_minutes": [-30.0, 45.0, 60.0, 20.0, 30.0],
    })
```

- [ ] **Step 2: Write conftest.py with pytest fixtures**

Write `tests/conftest.py`:

```python
"""Shared pytest fixtures."""
import pytest
from tests.helpers.builders import (
    activities_df,
    sales_df,
    degenerate_df,
    negative_duration_df,
)


@pytest.fixture
def activities():
    return activities_df()


@pytest.fixture
def sales():
    return sales_df()


@pytest.fixture
def degenerate():
    return degenerate_df()


@pytest.fixture
def negative_duration():
    return negative_duration_df()
```

- [ ] **Step 3: Verify fixtures load**

Add a smoke test temporarily — write `tests/unit/test_fixtures_smoke.py`:

```python
def test_activities_loads(activities):
    assert len(activities) == 20
    assert "activity_type" in activities.columns


def test_sales_loads(sales):
    assert len(sales) == 15
    assert sales["revenue"].sum() > 0
```

Run:

```bash
pytest tests/unit/test_fixtures_smoke.py -v
```

Expected: 2 passed.

- [ ] **Step 4: Delete the smoke file (fixtures will be re-tested via real unit tests)**

```bash
rm tests/unit/test_fixtures_smoke.py
```

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/helpers/builders.py
git commit -m "test: shared DataFrame fixtures"
```

---

## Phase 1 — Schemas, config, and Claude client

### Task 3: Pydantic schemas

**Files:**
- Create: `src/api/schemas.py`
- Create: `tests/unit/test_schemas.py`

- [ ] **Step 1: Write the failing test**

Write `tests/unit/test_schemas.py`:

```python
import pytest
from pydantic import ValidationError
from schemas import ChartSpec, ToolError, ColumnInfo, DataProfile, Report, ReportNarrative


def test_chart_spec_minimum_required():
    spec = ChartSpec(
        kind="bar",
        title="t",
        intent="i",
        x=["a", "b"],
        y=[1, 2],
        x_label="X",
        y_label="Y",
        x_display_type="category",
        y_display_type="count",
        source_columns=["col1"],
        data_point_count=2,
    )
    assert spec.kind == "bar"


def test_chart_spec_rejects_invalid_kind():
    with pytest.raises(ValidationError):
        ChartSpec(
            kind="banana",  # not in literal
            title="t", intent="i",
            x=["a"], y=[1],
            x_label="X", y_label="Y",
            x_display_type="category", y_display_type="count",
            source_columns=["col1"], data_point_count=1,
        )


def test_tool_error_holds_reason():
    err = ToolError(reason="'revenue' is not a column")
    assert "revenue" in err.reason


def test_column_info_categorical():
    col = ColumnInfo(
        name="region", dtype="object", role="categorical",
        cardinality=4, null_count=0,
        top_values=[("north", 5), ("south", 4)],
    )
    assert col.role == "categorical"


def test_data_profile_basic():
    profile = DataProfile(
        row_count=100,
        columns=[ColumnInfo(name="x", dtype="int64", role="numeric",
                            cardinality=50, null_count=0, min=0, max=100)],
        correlations={},
        anomalies=[],
    )
    assert profile.row_count == 100
    text = profile.to_text()
    assert "x" in text


def test_report_round_trips_json():
    r = Report(
        generated_at="2026-05-23T12:00:00",
        summary="...",
        data_quality=[],
        charts=[],
        metadata={"model": "haiku-4-5"},
    )
    payload = r.model_dump_json()
    r2 = Report.model_validate_json(payload)
    assert r2.summary == "..."
```

- [ ] **Step 2: Run test, expect import error**

```bash
pytest tests/unit/test_schemas.py -v
```

Expected: collection errors (ModuleNotFoundError: No module named 'schemas').

- [ ] **Step 3: Implement schemas.py**

Write `src/api/schemas.py`:

```python
"""Pydantic models shared across the backend."""
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


ChartKind = Literal["bar", "histogram", "scatter", "line", "pie", "box", "heatmap"]
ColumnRole = Literal["categorical", "numeric", "date", "identifier", "unusable"]
XDisplayType = Literal["category", "number", "date", "text"]
YDisplayType = Literal["count", "currency", "percentage", "number"]


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    role: ColumnRole
    cardinality: int
    null_count: int
    top_values: Optional[list[tuple[Any, int]]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    unusable_reason: Optional[str] = None


class DataProfile(BaseModel):
    row_count: int
    columns: list[ColumnInfo]
    correlations: dict[str, float] = Field(default_factory=dict)  # key = "col1||col2"
    anomalies: list[str] = Field(default_factory=list)

    def to_text(self) -> str:
        """Compact text form for sending to Claude."""
        lines = [f"Rows: {self.row_count}", f"Columns ({len(self.columns)}):"]
        for c in self.columns:
            parts = [f"- {c.name}: role={c.role}, dtype={c.dtype}, cardinality={c.cardinality}, nulls={c.null_count}"]
            if c.role == "numeric":
                parts.append(f"  min={c.min}, max={c.max}, mean={c.mean}, median={c.median}, std={c.std}")
            elif c.role == "categorical" and c.top_values:
                top = ", ".join(f"{v}={n}" for v, n in c.top_values[:5])
                parts.append(f"  top: {top}")
            elif c.role == "date":
                parts.append(f"  range: {c.min_date} → {c.max_date}")
            elif c.role == "unusable":
                parts.append(f"  unusable_reason: {c.unusable_reason}")
            lines.extend(parts)
        if self.correlations:
            lines.append("Correlations (|r| ≥ 0.3):")
            for pair, r in self.correlations.items():
                lines.append(f"- {pair}: {r:.2f}")
        if self.anomalies:
            lines.append("Anomalies:")
            for a in self.anomalies:
                lines.append(f"- {a}")
        return "\n".join(lines)


class ChartSpec(BaseModel):
    kind: ChartKind
    title: str
    intent: str
    x: Optional[list[Any]] = None
    y: Optional[list[Any]] = None
    series: Optional[list[dict]] = None
    x_label: str = ""
    y_label: str = ""
    x_display_type: XDisplayType = "category"
    y_display_type: YDisplayType = "number"
    source_columns: list[str]
    data_point_count: int


class ToolError(BaseModel):
    reason: str


class ChartWithCaption(BaseModel):
    spec: ChartSpec
    caption: str


class ReportNarrative(BaseModel):
    summary: str
    captions: list[str]
    data_quality: list[str]


class Report(BaseModel):
    generated_at: str
    summary: str
    data_quality: list[str]
    charts: list[ChartWithCaption]
    metadata: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest tests/unit/test_schemas.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/schemas.py tests/unit/test_schemas.py
git commit -m "feat: Pydantic schemas for chart specs and reports"
```

---

### Task 4: Model alias config

**Files:**
- Create: `src/api/llm_config.py`
- Create: `tests/unit/test_llm_config.py`

- [ ] **Step 1: Write the failing test**

Write `tests/unit/test_llm_config.py`:

```python
import os
import importlib
import pytest


def reload_config():
    import llm_config
    importlib.reload(llm_config)
    return llm_config


def test_default_is_haiku(monkeypatch):
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL_SELECTION", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL_NARRATIVE", raising=False)
    cfg = reload_config()
    assert cfg.MODEL_SELECTION == "claude-haiku-4-5-20251001"
    assert cfg.MODEL_NARRATIVE == "claude-haiku-4-5-20251001"


def test_generic_override(monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "sonnet-4-6")
    monkeypatch.delenv("CLAUDE_MODEL_SELECTION", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL_NARRATIVE", raising=False)
    cfg = reload_config()
    assert cfg.MODEL_SELECTION == "claude-sonnet-4-6"
    assert cfg.MODEL_NARRATIVE == "claude-sonnet-4-6"


def test_per_pass_override(monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "haiku-4-5")
    monkeypatch.setenv("CLAUDE_MODEL_NARRATIVE", "sonnet-4-6")
    cfg = reload_config()
    assert cfg.MODEL_SELECTION == "claude-haiku-4-5-20251001"
    assert cfg.MODEL_NARRATIVE == "claude-sonnet-4-6"


def test_passthrough_full_id(monkeypatch):
    monkeypatch.setenv("CLAUDE_MODEL", "claude-opus-4-7")
    cfg = reload_config()
    assert cfg.MODEL_SELECTION == "claude-opus-4-7"


def test_resolve_unknown_alias_passes_through():
    cfg = reload_config()
    assert cfg.resolve("custom-id-string") == "custom-id-string"
```

- [ ] **Step 2: Run test, expect import error**

```bash
pytest tests/unit/test_llm_config.py -v
```

Expected: ModuleNotFoundError for `llm_config`.

- [ ] **Step 3: Implement llm_config.py**

Write `src/api/llm_config.py`:

```python
"""Model alias resolution.

Env var resolution order (first non-empty wins):
1. Per-pass override: CLAUDE_MODEL_SELECTION / CLAUDE_MODEL_NARRATIVE
2. Generic: CLAUDE_MODEL
3. Default: 'haiku-4-5'
"""
import os


MODEL_ALIASES = {
    "haiku-4-5":  "claude-haiku-4-5-20251001",
    "sonnet-4-6": "claude-sonnet-4-6",
    "opus-4-7":   "claude-opus-4-7",
}


def resolve(alias_or_id: str) -> str:
    """Map a friendly alias to a full model ID; pass through unknown strings."""
    return MODEL_ALIASES.get(alias_or_id, alias_or_id)


def _pick(*candidates: str | None, default: str) -> str:
    for c in candidates:
        if c:
            return c
    return default


MODEL_SELECTION = resolve(_pick(
    os.getenv("CLAUDE_MODEL_SELECTION"),
    os.getenv("CLAUDE_MODEL"),
    default="haiku-4-5",
))

MODEL_NARRATIVE = resolve(_pick(
    os.getenv("CLAUDE_MODEL_NARRATIVE"),
    os.getenv("CLAUDE_MODEL"),
    default="haiku-4-5",
))
```

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest tests/unit/test_llm_config.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/llm_config.py tests/unit/test_llm_config.py
git commit -m "feat: env-driven model alias config"
```

---

### Task 5: Claude client wrapper

**Files:**
- Create: `src/api/claude_client.py`
- Create: `tests/unit/test_claude_client.py`

- [ ] **Step 1: Write the failing test**

Write `tests/unit/test_claude_client.py`:

```python
"""Tests for the Claude client wrapper using mocks (no real API calls)."""
from unittest.mock import MagicMock, patch
import pytest
from claude_client import ClaudeClient, RetryableBusy


def make_response(content_blocks=None, usage=None):
    r = MagicMock()
    r.content = content_blocks or []
    r.usage = usage or MagicMock(input_tokens=100, output_tokens=50,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0)
    r.model = "claude-haiku-4-5-20251001"
    return r


def test_simple_call_returns_response():
    fake_sdk = MagicMock()
    fake_sdk.messages.create.return_value = make_response()

    client = ClaudeClient(api_key="test-key", _sdk=fake_sdk)
    resp = client.messages_create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp is not None
    fake_sdk.messages.create.assert_called_once()


def test_caches_system_and_tools_when_requested():
    fake_sdk = MagicMock()
    fake_sdk.messages.create.return_value = make_response()

    client = ClaudeClient(api_key="test-key", _sdk=fake_sdk)
    client.messages_create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        system="You are a helper.",
        tools=[{"name": "t", "description": "d", "input_schema": {"type": "object"}}],
        messages=[{"role": "user", "content": "hi"}],
        cache_static=True,
    )

    call_args = fake_sdk.messages.create.call_args.kwargs
    assert call_args["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call_args["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_retries_on_transient_5xx():
    from anthropic import APIStatusError
    err = APIStatusError("server error", response=MagicMock(status_code=502), body=None)
    fake_sdk = MagicMock()
    fake_sdk.messages.create.side_effect = [err, err, make_response()]

    client = ClaudeClient(api_key="test-key", _sdk=fake_sdk, _sleep=lambda s: None)
    resp = client.messages_create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp is not None
    assert fake_sdk.messages.create.call_count == 3


def test_surfaces_529_as_retryable_busy():
    from anthropic import APIStatusError
    err = APIStatusError("overloaded", response=MagicMock(status_code=529), body=None)
    fake_sdk = MagicMock()
    fake_sdk.messages.create.side_effect = err

    client = ClaudeClient(api_key="test-key", _sdk=fake_sdk, _sleep=lambda s: None)
    with pytest.raises(RetryableBusy):
        client.messages_create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )


def test_gives_up_after_max_retries():
    from anthropic import APIStatusError
    err = APIStatusError("server error", response=MagicMock(status_code=502), body=None)
    fake_sdk = MagicMock()
    fake_sdk.messages.create.side_effect = err

    client = ClaudeClient(api_key="test-key", _sdk=fake_sdk, _sleep=lambda s: None)
    with pytest.raises(APIStatusError):
        client.messages_create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert fake_sdk.messages.create.call_count == 3  # max attempts
```

- [ ] **Step 2: Run test, expect import error**

```bash
pytest tests/unit/test_claude_client.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement claude_client.py**

Write `src/api/claude_client.py`:

```python
"""Thin wrapper around the anthropic SDK.

Responsibilities:
- Exponential backoff retry on transient 5xx (max 3 attempts).
- Surface 529 (overloaded) as RetryableBusy so the API layer can return 503.
- Optionally cache_control system + tools (saves ~90% on input tokens for repeats).
- Log token usage per call (caller decides what to do with it).
"""
import logging
import time
from typing import Any, Callable, Optional

import anthropic
from anthropic import APIStatusError


class RetryableBusy(Exception):
    """Raised when Claude returns 529 (overloaded). The API layer maps this to HTTP 503."""


class ClaudeClient:
    MAX_ATTEMPTS = 3
    BACKOFF_SECONDS = (1.0, 2.0, 4.0)

    def __init__(self, api_key: str, _sdk: Any = None, _sleep: Callable[[float], None] = time.sleep):
        self._sdk = _sdk if _sdk is not None else anthropic.Anthropic(api_key=api_key)
        self._sleep = _sleep

    def messages_create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict],
        system: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[dict] = None,
        cache_static: bool = False,
    ):
        """Call anthropic.messages.create with retries and optional caching."""
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        if system:
            if cache_static:
                kwargs["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            else:
                kwargs["system"] = system

        if tools:
            if cache_static:
                tools_with_cache = [dict(t) for t in tools]
                tools_with_cache[-1] = {**tools_with_cache[-1], "cache_control": {"type": "ephemeral"}}
                kwargs["tools"] = tools_with_cache
            else:
                kwargs["tools"] = tools

        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        return self._call_with_retries(kwargs)

    def _call_with_retries(self, kwargs: dict):
        last_exc: Optional[Exception] = None
        for attempt in range(self.MAX_ATTEMPTS):
            try:
                response = self._sdk.messages.create(**kwargs)
                if hasattr(response, "usage"):
                    u = response.usage
                    logging.info(
                        "[CLAUDE] model=%s input=%d output=%d cache_read=%d cache_write=%d",
                        getattr(response, "model", "?"),
                        getattr(u, "input_tokens", 0),
                        getattr(u, "output_tokens", 0),
                        getattr(u, "cache_read_input_tokens", 0),
                        getattr(u, "cache_creation_input_tokens", 0),
                    )
                return response
            except APIStatusError as e:
                status = getattr(e.response, "status_code", 0) if hasattr(e, "response") else 0
                if status == 529:
                    raise RetryableBusy("Claude API is overloaded") from e
                if 500 <= status < 600:
                    last_exc = e
                    if attempt < self.MAX_ATTEMPTS - 1:
                        self._sleep(self.BACKOFF_SECONDS[attempt])
                        continue
                raise
        assert last_exc is not None
        raise last_exc
```

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest tests/unit/test_claude_client.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/claude_client.py tests/unit/test_claude_client.py
git commit -m "feat: Claude client with retries and prompt caching"
```

---

### Task 6: Data profiling

**Files:**
- Create: `src/api/profile.py`
- Create: `tests/unit/test_profile.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/unit/test_profile.py`:

```python
import pandas as pd
from profile import profile_dataframe


def test_detects_numeric_role(activities):
    profile = profile_dataframe(activities)
    duration = next(c for c in profile.columns if c.name == "duration_minutes")
    assert duration.role == "numeric"
    assert duration.min is not None
    assert duration.max is not None


def test_detects_categorical_role(activities):
    profile = profile_dataframe(activities)
    atype = next(c for c in profile.columns if c.name == "activity_type")
    assert atype.role == "categorical"
    assert atype.top_values is not None
    assert len(atype.top_values) > 0


def test_detects_date_role(activities):
    profile = profile_dataframe(activities)
    adate = next(c for c in profile.columns if c.name == "activity_date")
    assert adate.role == "date"
    assert adate.min_date is not None


def test_identifier_by_name_suffix():
    df = pd.DataFrame({"activity_id": [1, 2, 3, 4, 5], "x": [10, 20, 30, 40, 50]})
    profile = profile_dataframe(df)
    aid = next(c for c in profile.columns if c.name == "activity_id")
    assert aid.role == "identifier"


def test_identifier_by_cardinality():
    df = pd.DataFrame({"unique_col": list(range(100)), "low_card": [1, 2] * 50})
    profile = profile_dataframe(df)
    unique = next(c for c in profile.columns if c.name == "unique_col")
    assert unique.role == "identifier"


def test_low_cardinality_numeric_still_numeric():
    """Ratings 1-5 should be numeric, not unusable."""
    df = pd.DataFrame({"rating": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]})
    profile = profile_dataframe(df)
    rating = next(c for c in profile.columns if c.name == "rating")
    assert rating.role == "numeric"


def test_anomaly_negative_duration(negative_duration):
    profile = profile_dataframe(negative_duration)
    assert any("negative" in a.lower() for a in profile.anomalies)


def test_correlations_for_numeric_pairs():
    df = pd.DataFrame({
        "x": list(range(20)),
        "y": [i * 2 + 1 for i in range(20)],  # perfectly correlated
        "category": ["a"] * 20,
    })
    profile = profile_dataframe(df)
    assert len(profile.correlations) >= 1
    assert any(abs(r) > 0.9 for r in profile.correlations.values())


def test_degenerate_df_still_profiles(degenerate):
    profile = profile_dataframe(degenerate)
    assert profile.row_count == 8
    assert len(profile.columns) == 1


def test_to_text_includes_columns(activities):
    profile = profile_dataframe(activities)
    text = profile.to_text()
    assert "activity_type" in text
    assert "duration_minutes" in text
```

- [ ] **Step 2: Run tests, expect failure**

```bash
pytest tests/unit/test_profile.py -v
```

Expected: ModuleNotFoundError for `profile`.

- [ ] **Step 3: Implement profile.py**

Write `src/api/profile.py`:

```python
"""DataFrame profiling for AI chart selection.

profile_dataframe(df) returns a DataProfile that captures the structural facts
Claude needs to pick good charts: column roles, basic stats, correlations, anomalies.
The raw DataFrame is never sent to Claude — only this profile.
"""
import re
import numpy as np
import pandas as pd
from schemas import ColumnInfo, DataProfile


_IDENTIFIER_SUFFIXES = ("_id", "_code", "_uuid", "_key")
_NON_NEGATIVE_KEYWORDS = ("duration", "count", "quantity", "age", "price",
                          "amount", "revenue", "sales", "cost")
_DATE_KEYWORDS = ("date", "time", "created", "updated", "start", "end", "timestamp")


def _is_identifier(name: str, dtype: str, cardinality: int, row_count: int) -> bool:
    lower = name.lower()
    if any(lower.endswith(s) for s in _IDENTIFIER_SUFFIXES):
        return True
    if np.issubdtype(np.dtype(dtype), np.number) and cardinality > 0.5 * row_count and row_count >= 10:
        return True
    return False


def _is_date(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    nonnull = series.dropna()
    if len(nonnull) == 0:
        return False
    # Don't try to coerce purely numeric data to datetime
    if pd.api.types.is_numeric_dtype(nonnull):
        return False
    try:
        parsed = pd.to_datetime(nonnull, errors="coerce")
    except Exception:
        return False
    return (parsed.notna().sum() / len(nonnull)) >= 0.8


def _profile_column(name: str, series: pd.Series, row_count: int) -> ColumnInfo:
    dtype = str(series.dtype)
    cardinality = int(series.nunique(dropna=True))
    null_count = int(series.isna().sum())

    if _is_identifier(name, dtype, cardinality, row_count):
        return ColumnInfo(name=name, dtype=dtype, role="identifier",
                          cardinality=cardinality, null_count=null_count)

    if _is_date(series):
        parsed = pd.to_datetime(series, errors="coerce").dropna()
        return ColumnInfo(
            name=name, dtype=dtype, role="date",
            cardinality=cardinality, null_count=null_count,
            min_date=parsed.min().isoformat() if len(parsed) else None,
            max_date=parsed.max().isoformat() if len(parsed) else None,
        )

    if pd.api.types.is_numeric_dtype(series):
        nonnull = series.dropna()
        return ColumnInfo(
            name=name, dtype=dtype, role="numeric",
            cardinality=cardinality, null_count=null_count,
            min=float(nonnull.min()) if len(nonnull) else None,
            max=float(nonnull.max()) if len(nonnull) else None,
            mean=float(nonnull.mean()) if len(nonnull) else None,
            median=float(nonnull.median()) if len(nonnull) else None,
            std=float(nonnull.std()) if len(nonnull) > 1 else None,
        )

    if cardinality <= 50:
        top = series.value_counts(dropna=True).head(5)
        return ColumnInfo(
            name=name, dtype=dtype, role="categorical",
            cardinality=cardinality, null_count=null_count,
            top_values=[(k, int(v)) for k, v in top.items()],
        )

    return ColumnInfo(
        name=name, dtype=dtype, role="unusable",
        cardinality=cardinality, null_count=null_count,
        unusable_reason=f"object column with {cardinality} unique values, too high-cardinality to chart",
    )


def _compute_correlations(df: pd.DataFrame, columns: list[ColumnInfo]) -> dict[str, float]:
    numeric_cols = [c.name for c in columns if c.role == "numeric"]
    if len(numeric_cols) < 2:
        return {}
    corr_matrix = df[numeric_cols].corr(numeric_only=True)
    result: dict[str, float] = {}
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1:]:
            r = corr_matrix.loc[c1, c2]
            if pd.notna(r) and abs(r) >= 0.3:
                result[f"{c1}||{c2}"] = float(round(r, 3))
    return result


def _detect_anomalies(df: pd.DataFrame, columns: list[ColumnInfo]) -> list[str]:
    anomalies: list[str] = []
    now = pd.Timestamp.utcnow().tz_localize(None)

    for col in columns:
        lower = col.name.lower()

        if col.role == "numeric" and any(kw in lower for kw in _NON_NEGATIVE_KEYWORDS):
            if col.min is not None and col.min < 0:
                anomalies.append(
                    f"{col.name} contains negative values (min={col.min}); "
                    f"column name suggests it should be non-negative."
                )

        if col.role == "date" and col.max_date:
            try:
                max_ts = pd.Timestamp(col.max_date).tz_localize(None)
                if max_ts > now + pd.Timedelta(days=365):
                    anomalies.append(f"{col.name} contains future dates (max={col.max_date}).")
            except Exception:
                pass

        if col.null_count > 0.95 * df.shape[0] and df.shape[0] > 0:
            anomalies.append(f"{col.name} is >95% null ({col.null_count}/{df.shape[0]}); unusable.")

        if col.role == "numeric" and col.cardinality <= 2:
            anomalies.append(f"{col.name} has cardinality {col.cardinality}; behaves like a boolean.")

        if (col.role == "numeric" and col.std is not None and col.std > 0
                and col.max is not None and col.mean is not None
                and col.max > col.mean + 10 * col.std):
            anomalies.append(
                f"{col.name} has an extreme outlier (max={col.max}, mean={col.mean}, std={col.std}); "
                f"histograms may show empty bins."
            )

    return anomalies


def profile_dataframe(df: pd.DataFrame) -> DataProfile:
    """Build a DataProfile from a pandas DataFrame."""
    row_count = int(df.shape[0])
    columns = [_profile_column(name, df[name], row_count) for name in df.columns]
    correlations = _compute_correlations(df, columns)
    anomalies = _detect_anomalies(df, columns)
    return DataProfile(
        row_count=row_count,
        columns=columns,
        correlations=correlations,
        anomalies=anomalies,
    )
```

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest tests/unit/test_profile.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/profile.py tests/unit/test_profile.py
git commit -m "feat: DataFrame profiling for chart selection"
```

---

## Phase 2 — Chart executors

All executors live in `src/api/chart_executor.py` and share a dispatch dict `TOOL_EXECUTORS`. Each executor takes `(df: DataFrame, params: dict) → ChartSpec | ToolError`.

### Task 7: frequency_bar_chart executor

**Files:**
- Create: `src/api/chart_executor.py`
- Create: `tests/unit/test_executors_frequency.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/unit/test_executors_frequency.py`:

```python
"""Regression net for the original frequency-chart bug.

The old system computed value_counts on the counts themselves, producing a
'histogram of frequencies' instead of an actual frequency chart. These tests
assert the EXACT x/y values that the correct implementation produces.
"""
import pandas as pd
import pytest
from chart_executor import execute_frequency_bar_chart
from schemas import ChartSpec, ToolError


def _params(column="activity_type", title="t", intent="i"):
    return {"column": column, "title": title, "intent": intent}


def test_happy_path_exact_values():
    df = pd.DataFrame({"activity_type": ["a", "b", "a", "a", "b", "c"]})
    result = execute_frequency_bar_chart(df, _params())
    assert isinstance(result, ChartSpec)
    assert result.x == ["a", "b", "c"]
    assert result.y == [3, 2, 1]


def test_sorted_descending_by_count():
    df = pd.DataFrame({"activity_type": ["c"] * 5 + ["a"] * 2 + ["b"] * 3})
    result = execute_frequency_bar_chart(df, _params())
    assert result.x == ["c", "b", "a"]
    assert result.y == [5, 3, 2]


def test_with_real_fixture(activities):
    result = execute_frequency_bar_chart(activities, _params())
    assert isinstance(result, ChartSpec)
    assert "consultation" in result.x
    assert sum(result.y) == len(activities)


def test_missing_column_returns_error(activities):
    result = execute_frequency_bar_chart(activities, _params(column="not_a_column"))
    assert isinstance(result, ToolError)
    assert "not_a_column" in result.reason
    assert "activity_type" in result.reason  # offers alternatives


def test_too_many_categories_returns_error():
    df = pd.DataFrame({"high_card": [f"v{i}" for i in range(40)]})
    result = execute_frequency_bar_chart(df, _params(column="high_card"))
    assert isinstance(result, ToolError)
    assert "40" in result.reason or "too many" in result.reason.lower()


def test_all_null_returns_error():
    df = pd.DataFrame({"col": [None, None, None]})
    result = execute_frequency_bar_chart(df, _params(column="col"))
    assert isinstance(result, ToolError)


def test_chartspec_metadata_populated():
    df = pd.DataFrame({"activity_type": ["a", "b", "a"]})
    result = execute_frequency_bar_chart(df, _params())
    assert result.kind == "bar"
    assert result.source_columns == ["activity_type"]
    assert result.data_point_count == 3
    assert result.y_display_type == "count"
```

- [ ] **Step 2: Run tests, expect failure**

```bash
pytest tests/unit/test_executors_frequency.py -v
```

Expected: ModuleNotFoundError for `chart_executor`.

- [ ] **Step 3: Implement chart_executor.py with frequency_bar_chart**

Write `src/api/chart_executor.py`:

```python
"""Chart executors: pure functions from (df, tool_params) to ChartSpec or ToolError.

One executor per Anthropic tool. Each must:
- Validate columns exist with correct roles
- Validate cardinality constraints
- Compute the chart data and return a fully-populated ChartSpec
- On any failure, return ToolError with an actionable, specific reason
"""
from typing import Any, Callable
import numpy as np
import pandas as pd
from schemas import ChartSpec, ToolError


MAX_CATEGORIES = 30
MAX_PIE_SLICES = 8
MAX_SCATTER_POINTS = 5000


def _err(reason: str) -> ToolError:
    return ToolError(reason=reason)


def _available_columns_by_role(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group columns by inferred role for error messages."""
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in df.columns if df[c].dtype == "object" and df[c].nunique(dropna=True) <= 50]
    return {"numeric": numeric, "categorical": categorical, "all": list(df.columns)}


def execute_frequency_bar_chart(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    column = params["column"]
    title = params["title"]
    intent = params["intent"]

    if column not in df.columns:
        cats = _available_columns_by_role(df)["categorical"]
        return _err(f"'{column}' is not a column. Available categorical columns: {cats}")

    series = df[column].dropna()
    if len(series) == 0:
        return _err(f"'{column}' has no non-null values.")

    counts = series.value_counts()
    if len(counts) > MAX_CATEGORIES:
        return _err(
            f"'{column}' has {len(counts)} unique values, more than the max ({MAX_CATEGORIES}). "
            f"Use frequency charts on lower-cardinality columns; this column may be an identifier."
        )

    x = [str(v) for v in counts.index.tolist()]
    y = [int(v) for v in counts.values.tolist()]

    return ChartSpec(
        kind="bar",
        title=title,
        intent=intent,
        x=x,
        y=y,
        x_label=column,
        y_label="Count",
        x_display_type="category",
        y_display_type="count",
        source_columns=[column],
        data_point_count=int(len(series)),
    )


TOOL_EXECUTORS: dict[str, Callable[[pd.DataFrame, dict], ChartSpec | ToolError]] = {
    "frequency_bar_chart": execute_frequency_bar_chart,
}
```

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest tests/unit/test_executors_frequency.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/chart_executor.py tests/unit/test_executors_frequency.py
git commit -m "feat: frequency_bar_chart executor (kills the regression bug)"
```

---

### Task 8: aggregation_bar_chart executor

**Files:**
- Modify: `src/api/chart_executor.py`
- Create: `tests/unit/test_executors_aggregation.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/unit/test_executors_aggregation.py`:

```python
import pandas as pd
import pytest
from chart_executor import execute_aggregation_bar_chart
from schemas import ChartSpec, ToolError


def _params(value_col="revenue", group_col="region", agg="sum", title="t", intent="i"):
    return {"value_col": value_col, "group_col": group_col, "agg": agg, "title": title, "intent": intent}


def test_sum_happy_path(sales):
    result = execute_aggregation_bar_chart(sales, _params())
    assert isinstance(result, ChartSpec)
    assert set(result.x) == {"north", "south", "east", "west"}
    # Verify exact sum for one region
    idx = result.x.index("north")
    expected = sales[sales["region"] == "north"]["revenue"].sum()
    assert result.y[idx] == pytest.approx(expected)


def test_mean_aggregation(sales):
    result = execute_aggregation_bar_chart(sales, _params(agg="mean"))
    assert isinstance(result, ChartSpec)
    idx = result.x.index("north")
    expected = sales[sales["region"] == "north"]["revenue"].mean()
    assert result.y[idx] == pytest.approx(expected)


def test_median_aggregation(sales):
    result = execute_aggregation_bar_chart(sales, _params(agg="median"))
    assert isinstance(result, ChartSpec)


def test_min_max_aggregations(sales):
    rmin = execute_aggregation_bar_chart(sales, _params(agg="min"))
    rmax = execute_aggregation_bar_chart(sales, _params(agg="max"))
    assert isinstance(rmin, ChartSpec) and isinstance(rmax, ChartSpec)
    for region in rmin.x:
        i_min = rmin.x.index(region)
        i_max = rmax.x.index(region)
        assert rmin.y[i_min] <= rmax.y[i_max]


def test_count_agg_rejected():
    """count is for frequency_bar_chart, not aggregation_bar_chart."""
    df = pd.DataFrame({"v": [1, 2, 3], "g": ["a", "b", "a"]})
    result = execute_aggregation_bar_chart(df, _params(value_col="v", group_col="g", agg="count"))
    assert isinstance(result, ToolError)
    assert "frequency_bar_chart" in result.reason


def test_value_col_must_be_numeric():
    df = pd.DataFrame({"v": ["x", "y", "z"], "g": ["a", "b", "a"]})
    result = execute_aggregation_bar_chart(df, _params(value_col="v", group_col="g"))
    assert isinstance(result, ToolError)
    assert "numeric" in result.reason.lower()


def test_missing_value_col(sales):
    result = execute_aggregation_bar_chart(sales, _params(value_col="nope"))
    assert isinstance(result, ToolError)
    assert "nope" in result.reason


def test_missing_group_col(sales):
    result = execute_aggregation_bar_chart(sales, _params(group_col="nope"))
    assert isinstance(result, ToolError)
    assert "nope" in result.reason


def test_too_many_groups():
    df = pd.DataFrame({"v": list(range(50)), "g": [f"g{i}" for i in range(50)]})
    result = execute_aggregation_bar_chart(df, _params(value_col="v", group_col="g"))
    assert isinstance(result, ToolError)
```

- [ ] **Step 2: Run tests, expect failure**

```bash
pytest tests/unit/test_executors_aggregation.py -v
```

Expected: ImportError on `execute_aggregation_bar_chart`.

- [ ] **Step 3: Add executor to chart_executor.py**

Append to `src/api/chart_executor.py` (before `TOOL_EXECUTORS`):

```python
_AGGREGATION_BAR_AGGS = {"sum", "mean", "median", "min", "max"}


def execute_aggregation_bar_chart(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    value_col = params["value_col"]
    group_col = params["group_col"]
    agg = params["agg"]
    title = params["title"]
    intent = params["intent"]

    if agg == "count":
        return _err("aggregation_bar_chart does not support agg='count'. Use frequency_bar_chart for counts by category.")

    if agg not in _AGGREGATION_BAR_AGGS:
        return _err(f"agg='{agg}' is not allowed. Allowed: {sorted(_AGGREGATION_BAR_AGGS)}.")

    avail = _available_columns_by_role(df)
    if value_col not in df.columns:
        return _err(f"'{value_col}' is not a column. Available numeric columns: {avail['numeric']}")

    if group_col not in df.columns:
        return _err(f"'{group_col}' is not a column. Available categorical columns: {avail['categorical']}")

    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return _err(f"'{value_col}' is not numeric. Available numeric columns: {avail['numeric']}")

    groups = df[group_col].dropna().nunique()
    if groups > MAX_CATEGORIES:
        return _err(f"'{group_col}' has {groups} unique values, more than max ({MAX_CATEGORIES}).")

    if groups == 0:
        return _err(f"'{group_col}' has no non-null values.")

    work = df[[group_col, value_col]].dropna()
    grouped = work.groupby(group_col)[value_col].agg(agg)
    grouped = grouped.sort_values(ascending=False)

    return ChartSpec(
        kind="bar",
        title=title,
        intent=intent,
        x=[str(k) for k in grouped.index.tolist()],
        y=[float(v) for v in grouped.values.tolist()],
        x_label=group_col,
        y_label=f"{agg.capitalize()} of {value_col}",
        x_display_type="category",
        y_display_type="number",
        source_columns=[value_col, group_col],
        data_point_count=int(len(work)),
    )
```

Then update `TOOL_EXECUTORS` at the bottom of the file:

```python
TOOL_EXECUTORS: dict[str, Callable[[pd.DataFrame, dict], ChartSpec | ToolError]] = {
    "frequency_bar_chart": execute_frequency_bar_chart,
    "aggregation_bar_chart": execute_aggregation_bar_chart,
}
```

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest tests/unit/test_executors_aggregation.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/chart_executor.py tests/unit/test_executors_aggregation.py
git commit -m "feat: aggregation_bar_chart executor"
```

---

### Task 9: histogram_chart executor (with trimmed-IQR binning)

**Files:**
- Modify: `src/api/chart_executor.py`
- Modify: `src/api/data_processing_utils.py` (rewrite the bin-selection logic)
- Create: `tests/unit/test_executors_histogram.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/unit/test_executors_histogram.py`:

```python
import numpy as np
import pandas as pd
import pytest
from chart_executor import execute_histogram_chart
from schemas import ChartSpec, ToolError


def _params(column="duration_minutes", title="t", intent="i"):
    return {"column": column, "title": title, "intent": intent}


def test_happy_path_returns_spec(activities):
    result = execute_histogram_chart(activities, _params())
    assert isinstance(result, ChartSpec)
    assert result.kind == "histogram"
    assert len(result.x) == len(result.y)
    assert len(result.x) >= 5
    assert len(result.x) <= 20


def test_y_values_sum_to_input_count():
    df = pd.DataFrame({"x": list(np.random.RandomState(0).normal(100, 15, 200))})
    result = execute_histogram_chart(df, _params(column="x"))
    assert isinstance(result, ChartSpec)
    assert sum(result.y) == 200


def test_outlier_trimmed_bins_are_useful():
    """One extreme outlier should NOT destroy bin distribution."""
    values = list(np.random.RandomState(0).normal(100, 15, 200))
    values.append(100_000.0)  # extreme outlier
    df = pd.DataFrame({"x": values})
    result = execute_histogram_chart(df, _params(column="x"))
    assert isinstance(result, ChartSpec)
    non_empty_bins = sum(1 for v in result.y if v > 0)
    assert non_empty_bins >= 5, f"only {non_empty_bins} non-empty bins; outlier trimming failed"


def test_constant_column_returns_error():
    df = pd.DataFrame({"x": [5.0] * 50})
    result = execute_histogram_chart(df, _params(column="x"))
    assert isinstance(result, ToolError)
    assert "constant" in result.reason.lower() or "no variance" in result.reason.lower()


def test_non_numeric_returns_error():
    df = pd.DataFrame({"x": ["a", "b", "c"]})
    result = execute_histogram_chart(df, _params(column="x"))
    assert isinstance(result, ToolError)
    assert "numeric" in result.reason.lower()


def test_missing_column_returns_error(activities):
    result = execute_histogram_chart(activities, _params(column="nope"))
    assert isinstance(result, ToolError)


def test_all_null_returns_error():
    df = pd.DataFrame({"x": [None, None, None]})
    result = execute_histogram_chart(df, _params(column="x"))
    assert isinstance(result, ToolError)
```

- [ ] **Step 2: Run tests, expect failure**

```bash
pytest tests/unit/test_executors_histogram.py -v
```

Expected: ImportError on `execute_histogram_chart`.

- [ ] **Step 3: Replace data_processing_utils.py with trimmed-IQR binning**

Overwrite `src/api/data_processing_utils.py`:

```python
"""Pure data-processing helpers shared across executors."""
import logging
import numpy as np
import pandas as pd


def compute_group_count(df: pd.DataFrame, group_col: str) -> tuple[list, list]:
    """Return (categories, counts) sorted by count descending."""
    if group_col not in df.columns:
        return [], []
    counts = df[group_col].value_counts()
    return counts.index.tolist(), counts.values.tolist()


def compute_histogram_bins_and_freqs(
    values: pd.Series, max_bins: int = 20, min_bins: int = 5
) -> tuple[list[str], list[int]]:
    """Trimmed-IQR histogram binning robust to outliers.

    Drops values outside [Q1 - 3·IQR, Q3 + 3·IQR] before binning so a single
    extreme outlier doesn't pull the bin range and leave most bins empty.
    Falls back to full range if trimming removes >20% of data.
    """
    nonnull = pd.to_numeric(values, errors="coerce").dropna()
    if len(nonnull) == 0:
        return [], []

    if nonnull.nunique() < 2:
        return [], []

    q1, q3 = nonnull.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr > 0:
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        trimmed = nonnull[(nonnull >= lo) & (nonnull <= hi)]
        if len(trimmed) < 0.8 * len(nonnull):
            trimmed = nonnull  # too many dropped, use full range
    else:
        trimmed = nonnull

    n = len(trimmed)
    target_bins = max(min_bins, min(max_bins, int(np.ceil(2 * n ** (1 / 3)))))

    counts, edges = np.histogram(trimmed, bins=target_bins)
    labels = [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(edges) - 1)]
    return labels, counts.tolist()
```

- [ ] **Step 4: Add executor to chart_executor.py**

Append to `src/api/chart_executor.py` (before `TOOL_EXECUTORS`):

```python
def execute_histogram_chart(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    from data_processing_utils import compute_histogram_bins_and_freqs

    column = params["column"]
    title = params["title"]
    intent = params["intent"]

    if column not in df.columns:
        return _err(f"'{column}' is not a column. Available numeric columns: {_available_columns_by_role(df)['numeric']}")

    if not pd.api.types.is_numeric_dtype(df[column]):
        return _err(f"'{column}' is not numeric. Histograms require numeric columns.")

    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(series) == 0:
        return _err(f"'{column}' has no non-null numeric values.")

    if series.nunique() < 2:
        return _err(f"'{column}' is constant (or has no variance); can't build a histogram.")

    labels, freqs = compute_histogram_bins_and_freqs(series)
    if not labels:
        return _err(f"'{column}' did not produce usable histogram bins.")

    return ChartSpec(
        kind="histogram",
        title=title,
        intent=intent,
        x=labels,
        y=[int(v) for v in freqs],
        x_label=column,
        y_label="Frequency",
        x_display_type="text",
        y_display_type="count",
        source_columns=[column],
        data_point_count=int(len(series)),
    )
```

Then add to `TOOL_EXECUTORS`:

```python
TOOL_EXECUTORS: dict[str, Callable[[pd.DataFrame, dict], ChartSpec | ToolError]] = {
    "frequency_bar_chart": execute_frequency_bar_chart,
    "aggregation_bar_chart": execute_aggregation_bar_chart,
    "histogram_chart": execute_histogram_chart,
}
```

- [ ] **Step 5: Run tests, expect pass**

```bash
pytest tests/unit/test_executors_histogram.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/api/chart_executor.py src/api/data_processing_utils.py tests/unit/test_executors_histogram.py
git commit -m "feat: histogram executor with trimmed-IQR binning"
```

---

### Task 10: scatter_chart executor

**Files:**
- Modify: `src/api/chart_executor.py`
- Create: `tests/unit/test_executors_scatter.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/unit/test_executors_scatter.py`:

```python
import numpy as np
import pandas as pd
import pytest
from chart_executor import execute_scatter_chart
from schemas import ChartSpec, ToolError


def _params(x_col="duration_minutes", y_col="activity_id", color_by=None, title="t", intent="i"):
    p = {"x_col": x_col, "y_col": y_col, "title": title, "intent": intent}
    if color_by is not None:
        p["color_by"] = color_by
    return p


def test_happy_path(activities):
    result = execute_scatter_chart(activities, _params())
    assert isinstance(result, ChartSpec)
    assert result.kind == "scatter"
    assert len(result.x) == len(result.y)
    assert len(result.x) <= len(activities)


def test_with_color_by(activities):
    result = execute_scatter_chart(activities, _params(color_by="activity_type"))
    assert isinstance(result, ChartSpec)
    assert result.series is not None
    assert len(result.series) > 0


def test_drops_nan_pairs():
    df = pd.DataFrame({
        "x": [1.0, 2.0, None, 4.0, 5.0],
        "y": [10.0, None, 30.0, 40.0, 50.0],
    })
    result = execute_scatter_chart(df, _params(x_col="x", y_col="y"))
    assert isinstance(result, ChartSpec)
    # Only rows 0 (1,10), 3 (4,40), 4 (5,50) have both
    assert len(result.x) == 3


def test_samples_when_over_max():
    np.random.seed(0)
    n = 6000
    df = pd.DataFrame({"x": np.random.rand(n), "y": np.random.rand(n)})
    result = execute_scatter_chart(df, _params(x_col="x", y_col="y"))
    assert isinstance(result, ChartSpec)
    assert len(result.x) <= 5000


def test_non_numeric_x_returns_error():
    df = pd.DataFrame({"x": ["a", "b"], "y": [1, 2]})
    result = execute_scatter_chart(df, _params(x_col="x", y_col="y"))
    assert isinstance(result, ToolError)
    assert "numeric" in result.reason.lower()


def test_missing_column(activities):
    result = execute_scatter_chart(activities, _params(x_col="nope"))
    assert isinstance(result, ToolError)


def test_color_by_must_be_categorical(activities):
    result = execute_scatter_chart(activities, _params(color_by="duration_minutes"))
    assert isinstance(result, ToolError)
    assert "color_by" in result.reason.lower() or "categorical" in result.reason.lower()


def test_color_by_high_cardinality_rejected():
    df = pd.DataFrame({
        "x": list(range(40)),
        "y": list(range(40)),
        "cat": [f"c{i}" for i in range(40)],
    })
    result = execute_scatter_chart(df, _params(x_col="x", y_col="y", color_by="cat"))
    assert isinstance(result, ToolError)
```

- [ ] **Step 2: Run tests, expect failure**

```bash
pytest tests/unit/test_executors_scatter.py -v
```

Expected: ImportError on `execute_scatter_chart`.

- [ ] **Step 3: Add executor**

Append to `src/api/chart_executor.py` (before `TOOL_EXECUTORS`):

```python
def execute_scatter_chart(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    x_col = params["x_col"]
    y_col = params["y_col"]
    color_by = params.get("color_by")
    title = params["title"]
    intent = params["intent"]

    avail = _available_columns_by_role(df)

    for col, label in [(x_col, "x_col"), (y_col, "y_col")]:
        if col not in df.columns:
            return _err(f"{label}='{col}' is not a column. Available numeric columns: {avail['numeric']}")
        if not pd.api.types.is_numeric_dtype(df[col]):
            return _err(f"{label}='{col}' is not numeric. Scatter charts need two numeric columns.")

    cols = [x_col, y_col]
    if color_by is not None:
        if color_by not in df.columns:
            return _err(f"color_by='{color_by}' is not a column.")
        if pd.api.types.is_numeric_dtype(df[color_by]) and df[color_by].nunique() > MAX_CATEGORIES:
            return _err(f"color_by='{color_by}' is numeric or high-cardinality; pass a categorical column with ≤{MAX_CATEGORIES} groups.")
        if df[color_by].nunique() > MAX_CATEGORIES:
            return _err(f"color_by='{color_by}' has {df[color_by].nunique()} unique values; max is {MAX_CATEGORIES}.")
        cols.append(color_by)

    work = df[cols].dropna()
    if len(work) == 0:
        return _err(f"No rows where both '{x_col}' and '{y_col}' are non-null.")

    if len(work) > MAX_SCATTER_POINTS:
        work = work.sample(n=MAX_SCATTER_POINTS, random_state=42)

    if color_by is None:
        return ChartSpec(
            kind="scatter",
            title=title,
            intent=intent,
            x=[float(v) for v in work[x_col].tolist()],
            y=[float(v) for v in work[y_col].tolist()],
            x_label=x_col,
            y_label=y_col,
            x_display_type="number",
            y_display_type="number",
            source_columns=[x_col, y_col],
            data_point_count=int(len(work)),
        )

    series_list: list[dict] = []
    for group_value, group_df in work.groupby(color_by):
        series_list.append({
            "name": str(group_value),
            "x": [float(v) for v in group_df[x_col].tolist()],
            "y": [float(v) for v in group_df[y_col].tolist()],
        })

    return ChartSpec(
        kind="scatter",
        title=title,
        intent=intent,
        series=series_list,
        x_label=x_col,
        y_label=y_col,
        x_display_type="number",
        y_display_type="number",
        source_columns=[x_col, y_col, color_by],
        data_point_count=int(len(work)),
    )
```

Add to `TOOL_EXECUTORS`:

```python
    "scatter_chart": execute_scatter_chart,
```

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest tests/unit/test_executors_scatter.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/chart_executor.py tests/unit/test_executors_scatter.py
git commit -m "feat: scatter_chart executor"
```

---

### Task 11: line_chart executor

**Files:**
- Modify: `src/api/chart_executor.py`
- Create: `tests/unit/test_executors_line.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/unit/test_executors_line.py`:

```python
import pandas as pd
import pytest
from chart_executor import execute_line_chart
from schemas import ChartSpec, ToolError


def _params(date_col="activity_date", value_col="duration_minutes",
            agg="count", granularity="month", group_by=None, title="t", intent="i"):
    p = {"date_col": date_col, "value_col": value_col, "agg": agg,
         "granularity": granularity, "title": title, "intent": intent}
    if group_by is not None:
        p["group_by"] = group_by
    return p


def test_count_by_month(activities):
    result = execute_line_chart(activities, _params(agg="count"))
    assert isinstance(result, ChartSpec)
    assert result.kind == "line"
    assert len(result.x) == len(result.y)
    assert sum(result.y) == len(activities)


def test_mean_by_month(activities):
    result = execute_line_chart(activities, _params(agg="mean"))
    assert isinstance(result, ChartSpec)
    assert all(isinstance(v, float) for v in result.y)


def test_sum_by_quarter(activities):
    result = execute_line_chart(activities, _params(agg="sum", granularity="quarter"))
    assert isinstance(result, ChartSpec)


def test_with_group_by(activities):
    result = execute_line_chart(activities, _params(group_by="activity_type"))
    assert isinstance(result, ChartSpec)
    assert result.series is not None
    assert len(result.series) >= 1


def test_invalid_granularity(activities):
    result = execute_line_chart(activities, _params(granularity="century"))
    assert isinstance(result, ToolError)


def test_invalid_agg(activities):
    result = execute_line_chart(activities, _params(agg="nope"))
    assert isinstance(result, ToolError)


def test_non_date_column(activities):
    result = execute_line_chart(activities, _params(date_col="activity_type"))
    assert isinstance(result, ToolError)
    assert "date" in result.reason.lower()


def test_value_col_must_be_numeric_when_not_count(activities):
    result = execute_line_chart(activities, _params(value_col="activity_type", agg="mean"))
    assert isinstance(result, ToolError)
    assert "numeric" in result.reason.lower()
```

- [ ] **Step 2: Run tests, expect failure**

```bash
pytest tests/unit/test_executors_line.py -v
```

Expected: ImportError on `execute_line_chart`.

- [ ] **Step 3: Add executor**

Append to `src/api/chart_executor.py` (before `TOOL_EXECUTORS`):

```python
_LINE_AGGS = {"count", "sum", "mean", "median", "min", "max"}
_GRANULARITIES = {
    "day": "D", "week": "W", "month": "M",
    "quarter": "Q", "year": "Y",
}


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def execute_line_chart(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    date_col = params["date_col"]
    value_col = params["value_col"]
    agg = params["agg"]
    granularity = params["granularity"]
    group_by = params.get("group_by")
    title = params["title"]
    intent = params["intent"]

    if agg not in _LINE_AGGS:
        return _err(f"agg='{agg}' is not allowed. Allowed: {sorted(_LINE_AGGS)}.")

    if granularity not in _GRANULARITIES:
        return _err(f"granularity='{granularity}' is not allowed. Allowed: {sorted(_GRANULARITIES.keys())}.")

    if date_col not in df.columns:
        return _err(f"date_col='{date_col}' is not a column.")

    parsed_dates = _to_datetime(df[date_col])
    if parsed_dates.notna().sum() < 0.5 * len(df):
        return _err(f"'{date_col}' could not be parsed as a date column.")

    if agg != "count":
        if value_col not in df.columns:
            return _err(f"value_col='{value_col}' is not a column.")
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            return _err(f"value_col='{value_col}' is not numeric (required when agg != 'count').")

    work = df.copy()
    work["_date"] = parsed_dates
    work = work.dropna(subset=["_date"])
    work["_period"] = work["_date"].dt.to_period(_GRANULARITIES[granularity])

    def _agg_one(g: pd.DataFrame) -> float:
        if agg == "count":
            return float(len(g))
        return float(g[value_col].agg(agg))

    if group_by is None:
        grouped = work.groupby("_period").apply(_agg_one).sort_index()
        return ChartSpec(
            kind="line",
            title=title,
            intent=intent,
            x=[str(p) for p in grouped.index.tolist()],
            y=[float(v) for v in grouped.values.tolist()],
            x_label=f"{granularity.capitalize()} ({date_col})",
            y_label=f"{agg.capitalize()}" + (f" of {value_col}" if agg != "count" else ""),
            x_display_type="date",
            y_display_type="count" if agg == "count" else "number",
            source_columns=[date_col] + ([value_col] if agg != "count" else []),
            data_point_count=int(len(work)),
        )

    if group_by not in df.columns:
        return _err(f"group_by='{group_by}' is not a column.")
    if work[group_by].nunique() > MAX_CATEGORIES:
        return _err(f"group_by='{group_by}' has {work[group_by].nunique()} unique values; max is {MAX_CATEGORIES}.")

    series_list: list[dict] = []
    all_periods: set = set()
    per_group: dict[str, pd.Series] = {}
    for gv, sub in work.groupby(group_by):
        s = sub.groupby("_period").apply(_agg_one).sort_index()
        per_group[str(gv)] = s
        all_periods.update(s.index.tolist())

    periods_sorted = sorted(all_periods)
    period_labels = [str(p) for p in periods_sorted]
    for name, s in per_group.items():
        aligned = [float(s.get(p, 0.0)) for p in periods_sorted]
        series_list.append({"name": name, "x": period_labels, "y": aligned})

    return ChartSpec(
        kind="line",
        title=title,
        intent=intent,
        series=series_list,
        x_label=f"{granularity.capitalize()} ({date_col})",
        y_label=f"{agg.capitalize()}" + (f" of {value_col}" if agg != "count" else ""),
        x_display_type="date",
        y_display_type="count" if agg == "count" else "number",
        source_columns=[date_col, group_by] + ([value_col] if agg != "count" else []),
        data_point_count=int(len(work)),
    )
```

Add to `TOOL_EXECUTORS`:

```python
    "line_chart": execute_line_chart,
```

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest tests/unit/test_executors_line.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/chart_executor.py tests/unit/test_executors_line.py
git commit -m "feat: line_chart executor with grouping and granularity"
```

---

### Task 12: pie_chart executor

**Files:**
- Modify: `src/api/chart_executor.py`
- Create: `tests/unit/test_executors_pie.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/unit/test_executors_pie.py`:

```python
import pandas as pd
import pytest
from chart_executor import execute_pie_chart
from schemas import ChartSpec, ToolError


def _params(category_col="activity_type", value_col=None, agg="sum", title="t", intent="i"):
    p = {"category_col": category_col, "agg": agg, "title": title, "intent": intent}
    if value_col is not None:
        p["value_col"] = value_col
    return p


def test_count_no_value_col(activities):
    result = execute_pie_chart(activities, _params(agg="count"))
    assert isinstance(result, ChartSpec)
    assert result.kind == "pie"
    assert sum(result.y) == len(activities)


def test_sum_with_value_col(activities):
    result = execute_pie_chart(activities, _params(value_col="duration_minutes", agg="sum"))
    assert isinstance(result, ChartSpec)
    expected_total = activities["duration_minutes"].sum()
    assert sum(result.y) == pytest.approx(expected_total)


def test_caps_to_max_slices_plus_other():
    df = pd.DataFrame({"cat": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"] * 5})
    result = execute_pie_chart(df, _params(category_col="cat", agg="count"))
    assert isinstance(result, ChartSpec)
    assert len(result.x) <= 9
    if len(result.x) == 9:
        assert "Other" in result.x


def test_invalid_agg(activities):
    result = execute_pie_chart(activities, _params(agg="median"))
    assert isinstance(result, ToolError)


def test_missing_category_col(activities):
    result = execute_pie_chart(activities, _params(category_col="nope"))
    assert isinstance(result, ToolError)


def test_value_col_required_for_non_count():
    df = pd.DataFrame({"cat": ["a", "b"]})
    result = execute_pie_chart(df, _params(category_col="cat", agg="sum"))
    assert isinstance(result, ToolError)
    assert "value_col" in result.reason.lower()
```

- [ ] **Step 2: Run tests, expect failure**

```bash
pytest tests/unit/test_executors_pie.py -v
```

Expected: ImportError.

- [ ] **Step 3: Add executor**

Append to `src/api/chart_executor.py` (before `TOOL_EXECUTORS`):

```python
_PIE_AGGS = {"sum", "mean", "count"}


def execute_pie_chart(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    category_col = params["category_col"]
    value_col = params.get("value_col")
    agg = params["agg"]
    title = params["title"]
    intent = params["intent"]

    if agg not in _PIE_AGGS:
        return _err(f"agg='{agg}' is not allowed for pie_chart. Allowed: {sorted(_PIE_AGGS)}.")

    if category_col not in df.columns:
        return _err(f"category_col='{category_col}' is not a column.")

    if agg != "count":
        if value_col is None:
            return _err("value_col is required when agg is not 'count'.")
        if value_col not in df.columns:
            return _err(f"value_col='{value_col}' is not a column.")
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            return _err(f"value_col='{value_col}' is not numeric.")

    cols = [category_col] + ([value_col] if agg != "count" else [])
    work = df[cols].dropna(subset=[category_col])
    if agg != "count":
        work = work.dropna(subset=[value_col])

    if len(work) == 0:
        return _err(f"No usable rows for pie chart on '{category_col}'.")

    if agg == "count":
        s = work[category_col].value_counts()
    else:
        s = work.groupby(category_col)[value_col].agg(agg)

    s = s.sort_values(ascending=False)
    if len(s) > MAX_PIE_SLICES:
        top = s.head(MAX_PIE_SLICES)
        other_val = float(s.iloc[MAX_PIE_SLICES:].sum())
        x = [str(k) for k in top.index.tolist()] + ["Other"]
        y = [float(v) for v in top.values.tolist()] + [other_val]
    else:
        x = [str(k) for k in s.index.tolist()]
        y = [float(v) for v in s.values.tolist()]

    return ChartSpec(
        kind="pie",
        title=title,
        intent=intent,
        x=x,
        y=y,
        x_label=category_col,
        y_label=f"{agg.capitalize()}" + (f" of {value_col}" if agg != "count" else ""),
        x_display_type="category",
        y_display_type="count" if agg == "count" else "number",
        source_columns=cols,
        data_point_count=int(len(work)),
    )
```

Add to `TOOL_EXECUTORS`:

```python
    "pie_chart": execute_pie_chart,
```

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest tests/unit/test_executors_pie.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/chart_executor.py tests/unit/test_executors_pie.py
git commit -m "feat: pie_chart executor with Other rollup"
```

---

### Task 13: box_plot executor

**Files:**
- Modify: `src/api/chart_executor.py`
- Create: `tests/unit/test_executors_box.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/unit/test_executors_box.py`:

```python
import pandas as pd
import pytest
from chart_executor import execute_box_plot
from schemas import ChartSpec, ToolError


def _params(value_col="duration_minutes", group_col=None, title="t", intent="i"):
    p = {"value_col": value_col, "title": title, "intent": intent}
    if group_col is not None:
        p["group_col"] = group_col
    return p


def test_single_box(activities):
    result = execute_box_plot(activities, _params())
    assert isinstance(result, ChartSpec)
    assert result.kind == "box"
    assert result.series is not None
    assert len(result.series) == 1
    stats = result.series[0]
    for k in ("min", "q1", "median", "q3", "max"):
        assert k in stats


def test_grouped_box(activities):
    result = execute_box_plot(activities, _params(group_col="activity_type"))
    assert isinstance(result, ChartSpec)
    assert len(result.series) == activities["activity_type"].nunique()


def test_non_numeric_value(activities):
    result = execute_box_plot(activities, _params(value_col="activity_type"))
    assert isinstance(result, ToolError)
    assert "numeric" in result.reason.lower()


def test_missing_value_col(activities):
    result = execute_box_plot(activities, _params(value_col="nope"))
    assert isinstance(result, ToolError)


def test_group_too_many():
    df = pd.DataFrame({"v": list(range(50)), "g": [f"g{i}" for i in range(50)]})
    result = execute_box_plot(df, _params(value_col="v", group_col="g"))
    assert isinstance(result, ToolError)
```

- [ ] **Step 2: Run tests, expect failure**

```bash
pytest tests/unit/test_executors_box.py -v
```

Expected: ImportError.

- [ ] **Step 3: Add executor**

Append to `src/api/chart_executor.py` (before `TOOL_EXECUTORS`):

```python
def _box_stats(values: pd.Series) -> dict:
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lo = max(values.min(), q1 - 1.5 * iqr)
    hi = min(values.max(), q3 + 1.5 * iqr)
    outliers = values[(values < lo) | (values > hi)].tolist()
    return {
        "min": float(lo), "q1": float(q1), "median": float(median),
        "q3": float(q3), "max": float(hi),
        "outliers": [float(v) for v in outliers],
    }


def execute_box_plot(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    value_col = params["value_col"]
    group_col = params.get("group_col")
    title = params["title"]
    intent = params["intent"]

    if value_col not in df.columns:
        return _err(f"value_col='{value_col}' is not a column.")
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        return _err(f"value_col='{value_col}' is not numeric. Box plots need a numeric value column.")

    values = pd.to_numeric(df[value_col], errors="coerce").dropna()
    if len(values) < 5:
        return _err(f"'{value_col}' has only {len(values)} non-null values; need at least 5 for a meaningful box plot.")

    if group_col is None:
        return ChartSpec(
            kind="box",
            title=title,
            intent=intent,
            series=[{"name": value_col, **_box_stats(values)}],
            x_label=value_col,
            y_label="",
            x_display_type="category",
            y_display_type="number",
            source_columns=[value_col],
            data_point_count=int(len(values)),
        )

    if group_col not in df.columns:
        return _err(f"group_col='{group_col}' is not a column.")
    if df[group_col].nunique() > MAX_CATEGORIES:
        return _err(f"group_col='{group_col}' has {df[group_col].nunique()} unique values; max is {MAX_CATEGORIES}.")

    work = df[[value_col, group_col]].dropna()
    series_list: list[dict] = []
    for gv, sub in work.groupby(group_col):
        v = pd.to_numeric(sub[value_col], errors="coerce").dropna()
        if len(v) < 5:
            continue
        series_list.append({"name": str(gv), **_box_stats(v)})

    if not series_list:
        return _err(f"No group in '{group_col}' has ≥5 non-null values for '{value_col}'.")

    return ChartSpec(
        kind="box",
        title=title,
        intent=intent,
        series=series_list,
        x_label=group_col,
        y_label=value_col,
        x_display_type="category",
        y_display_type="number",
        source_columns=[value_col, group_col],
        data_point_count=int(len(work)),
    )
```

Add to `TOOL_EXECUTORS`:

```python
    "box_plot": execute_box_plot,
```

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest tests/unit/test_executors_box.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/chart_executor.py tests/unit/test_executors_box.py
git commit -m "feat: box_plot executor with grouped and ungrouped modes"
```

---

### Task 14: heatmap_chart executor

**Files:**
- Modify: `src/api/chart_executor.py`
- Create: `tests/unit/test_executors_heatmap.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/unit/test_executors_heatmap.py`:

```python
import pandas as pd
import pytest
from chart_executor import execute_heatmap_chart
from schemas import ChartSpec, ToolError


def test_correlation_mode_happy_path():
    df = pd.DataFrame({
        "a": list(range(20)),
        "b": [i * 2 for i in range(20)],
        "c": [20 - i for i in range(20)],
        "label": ["x"] * 20,
    })
    params = {"mode": "correlation", "title": "t", "intent": "i"}
    result = execute_heatmap_chart(df, params)
    assert isinstance(result, ChartSpec)
    assert result.kind == "heatmap"
    assert result.series is not None
    assert len(result.x) == 3
    assert len(result.y) == 3


def test_correlation_mode_needs_two_numeric():
    df = pd.DataFrame({"a": list(range(5)), "label": ["x"] * 5})
    params = {"mode": "correlation", "title": "t", "intent": "i"}
    result = execute_heatmap_chart(df, params)
    assert isinstance(result, ToolError)
    assert "numeric" in result.reason.lower()


def test_pivot_mode_happy_path():
    df = pd.DataFrame({
        "row_cat": ["a", "a", "b", "b", "c", "c"],
        "col_cat": ["x", "y", "x", "y", "x", "y"],
        "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    params = {
        "mode": "pivot", "title": "t", "intent": "i",
        "row_col": "row_cat", "col_col": "col_cat",
        "value_col": "value", "agg": "sum",
    }
    result = execute_heatmap_chart(df, params)
    assert isinstance(result, ChartSpec)
    assert sorted(result.x) == ["x", "y"]
    assert sorted(result.y) == ["a", "b", "c"]


def test_pivot_mode_count():
    df = pd.DataFrame({
        "row_cat": ["a", "a", "b", "b"],
        "col_cat": ["x", "y", "x", "y"],
    })
    params = {
        "mode": "pivot", "title": "t", "intent": "i",
        "row_col": "row_cat", "col_col": "col_cat", "agg": "count",
    }
    result = execute_heatmap_chart(df, params)
    assert isinstance(result, ChartSpec)


def test_invalid_mode():
    df = pd.DataFrame({"a": [1, 2, 3]})
    params = {"mode": "wrong", "title": "t", "intent": "i"}
    result = execute_heatmap_chart(df, params)
    assert isinstance(result, ToolError)


def test_pivot_missing_value_col_for_non_count():
    df = pd.DataFrame({"r": ["a", "b"], "c": ["x", "y"]})
    params = {
        "mode": "pivot", "title": "t", "intent": "i",
        "row_col": "r", "col_col": "c", "agg": "sum",
    }
    result = execute_heatmap_chart(df, params)
    assert isinstance(result, ToolError)
    assert "value_col" in result.reason.lower()
```

- [ ] **Step 2: Run tests, expect failure**

```bash
pytest tests/unit/test_executors_heatmap.py -v
```

Expected: ImportError.

- [ ] **Step 3: Add executor**

Append to `src/api/chart_executor.py` (before `TOOL_EXECUTORS`):

```python
_HEATMAP_AGGS = {"sum", "mean", "count"}


def execute_heatmap_chart(df: pd.DataFrame, params: dict) -> ChartSpec | ToolError:
    mode = params["mode"]
    title = params["title"]
    intent = params["intent"]

    if mode == "correlation":
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) < 2:
            return _err(f"Correlation heatmap needs ≥2 numeric columns; found {len(numeric_cols)}.")
        corr = df[numeric_cols].corr(numeric_only=True).round(3)
        series = []
        for i, row in enumerate(numeric_cols):
            for j, col in enumerate(numeric_cols):
                v = corr.loc[row, col]
                if pd.notna(v):
                    series.append({"row": row, "col": col, "value": float(v)})
        return ChartSpec(
            kind="heatmap",
            title=title,
            intent=intent,
            x=numeric_cols,
            y=numeric_cols,
            series=series,
            x_label="",
            y_label="",
            x_display_type="category",
            y_display_type="number",
            source_columns=numeric_cols,
            data_point_count=int(df.shape[0]),
        )

    if mode != "pivot":
        return _err(f"mode='{mode}' is not allowed. Allowed: 'correlation', 'pivot'.")

    row_col = params.get("row_col")
    col_col = params.get("col_col")
    agg = params.get("agg", "count")
    value_col = params.get("value_col")

    if agg not in _HEATMAP_AGGS:
        return _err(f"agg='{agg}' is not allowed for pivot. Allowed: {sorted(_HEATMAP_AGGS)}.")
    if not row_col or row_col not in df.columns:
        return _err(f"row_col='{row_col}' is not a column.")
    if not col_col or col_col not in df.columns:
        return _err(f"col_col='{col_col}' is not a column.")
    if agg != "count":
        if not value_col:
            return _err("value_col is required when agg is not 'count'.")
        if value_col not in df.columns:
            return _err(f"value_col='{value_col}' is not a column.")
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            return _err(f"value_col='{value_col}' is not numeric.")

    cols_needed = [row_col, col_col] + ([value_col] if agg != "count" else [])
    work = df[cols_needed].dropna()

    if agg == "count":
        pivot = work.groupby([row_col, col_col]).size().unstack(fill_value=0)
    else:
        pivot = work.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc=agg, fill_value=0)

    if pivot.shape[0] > MAX_CATEGORIES or pivot.shape[1] > MAX_CATEGORIES:
        return _err(f"Heatmap dimensions {pivot.shape} exceed {MAX_CATEGORIES} × {MAX_CATEGORIES} max.")

    series = []
    for r in pivot.index:
        for c in pivot.columns:
            series.append({"row": str(r), "col": str(c), "value": float(pivot.loc[r, c])})

    return ChartSpec(
        kind="heatmap",
        title=title,
        intent=intent,
        x=[str(c) for c in pivot.columns.tolist()],
        y=[str(r) for r in pivot.index.tolist()],
        series=series,
        x_label=col_col,
        y_label=row_col,
        x_display_type="category",
        y_display_type="category",
        source_columns=cols_needed,
        data_point_count=int(len(work)),
    )
```

Add to `TOOL_EXECUTORS`:

```python
    "heatmap_chart": execute_heatmap_chart,
```

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest tests/unit/test_executors_heatmap.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/chart_executor.py tests/unit/test_executors_heatmap.py
git commit -m "feat: heatmap_chart executor with correlation and pivot modes"
```

---

## Phase 3 — Tool definitions and prompts

### Task 15: chart_tools.py (8 Anthropic tool schemas)

**Files:**
- Create: `src/api/chart_tools.py`
- Create: `src/api/prompts/selection_system.txt`
- Create: `src/api/prompts/narrative_system.txt`
- Create: `tests/unit/test_chart_tools.py`

- [ ] **Step 1: Write the failing test**

Write `tests/unit/test_chart_tools.py`:

```python
from chart_tools import CHART_TOOLS, NARRATIVE_TOOL


def test_all_eight_tools_present():
    names = {t["name"] for t in CHART_TOOLS}
    assert names == {
        "frequency_bar_chart",
        "aggregation_bar_chart",
        "histogram_chart",
        "scatter_chart",
        "line_chart",
        "pie_chart",
        "box_plot",
        "heatmap_chart",
    }


def test_each_tool_has_required_shape():
    for tool in CHART_TOOLS:
        assert "name" in tool
        assert "description" in tool
        assert "input_schema" in tool
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        assert "title" in schema["properties"]
        assert "intent" in schema["properties"]
        assert "title" in schema["required"]
        assert "intent" in schema["required"]


def test_frequency_tool_signature():
    tool = next(t for t in CHART_TOOLS if t["name"] == "frequency_bar_chart")
    props = tool["input_schema"]["properties"]
    assert "column" in props
    assert tool["input_schema"]["required"] == ["column", "title", "intent"]


def test_aggregation_tool_excludes_count():
    tool = next(t for t in CHART_TOOLS if t["name"] == "aggregation_bar_chart")
    agg = tool["input_schema"]["properties"]["agg"]
    assert "count" not in agg["enum"]
    assert "sum" in agg["enum"]


def test_line_tool_includes_count():
    tool = next(t for t in CHART_TOOLS if t["name"] == "line_chart")
    agg = tool["input_schema"]["properties"]["agg"]
    assert "count" in agg["enum"]


def test_narrative_tool_shape():
    assert NARRATIVE_TOOL["name"] == "submit_narrative"
    props = NARRATIVE_TOOL["input_schema"]["properties"]
    assert "summary" in props
    assert "captions" in props
    assert "data_quality" in props
```

- [ ] **Step 2: Run test, expect failure**

```bash
pytest tests/unit/test_chart_tools.py -v
```

Expected: ImportError.

- [ ] **Step 3: Create prompts**

Write `src/api/prompts/selection_system.txt`:

```
You are a data analyst. Your job: pick 5–7 charts that tell the most useful story about the user's data.

You will be given a structured profile of the data — column roles, basic stats, correlations, and any anomalies. You will not see the raw rows.

Call chart tools in parallel — one tool call per chart you want to render. Each tool has a strict schema; only call tools with the parameters in their schema.

Rules:
- Pick 5–7 charts total. Fewer than 3 is too few; more than 10 is overwhelming.
- Never use a column with role="identifier" as a metric (it's an ID, not a value).
- Never use a column with role="unusable".
- If the profile mentions outliers or negative values, prefer median over mean.
- Vary chart kinds — frequencies, distributions, comparisons, trends, correlations.
- Every chart needs an `intent` field: one sentence about what this chart shows and why it's interesting.
- Reference anomalies in your `intent` when they're relevant (e.g., "Using median because the profile flagged a 600-minute outlier.").

Return only tool calls. No prose.
```

Write `src/api/prompts/narrative_system.txt`:

```
You are a data analyst writing the narrative for a data report.

You will be given:
- A profile of the user's data
- The 5–7 charts that were generated, each with its `intent` and a summary of the data it shows
- Any data-quality anomalies flagged during profiling

Call the `submit_narrative` tool exactly once with three fields:

- `summary`: 2–3 short paragraphs (under 250 words total). What is this data about? What are the most interesting findings across the charts? Don't list the charts mechanically — synthesize.
- `captions`: one 1–2 sentence caption per chart, IN THE SAME ORDER as the charts you were given. Each caption should call out what's notable in the actual data (use the data summary you were given).
- `data_quality`: zero or more notes for the user about data issues (negative values, missing data, suspicious outliers). Each note should be one sentence, plain English. Empty array if no issues.

Do not output prose outside the tool call.
```

- [ ] **Step 4: Implement chart_tools.py**

Write `src/api/chart_tools.py`:

```python
"""Anthropic tool definitions for chart selection and narrative."""


def _t(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


_TITLE_INTENT = {
    "title": {"type": "string", "description": "Short chart title for the user."},
    "intent": {"type": "string", "description": "One-sentence rationale for this chart."},
}


CHART_TOOLS: list[dict] = [
    _t(
        "frequency_bar_chart",
        "Bar chart of counts per category. Use for: 'how many rows fall in each category'.",
        {
            "column": {"type": "string", "description": "Categorical column to count by."},
            **_TITLE_INTENT,
        },
        ["column", "title", "intent"],
    ),
    _t(
        "aggregation_bar_chart",
        "Bar chart of an aggregation of a numeric column grouped by a categorical column. "
        "Use for: 'sum/mean/median/min/max of X by Y'. Does NOT support count — use frequency_bar_chart instead.",
        {
            "value_col": {"type": "string", "description": "Numeric column to aggregate."},
            "group_col": {"type": "string", "description": "Categorical column to group by."},
            "agg": {"type": "string", "enum": ["sum", "mean", "median", "min", "max"]},
            **_TITLE_INTENT,
        },
        ["value_col", "group_col", "agg", "title", "intent"],
    ),
    _t(
        "histogram_chart",
        "Histogram of a numeric column's distribution. Use for: 'how is X distributed'.",
        {
            "column": {"type": "string", "description": "Numeric column to bin."},
            **_TITLE_INTENT,
        },
        ["column", "title", "intent"],
    ),
    _t(
        "scatter_chart",
        "Scatter plot of two numeric columns. Use for: 'is there a relationship between X and Y'. "
        "Optional color_by categorical column.",
        {
            "x_col": {"type": "string"},
            "y_col": {"type": "string"},
            "color_by": {"type": "string", "description": "Optional categorical column for coloring points."},
            **_TITLE_INTENT,
        },
        ["x_col", "y_col", "title", "intent"],
    ),
    _t(
        "line_chart",
        "Line chart over time. Use for trends. Aggregates a value column (or counts) by a time granularity.",
        {
            "date_col": {"type": "string"},
            "value_col": {"type": "string", "description": "Numeric column to aggregate (ignored when agg='count')."},
            "agg": {"type": "string", "enum": ["count", "sum", "mean", "median", "min", "max"]},
            "granularity": {"type": "string", "enum": ["day", "week", "month", "quarter", "year"]},
            "group_by": {"type": "string", "description": "Optional categorical column for multiple lines."},
            **_TITLE_INTENT,
        },
        ["date_col", "value_col", "agg", "granularity", "title", "intent"],
    ),
    _t(
        "pie_chart",
        "Pie chart of composition. Best for ≤8 categories. Larger sets get rolled into 'Other'.",
        {
            "category_col": {"type": "string"},
            "value_col": {"type": "string", "description": "Numeric column to aggregate (omit when agg='count')."},
            "agg": {"type": "string", "enum": ["sum", "mean", "count"]},
            **_TITLE_INTENT,
        },
        ["category_col", "agg", "title", "intent"],
    ),
    _t(
        "box_plot",
        "Box plot of a numeric column, optionally grouped by a categorical column.",
        {
            "value_col": {"type": "string"},
            "group_col": {"type": "string", "description": "Optional categorical column for groups."},
            **_TITLE_INTENT,
        },
        ["value_col", "title", "intent"],
    ),
    _t(
        "heatmap_chart",
        "Heatmap. Two modes: 'correlation' (correlation matrix of all numeric columns) "
        "or 'pivot' (aggregation of value_col by row_col × col_col).",
        {
            "mode": {"type": "string", "enum": ["correlation", "pivot"]},
            "row_col": {"type": "string", "description": "Required when mode='pivot'."},
            "col_col": {"type": "string", "description": "Required when mode='pivot'."},
            "value_col": {"type": "string", "description": "Required when mode='pivot' and agg != 'count'."},
            "agg": {"type": "string", "enum": ["sum", "mean", "count"], "description": "Required when mode='pivot'."},
            **_TITLE_INTENT,
        },
        ["mode", "title", "intent"],
    ),
]


NARRATIVE_TOOL: dict = {
    "name": "submit_narrative",
    "description": "Submit the final report narrative.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "2-3 paragraph executive summary of what's in the data.",
            },
            "captions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One caption per chart, in the same order as the input charts.",
            },
            "data_quality": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Notes about data issues. Empty array if none.",
            },
        },
        "required": ["summary", "captions", "data_quality"],
        "additionalProperties": False,
    },
}
```

- [ ] **Step 5: Run tests, expect pass**

```bash
pytest tests/unit/test_chart_tools.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/api/chart_tools.py src/api/prompts/ tests/unit/test_chart_tools.py
git commit -m "feat: 8 Anthropic chart tools + narrative tool + system prompts"
```

---

## Phase 4 — Report generation

### Task 16: Fake-Claude helper for integration tests

**Files:**
- Create: `tests/helpers/fake_claude.py`

- [ ] **Step 1: Implement the fake Claude responder**

Write `tests/helpers/fake_claude.py`:

```python
"""Mock-able Claude response factory for integration tests.

Use:
    fake = FakeClaude(scripted_responses=[
        {"tool_calls": [{"name": "frequency_bar_chart", "input": {...}}]},
        {"tool_use": "submit_narrative", "input": {"summary": "...", "captions": [...], "data_quality": []}},
    ])
    client.messages_create = fake  # monkey-patch ClaudeClient
"""
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock


@dataclass
class _ScriptedResponse:
    tool_calls: list[dict] = field(default_factory=list)
    text: str = ""


class FakeClaude:
    """Callable that returns canned responses in sequence."""

    def __init__(self, scripted: list[dict]):
        self.scripted = scripted
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        idx = len(self.calls) - 1
        if idx >= len(self.scripted):
            raise AssertionError(f"FakeClaude received call #{idx+1} but only {len(self.scripted)} scripted")
        scripted = self.scripted[idx]

        content_blocks: list[Any] = []
        for tc in scripted.get("tool_calls", []):
            block = MagicMock()
            block.type = "tool_use"
            block.id = tc.get("id", f"tu_{idx}_{len(content_blocks)}")
            block.name = tc["name"]
            block.input = tc["input"]
            content_blocks.append(block)
        if scripted.get("text"):
            block = MagicMock()
            block.type = "text"
            block.text = scripted["text"]
            content_blocks.append(block)

        resp = MagicMock()
        resp.content = content_blocks
        resp.model = "claude-haiku-4-5-20251001"
        resp.usage = MagicMock(
            input_tokens=100, output_tokens=50,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
        )
        return resp


def tool_use(name: str, input_dict: dict, id_: str = None) -> dict:
    """Convenience helper for building scripted tool calls."""
    out = {"name": name, "input": input_dict}
    if id_:
        out["id"] = id_
    return out
```

- [ ] **Step 2: Commit (no test needed yet; tests will exercise it in Task 17)**

```bash
git add tests/helpers/fake_claude.py
git commit -m "test: FakeClaude helper for integration tests"
```

---

### Task 17: fallback.py heuristic chart picker

**Files:**
- Create: `src/api/fallback.py`
- Create: `tests/unit/test_fallback.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/unit/test_fallback.py`:

```python
import pandas as pd
import pytest
from fallback import pick_fallback_charts
from profile import profile_dataframe
from schemas import ChartSpec


def test_picks_at_least_three_for_normal_data(activities):
    profile = profile_dataframe(activities)
    specs = pick_fallback_charts(profile, activities)
    assert len(specs) >= 3


def test_uses_frequency_for_categoricals(sales):
    profile = profile_dataframe(sales)
    specs = pick_fallback_charts(profile, sales)
    kinds = {s.kind for s in specs}
    assert "bar" in kinds


def test_uses_histogram_for_numerics(activities):
    profile = profile_dataframe(activities)
    specs = pick_fallback_charts(profile, activities)
    kinds = {s.kind for s in specs}
    assert "histogram" in kinds or "bar" in kinds


def test_uses_scatter_for_correlated_pair():
    df = pd.DataFrame({
        "x": list(range(20)),
        "y": [i * 2 + 1 for i in range(20)],
        "label": ["a"] * 20,
    })
    profile = profile_dataframe(df)
    specs = pick_fallback_charts(profile, df)
    kinds = {s.kind for s in specs}
    assert "scatter" in kinds


def test_handles_degenerate(degenerate):
    profile = profile_dataframe(degenerate)
    specs = pick_fallback_charts(profile, degenerate)
    # Degenerate data may not produce any charts; that's OK
    assert isinstance(specs, list)


def test_specs_marked_as_fallback(activities):
    profile = profile_dataframe(activities)
    specs = pick_fallback_charts(profile, activities)
    if specs:
        assert all("fallback" in s.intent.lower() for s in specs)
```

- [ ] **Step 2: Run tests, expect failure**

```bash
pytest tests/unit/test_fallback.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement fallback.py**

Write `src/api/fallback.py`:

```python
"""Heuristic chart picker used when Claude pass #1 produces fewer than 3 usable charts."""
from typing import Any
import pandas as pd
from schemas import ChartSpec, DataProfile
from chart_executor import (
    execute_frequency_bar_chart,
    execute_histogram_chart,
    execute_scatter_chart,
)


def pick_fallback_charts(profile: DataProfile, df: pd.DataFrame, max_charts: int = 5) -> list[ChartSpec]:
    """Pure heuristic chart selection.

    Rules:
    - frequency_bar_chart for top 2 categorical columns (cardinality 2–30)
    - histogram_chart for top 2 numeric columns
    - scatter_chart for the strongest |correlation| ≥ 0.3 pair
    """
    specs: list[ChartSpec] = []
    intent = "fallback: Claude pass #1 didn't pick this; chosen by heuristic."

    # 1. Top 2 categoricals
    cats = [c for c in profile.columns if c.role == "categorical" and 2 <= c.cardinality <= 30]
    for col in cats[:2]:
        result = execute_frequency_bar_chart(df, {
            "column": col.name,
            "title": f"{col.name} — distribution",
            "intent": intent,
        })
        if isinstance(result, ChartSpec):
            specs.append(result)
            if len(specs) >= max_charts:
                return specs

    # 2. Top 2 numerics
    nums = [c for c in profile.columns if c.role == "numeric"]
    for col in nums[:2]:
        result = execute_histogram_chart(df, {
            "column": col.name,
            "title": f"{col.name} — distribution",
            "intent": intent,
        })
        if isinstance(result, ChartSpec):
            specs.append(result)
            if len(specs) >= max_charts:
                return specs

    # 3. Strongest correlation as scatter
    if profile.correlations:
        best_pair, _ = max(profile.correlations.items(), key=lambda kv: abs(kv[1]))
        x_col, y_col = best_pair.split("||")
        result = execute_scatter_chart(df, {
            "x_col": x_col,
            "y_col": y_col,
            "title": f"{x_col} vs {y_col}",
            "intent": intent,
        })
        if isinstance(result, ChartSpec):
            specs.append(result)

    return specs
```

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest tests/unit/test_fallback.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/fallback.py tests/unit/test_fallback.py
git commit -m "feat: heuristic fallback chart picker"
```

---

### Task 18: report_generator.py — pass #1 happy path

**Files:**
- Create: `src/api/report_generator.py`
- Create: `tests/integration/test_pipeline_happy.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/integration/test_pipeline_happy.py`:

```python
import pandas as pd
import pytest
from unittest.mock import MagicMock
from report_generator import ReportGenerator
from profile import profile_dataframe
from schemas import ChartSpec
from tests.helpers.fake_claude import FakeClaude, tool_use


def _make_generator(df, fake):
    profile = profile_dataframe(df)
    client = MagicMock()
    client.messages_create = fake
    gen = ReportGenerator(profile=profile, df=df, claude=client,
                          model_selection="m1", model_narrative="m2")
    return gen


def test_happy_path_returns_chart_specs(activities):
    fake = FakeClaude([
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "activity_type", "title": "Type counts", "intent": "show mix"}),
            tool_use("histogram_chart", {
                "column": "duration_minutes", "title": "Duration", "intent": "show spread"}),
            tool_use("aggregation_bar_chart", {
                "value_col": "duration_minutes", "group_col": "activity_type",
                "agg": "median", "title": "Median by type", "intent": "compare"}),
        ]},
    ])
    gen = _make_generator(activities, fake)
    specs = gen.generate_charts()
    assert len(specs) == 3
    assert all(isinstance(s, ChartSpec) for s in specs)


def test_caps_at_ten_charts(activities):
    """Even if Claude returns 15 tool calls, we only keep the first 10."""
    fake = FakeClaude([
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "activity_type", "title": f"t{i}", "intent": f"i{i}"})
            for i in range(15)
        ]},
    ])
    gen = _make_generator(activities, fake)
    specs = gen.generate_charts()
    assert len(specs) <= 10
```

- [ ] **Step 2: Run tests, expect failure**

```bash
pytest tests/integration/test_pipeline_happy.py -v
```

Expected: ImportError on `report_generator`.

- [ ] **Step 3: Implement report_generator.py (pass #1 only for now)**

Write `src/api/report_generator.py`:

```python
"""Two-pass report generation orchestrator.

Pass #1: chart selection via parallel tool calls. Tool errors get one retry round
         with structured tool_result messages, then fall back to heuristics if
         we still have fewer than 3 charts.
Pass #2: narrative generation via single forced submit_narrative call.
"""
import logging
import os
from pathlib import Path
from typing import Any
import pandas as pd
from schemas import ChartSpec, Report, ChartWithCaption, ReportNarrative, ToolError, DataProfile
from chart_tools import CHART_TOOLS, NARRATIVE_TOOL
from chart_executor import TOOL_EXECUTORS
from fallback import pick_fallback_charts


MAX_CHARTS = 10
MIN_CHARTS_FOR_NO_FALLBACK = 3

_PROMPT_DIR = Path(__file__).parent / "prompts"
SELECTION_SYSTEM = (_PROMPT_DIR / "selection_system.txt").read_text()
NARRATIVE_SYSTEM = (_PROMPT_DIR / "narrative_system.txt").read_text()


class ReportGenerator:
    def __init__(
        self,
        profile: DataProfile,
        df: pd.DataFrame,
        claude: Any,
        model_selection: str,
        model_narrative: str,
    ):
        self.profile = profile
        self.df = df
        self.claude = claude
        self.model_selection = model_selection
        self.model_narrative = model_narrative

    def generate_charts(self) -> list[ChartSpec]:
        """Pass #1: tool-use selection + (retry stub for now) + fallback."""
        specs, errors = self._call_selection()
        if len(specs) < MIN_CHARTS_FOR_NO_FALLBACK:
            specs.extend(pick_fallback_charts(self.profile, self.df, max_charts=MAX_CHARTS - len(specs)))
        return specs[:MAX_CHARTS]

    def _call_selection(self) -> tuple[list[ChartSpec], list[dict]]:
        response = self.claude.messages_create(
            model=self.model_selection,
            max_tokens=4096,
            system=SELECTION_SYSTEM,
            tools=CHART_TOOLS,
            messages=[{"role": "user", "content": self.profile.to_text()}],
            cache_static=True,
        )
        return self._execute_tool_calls(response.content)

    def _execute_tool_calls(self, content_blocks: list[Any]) -> tuple[list[ChartSpec], list[dict]]:
        specs: list[ChartSpec] = []
        errors: list[dict] = []
        for block in content_blocks:
            if getattr(block, "type", None) != "tool_use":
                continue
            executor = TOOL_EXECUTORS.get(block.name)
            if executor is None:
                errors.append({"id": block.id, "reason": f"unknown tool '{block.name}'"})
                continue
            result = executor(self.df, block.input)
            if isinstance(result, ToolError):
                errors.append({"id": block.id, "reason": result.reason})
                logging.warning("[GEN] tool '%s' error: %s", block.name, result.reason)
            else:
                specs.append(result)
                if len(specs) >= MAX_CHARTS:
                    break
        return specs, errors
```

- [ ] **Step 4: Run tests, expect pass**

```bash
pytest tests/integration/test_pipeline_happy.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/report_generator.py tests/integration/test_pipeline_happy.py
git commit -m "feat: report generator pass #1 with fallback (no retry yet)"
```

---

### Task 19: Tool-error retry round

**Files:**
- Modify: `src/api/report_generator.py`
- Create: `tests/integration/test_pipeline_retry.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/integration/test_pipeline_retry.py`:

```python
import pandas as pd
import pytest
from unittest.mock import MagicMock
from report_generator import ReportGenerator
from profile import profile_dataframe
from schemas import ChartSpec
from tests.helpers.fake_claude import FakeClaude, tool_use


def _make_generator(df, fake):
    profile = profile_dataframe(df)
    client = MagicMock()
    client.messages_create = fake
    return ReportGenerator(profile=profile, df=df, claude=client,
                           model_selection="m1", model_narrative="m2")


def test_retry_recovers_from_bad_column(activities):
    fake = FakeClaude([
        # First call: one tool fails (bad column)
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "nope", "title": "Bad", "intent": "fail"}, id_="bad1"),
            tool_use("frequency_bar_chart", {
                "column": "activity_type", "title": "Good", "intent": "good"}, id_="good1"),
        ]},
        # Retry call: Claude corrects the bad one
        {"tool_calls": [
            tool_use("histogram_chart", {
                "column": "duration_minutes", "title": "Fixed", "intent": "fixed"}, id_="fix1"),
        ]},
    ])
    gen = _make_generator(activities, fake)
    specs = gen.generate_charts()
    titles = {s.title for s in specs}
    assert "Good" in titles
    assert "Fixed" in titles


def test_retry_failures_are_dropped(activities):
    fake = FakeClaude([
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "nope1", "title": "t1", "intent": "i"}, id_="b1"),
            tool_use("frequency_bar_chart", {
                "column": "activity_type", "title": "Good", "intent": "good"}, id_="g1"),
        ]},
        # Retry returns more broken tools
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "still_bad", "title": "t2", "intent": "i"}, id_="b2"),
        ]},
    ])
    gen = _make_generator(activities, fake)
    specs = gen.generate_charts()
    titles = {s.title for s in specs}
    assert "Good" in titles
    assert "t1" not in titles
    assert "t2" not in titles


def test_no_retry_when_no_errors(activities):
    fake = FakeClaude([
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "activity_type", "title": "Good", "intent": "good"}),
            tool_use("histogram_chart", {
                "column": "duration_minutes", "title": "Hist", "intent": "spread"}),
            tool_use("aggregation_bar_chart", {
                "value_col": "duration_minutes", "group_col": "activity_type",
                "agg": "mean", "title": "Mean", "intent": "compare"}),
        ]},
        # If retry happens, this will be hit and tests fail
    ])
    gen = _make_generator(activities, fake)
    specs = gen.generate_charts()
    assert len(specs) == 3
    assert len(fake.calls) == 1
```

- [ ] **Step 2: Run tests, expect mixed (some pass, retry-specific ones fail)**

```bash
pytest tests/integration/test_pipeline_retry.py -v
```

Expected: 1 pass (no_retry_when_no_errors), 2 fail (no retry implemented yet).

- [ ] **Step 3: Add retry logic to report_generator.py**

Replace the `generate_charts` and `_call_selection` methods in `src/api/report_generator.py` with:

```python
    def generate_charts(self) -> list[ChartSpec]:
        """Pass #1: tool-use selection + 1 retry round + fallback."""
        specs, errors, response_content = self._call_selection_initial()

        if errors:
            specs2, _ = self._call_selection_retry(response_content, errors)
            specs.extend(specs2)

        if len(specs) < MIN_CHARTS_FOR_NO_FALLBACK:
            specs.extend(pick_fallback_charts(
                self.profile, self.df, max_charts=MAX_CHARTS - len(specs),
            ))

        return specs[:MAX_CHARTS]

    def _call_selection_initial(self) -> tuple[list[ChartSpec], list[dict], list[Any]]:
        response = self.claude.messages_create(
            model=self.model_selection,
            max_tokens=4096,
            system=SELECTION_SYSTEM,
            tools=CHART_TOOLS,
            messages=[{"role": "user", "content": self.profile.to_text()}],
            cache_static=True,
        )
        specs, errors = self._execute_tool_calls(response.content)
        return specs, errors, response.content

    def _call_selection_retry(
        self, prior_content: list[Any], errors: list[dict],
    ) -> tuple[list[ChartSpec], list[dict]]:
        tool_results = []
        for err in errors:
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": err["id"],
                "content": err["reason"],
                "is_error": True,
            })

        messages = [
            {"role": "user", "content": self.profile.to_text()},
            {"role": "assistant", "content": _serialize_content(prior_content)},
            {"role": "user", "content": tool_results},
        ]

        response = self.claude.messages_create(
            model=self.model_selection,
            max_tokens=4096,
            system=SELECTION_SYSTEM,
            tools=CHART_TOOLS,
            messages=messages,
            cache_static=True,
        )
        return self._execute_tool_calls(response.content)
```

And at the top of the file (after imports), add the helper:

```python
def _serialize_content(blocks: list[Any]) -> list[dict]:
    """Convert response.content blocks back into request-shape dicts."""
    out: list[dict] = []
    for b in blocks:
        if getattr(b, "type", None) == "tool_use":
            out.append({
                "type": "tool_use",
                "id": b.id,
                "name": b.name,
                "input": b.input,
            })
        elif getattr(b, "type", None) == "text":
            out.append({"type": "text", "text": b.text})
    return out
```

Delete the old `_call_selection` method (replaced by the two new ones).

- [ ] **Step 4: Run tests, expect all pass**

```bash
pytest tests/integration/test_pipeline_retry.py tests/integration/test_pipeline_happy.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/api/report_generator.py tests/integration/test_pipeline_retry.py
git commit -m "feat: retry round for tool errors in pass #1"
```

---

### Task 20: Pass #2 narrative + complete report assembly

**Files:**
- Modify: `src/api/report_generator.py`
- Create: `tests/integration/test_pipeline_fallback.py`
- Modify: `tests/integration/test_pipeline_happy.py` — add narrative test

- [ ] **Step 1: Write the failing tests**

Write `tests/integration/test_pipeline_fallback.py`:

```python
import pandas as pd
import pytest
from unittest.mock import MagicMock
from report_generator import ReportGenerator
from profile import profile_dataframe
from tests.helpers.fake_claude import FakeClaude, tool_use


def _make_generator(df, fake):
    profile = profile_dataframe(df)
    client = MagicMock()
    client.messages_create = fake
    return ReportGenerator(profile=profile, df=df, claude=client,
                           model_selection="m1", model_narrative="m2")


def test_fallback_when_claude_returns_nothing(activities):
    fake = FakeClaude([
        {"tool_calls": []},  # pass #1 first call: nothing
        # No retry call expected because there are no errors to send back.
        # Narrative is still called because we have fallback charts.
        {"tool_calls": [
            tool_use("submit_narrative", {
                "summary": "Auto summary.",
                "captions": ["c1", "c2", "c3"],
                "data_quality": [],
            })
        ]},
    ])
    gen = _make_generator(activities, fake)
    report = gen.build_report()
    assert len(report.charts) >= 3
    assert all("fallback" in c.spec.intent.lower() for c in report.charts)


def test_fallback_when_all_tool_calls_error(activities):
    fake = FakeClaude([
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "nope", "title": "x", "intent": "x"}, id_="e1"),
        ]},
        # Retry returns more errors
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "still_nope", "title": "x", "intent": "x"}, id_="e2"),
        ]},
        # Narrative
        {"tool_calls": [
            tool_use("submit_narrative", {
                "summary": "s", "captions": ["c1", "c2", "c3"], "data_quality": [],
            })
        ]},
    ])
    gen = _make_generator(activities, fake)
    report = gen.build_report()
    assert len(report.charts) >= 3
```

Append to `tests/integration/test_pipeline_happy.py`:

```python
def test_full_report_includes_narrative(activities):
    fake = FakeClaude([
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "activity_type", "title": "T1", "intent": "i1"}),
            tool_use("histogram_chart", {
                "column": "duration_minutes", "title": "T2", "intent": "i2"}),
            tool_use("aggregation_bar_chart", {
                "value_col": "duration_minutes", "group_col": "activity_type",
                "agg": "median", "title": "T3", "intent": "i3"}),
        ]},
        {"tool_calls": [
            tool_use("submit_narrative", {
                "summary": "Three findings emerged.",
                "captions": ["Caption A.", "Caption B.", "Caption C."],
                "data_quality": ["Note: some negative durations."],
            }),
        ]},
    ])
    gen = _make_generator(activities, fake)
    report = gen.build_report()
    assert report.summary == "Three findings emerged."
    assert len(report.charts) == 3
    assert report.charts[0].caption == "Caption A."
    assert "negative" in report.data_quality[0]


def test_narrative_template_fallback_on_failure(activities):
    """If pass #2 returns nothing usable, we fill in a template."""
    fake = FakeClaude([
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "activity_type", "title": "T1", "intent": "i1"}),
            tool_use("histogram_chart", {
                "column": "duration_minutes", "title": "T2", "intent": "i2"}),
            tool_use("aggregation_bar_chart", {
                "value_col": "duration_minutes", "group_col": "activity_type",
                "agg": "median", "title": "T3", "intent": "i3"}),
        ]},
        # Pass #2: no submit_narrative call (degraded)
        {"tool_calls": [], "text": "I cannot."},
    ])
    gen = _make_generator(activities, fake)
    report = gen.build_report()
    assert report.summary  # non-empty fallback
    assert len(report.charts) == 3
    # captions degrade to spec.intent
    assert report.charts[0].caption == "i1"
```

- [ ] **Step 2: Run tests, expect failures (build_report doesn't exist)**

```bash
pytest tests/integration/ -v
```

Expected: errors on `build_report`.

- [ ] **Step 3: Implement build_report + generate_narrative**

Add to `src/api/report_generator.py` (at the end of the `ReportGenerator` class):

```python
    def generate_narrative(self, charts: list[ChartSpec]) -> ReportNarrative:
        """Pass #2: forced submit_narrative tool call."""
        user_message = self._format_charts_for_narrative(charts)
        response = self.claude.messages_create(
            model=self.model_narrative,
            max_tokens=2048,
            system=NARRATIVE_SYSTEM,
            tools=[NARRATIVE_TOOL],
            tool_choice={"type": "tool", "name": "submit_narrative"},
            messages=[{"role": "user", "content": user_message}],
            cache_static=True,
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "submit_narrative":
                data = block.input
                return ReportNarrative(
                    summary=data.get("summary", ""),
                    captions=list(data.get("captions", [])),
                    data_quality=list(data.get("data_quality", [])),
                )
        return self._narrative_template_fallback(charts)

    def _format_charts_for_narrative(self, charts: list[ChartSpec]) -> str:
        lines = ["Profile:", self.profile.to_text(), "", "Charts to caption (in order):"]
        for i, c in enumerate(charts, 1):
            data_sample = self._summarize_chart_data(c)
            lines.append(f"{i}. [{c.kind}] {c.title}")
            lines.append(f"   Intent: {c.intent}")
            lines.append(f"   Data: {data_sample}")
        return "\n".join(lines)

    @staticmethod
    def _summarize_chart_data(spec: ChartSpec) -> str:
        if spec.series:
            return f"{len(spec.series)} series, e.g. {spec.series[0]}"[:200]
        if spec.x and spec.y:
            n = len(spec.x)
            sample_x = spec.x[:5]
            sample_y = spec.y[:5]
            return f"{n} points; sample x={sample_x} y={sample_y}"[:200]
        return f"{spec.data_point_count} data points"

    def _narrative_template_fallback(self, charts: list[ChartSpec]) -> ReportNarrative:
        return ReportNarrative(
            summary=f"Automated analysis of your data. The report contains {len(charts)} charts highlighting "
                    f"key patterns across the {self.profile.row_count} rows.",
            captions=[c.intent for c in charts],
            data_quality=list(self.profile.anomalies),
        )

    def build_report(self) -> Report:
        from datetime import datetime
        charts = self.generate_charts()
        narrative = self.generate_narrative(charts)

        captions = narrative.captions
        if len(captions) < len(charts):
            captions = captions + [c.intent for c in charts[len(captions):]]

        return Report(
            generated_at=datetime.utcnow().isoformat(),
            summary=narrative.summary or self._narrative_template_fallback(charts).summary,
            data_quality=narrative.data_quality,
            charts=[ChartWithCaption(spec=spec, caption=cap)
                    for spec, cap in zip(charts, captions)],
            metadata={
                "model_selection": self.model_selection,
                "model_narrative": self.model_narrative,
                "row_count": self.profile.row_count,
                "column_count": len(self.profile.columns),
            },
        )
```

- [ ] **Step 4: Run tests, expect all pass**

```bash
pytest tests/integration/ -v
```

Expected: 7 passed (3 happy + 2 retry + 2 fallback).

- [ ] **Step 5: Commit**

```bash
git add src/api/report_generator.py tests/integration/test_pipeline_fallback.py tests/integration/test_pipeline_happy.py
git commit -m "feat: pass #2 narrative + full report assembly with template fallback"
```

---

## Phase 5 — FastAPI endpoints

### Task 21: Replace main.py with new /generate-report and /report/{id}

**Files:**
- Modify: `src/api/main.py` (full rewrite)
- Create: `tests/integration/test_api_errors.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/integration/test_api_errors.py`:

```python
import io
import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


@pytest.fixture
def client(sales, monkeypatch):
    """Boot the app with a mock claude client and an in-memory redis stand-in."""
    from tests.helpers.fake_claude import FakeClaude, tool_use

    fake = FakeClaude([
        {"tool_calls": [
            tool_use("frequency_bar_chart", {
                "column": "region", "title": "Regions", "intent": "show mix"}),
            tool_use("aggregation_bar_chart", {
                "value_col": "revenue", "group_col": "region",
                "agg": "sum", "title": "Revenue by region", "intent": "show winners"}),
            tool_use("line_chart", {
                "date_col": "order_date", "value_col": "revenue",
                "agg": "sum", "granularity": "week",
                "title": "Weekly revenue", "intent": "trend"}),
        ]},
        {"tool_calls": [
            tool_use("submit_narrative", {
                "summary": "Sales report.",
                "captions": ["c1", "c2", "c3"],
                "data_quality": [],
            }),
        ]},
    ])

    fake_redis = {}

    class FakeRedis:
        def set(self, key, val, ex=None):
            fake_redis[key] = val
        def get(self, key):
            return fake_redis.get(key)

    from main import app, get_claude_client, get_redis
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=fake)
    app.dependency_overrides[get_redis] = lambda: FakeRedis()

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_happy_path_post_then_get(client, sales):
    resp = client.post(
        "/generate-report",
        files={"file": ("sales.csv", _csv_bytes(sales), "text/csv")},
    )
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    resp2 = client.get(f"/report/{session_id}")
    assert resp2.status_code == 200
    body = resp2.json()
    assert body["summary"] == "Sales report."
    assert len(body["charts"]) == 3


def test_rejects_non_csv_xlsx(client):
    resp = client.post(
        "/generate-report",
        files={"file": ("data.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 422


def test_rejects_oversize_file(client):
    big = b"a,b\n" + b"1,2\n" * 5_000_000   # ~25 MB
    resp = client.post(
        "/generate-report",
        files={"file": ("big.csv", big, "text/csv")},
    )
    assert resp.status_code == 422


def test_rejects_corrupt_csv(client):
    resp = client.post(
        "/generate-report",
        files={"file": ("bad.csv", b"\x00\x01\x02broken", "text/csv")},
    )
    assert resp.status_code == 422


def test_get_nonexistent_session(client):
    resp = client.get("/report/does-not-exist")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run tests, expect failures**

```bash
pytest tests/integration/test_api_errors.py -v
```

Expected: errors because main.py still has the old endpoints / imports.

- [ ] **Step 3: Rewrite main.py**

Overwrite `src/api/main.py`:

```python
"""FastAPI app — only /generate-report and /report/{session_id}."""
import io
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

import pandas as pd
import redis
from anthropic import APIStatusError
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from claude_client import ClaudeClient, RetryableBusy
from llm_config import MODEL_SELECTION, MODEL_NARRATIVE
from profile import profile_dataframe
from report_generator import ReportGenerator


load_dotenv()


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = (".csv", ".xlsx")
SESSION_TTL_SECONDS = 86400


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
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w", encoding="utf-8")],
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


def get_claude_client() -> ClaudeClient:
    global _claude_singleton
    if _claude_singleton is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _claude_singleton = ClaudeClient(api_key=api_key)
    return _claude_singleton


def get_redis():
    return redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


# ---- App -------------------------------------------------------------------

app = FastAPI(title="ChartSage v2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
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


# ---- Endpoints -------------------------------------------------------------

@app.post("/generate-report")
async def generate_report(
    file: UploadFile = File(...),
    claude: ClaudeClient = Depends(get_claude_client),
    r=Depends(get_redis),
):
    run_id, log_path = setup_run_logging()
    logging.info("Generating report for %s", file.filename)

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

    try:
        profile = profile_dataframe(df)
        gen = ReportGenerator(
            profile=profile, df=df, claude=claude,
            model_selection=MODEL_SELECTION, model_narrative=MODEL_NARRATIVE,
        )
        report = gen.build_report()
    except RetryableBusy:
        raise HTTPException(
            status_code=503,
            detail={"status": "busy", "message": "Claude is busy. Please retry in 30 seconds."},
        )
    except APIStatusError as e:
        logging.exception("Claude API error")
        raise HTTPException(status_code=502, detail=f"Upstream model error: {e}")
    except Exception as e:
        logging.exception("Report generation failed")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    session_id = uuid.uuid4().hex
    r.set(f"report:{session_id}", report.model_dump_json(), ex=SESSION_TTL_SECONDS)

    logging.info(
        "=== RUN SUMMARY ===\nrun_id: %s\nfile: %s\nrows: %d  cols: %d\n"
        "model_selection: %s\nmodel_narrative: %s\ncharts: %d\nresult: success",
        run_id, file.filename, df.shape[0], df.shape[1],
        MODEL_SELECTION, MODEL_NARRATIVE, len(report.charts),
    )

    return {"session_id": session_id}


@app.get("/report/{session_id}")
async def get_report(session_id: str, r=Depends(get_redis)):
    raw = r.get(f"report:{session_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Report not found or expired.")
    return JSONResponse(content=json.loads(raw))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 4: Run tests, expect all pass**

```bash
pytest tests/integration/test_api_errors.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Run full test suite to verify nothing broke**

```bash
pytest tests/unit/ tests/integration/ -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/api/main.py tests/integration/test_api_errors.py
git commit -m "feat: replace FastAPI app with /generate-report + /report/{id}"
```

---

## Phase 6 — Delete the dead backend

### Task 22: Remove old backend modules

**Files:**
- Delete: `src/api/bar_chart_processor.py`
- Delete: `src/api/chart_processing.py`
- Delete: `src/api/insight_prompt.txt`
- Delete: `src/api/bar_chart_prompt.txt`
- Delete: `src/api/log_viewer.py`
- Delete: `src/api/field_type_utils.py`
- Delete: `src/api/derived_fields.py`
- Delete: `src/api/api_config.json`
- Delete: `src/api/CHART_GENERATION_FIXES.md`
- Delete: `src/api/LOGGING_SYSTEM.md`
- Delete: `BAR_CHART_SYSTEM.md`
- Delete: `REFACTORING_SUMMARY.md`
- Delete: `cursor_rule_prompt_json_formatting.mdc`
- Delete: `src/api/temp/` (entire dir)
- Delete: `src/api/logs/*.log` (folder kept)
- Delete: `src/api/__pycache__/` (regenerated on next import)

- [ ] **Step 1: Verify these files are no longer imported anywhere**

Run:

```bash
grep -rn "from bar_chart_processor\|from chart_processing\|from derived_fields\|from field_type_utils\|import bar_chart_processor\|import chart_processing\|import derived_fields\|import field_type_utils" src/ tests/
```

Expected: no matches.

- [ ] **Step 2: Delete the files**

```bash
rm -f src/api/bar_chart_processor.py
rm -f src/api/chart_processing.py
rm -f src/api/insight_prompt.txt
rm -f src/api/bar_chart_prompt.txt
rm -f src/api/log_viewer.py
rm -f src/api/field_type_utils.py
rm -f src/api/derived_fields.py
rm -f src/api/api_config.json
rm -f src/api/CHART_GENERATION_FIXES.md
rm -f src/api/LOGGING_SYSTEM.md
rm -f BAR_CHART_SYSTEM.md
rm -f REFACTORING_SUMMARY.md
rm -f cursor_rule_prompt_json_formatting.mdc
rm -rf src/api/temp
rm -f src/api/logs/*.log
rm -rf src/api/__pycache__
```

- [ ] **Step 3: Run the full backend test suite to verify nothing broke**

```bash
pytest tests/unit/ tests/integration/ -v
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add -A src/api BAR_CHART_SYSTEM.md REFACTORING_SUMMARY.md cursor_rule_prompt_json_formatting.mdc 2>/dev/null || true
git commit -m "chore: delete old backend modules and historical docs"
```

---

## Phase 7 — Frontend rebuild

### Task 23: Shared format library

**Files:**
- Create: `src/app/lib/format.ts`

- [ ] **Step 1: Create the format helpers**

Write `src/app/lib/format.ts`:

```typescript
export function formatShortCurrency(value: number): string {
  if (value == null || isNaN(value)) return '$0';
  const abs = Math.abs(value);
  if (abs >= 1e9) return '$' + (value / 1e9).toFixed(1) + 'B';
  if (abs >= 1e6) return '$' + (value / 1e6).toFixed(1) + 'M';
  if (abs >= 1e3) return '$' + (value / 1e3).toFixed(1) + 'K';
  return '$' + value.toFixed(2);
}

export function formatNumber(value: number): string {
  if (value == null || isNaN(value)) return '0';
  const abs = Math.abs(value);
  if (abs >= 1e9) return (value / 1e9).toFixed(1) + 'B';
  if (abs >= 1e6) return (value / 1e6).toFixed(1) + 'M';
  if (value % 1 !== 0) {
    if (abs < 1) return value.toFixed(3).replace(/\.?0+$/, '');
    return value.toFixed(2).replace(/\.?0+$/, '');
  }
  return value.toLocaleString();
}

export function formatCount(value: number): string {
  if (value == null || isNaN(value)) return '0';
  return Math.round(value).toLocaleString();
}

export function formatPercentage(value: number): string {
  if (value == null || isNaN(value)) return '0%';
  return (value * 100).toFixed(1) + '%';
}

export type YDisplayType = 'count' | 'currency' | 'percentage' | 'number';

export function getFormatter(t?: YDisplayType): (v: number) => string {
  switch (t) {
    case 'currency': return formatShortCurrency;
    case 'percentage': return formatPercentage;
    case 'count': return formatCount;
    case 'number':
    default: return formatNumber;
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/app/lib/format.ts
git commit -m "feat: shared number / currency / count formatters"
```

---

### Task 24: Report page scaffolding (loading, error, layout)

**Files:**
- Create: `src/app/report/[id]/page.tsx`

- [ ] **Step 1: Create the page**

Write `src/app/report/[id]/page.tsx`:

```tsx
'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';

const ChartCard = dynamic(() => import('./ChartCard'), { ssr: false });
const ReportSummary = dynamic(() => import('./ReportSummary'));
const DataQualityCallout = dynamic(() => import('./DataQualityCallout'));

interface ChartWithCaption {
  spec: any;
  caption: string;
}

interface Report {
  generated_at: string;
  summary: string;
  data_quality: string[];
  charts: ChartWithCaption[];
  metadata: Record<string, any>;
}

export default function ReportPage({ params }: { params: { id: string } }) {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/report/${params.id}`)
      .then(async (r) => {
        if (r.status === 404) throw new Error('This report has expired. Generate a new one.');
        if (!r.ok) throw new Error('Failed to load report');
        return r.json();
      })
      .then(setReport)
      .catch((e) => setError(e.message));
  }, [params.id]);

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-gray-900">Could not load report</h2>
          <p className="mt-2 text-gray-600">{error}</p>
          <a href="/" className="mt-4 inline-block px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
            Back to upload
          </a>
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50">
        <div className="animate-spin rounded-full h-10 w-10 border-4 border-blue-500 border-t-transparent mb-4" />
        <p className="text-gray-700">Loading report…</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-2 md:px-8 py-8 bg-gray-50 min-h-screen">
      <ReportSummary summary={report.summary} generatedAt={report.generated_at} />
      {report.data_quality && report.data_quality.length > 0 && (
        <DataQualityCallout notes={report.data_quality} />
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
        {report.charts.map((c, idx) => (
          <ChartCard key={idx} spec={c.spec} caption={c.caption} />
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add src/app/report/[id]/page.tsx
git commit -m "feat: report page scaffolding (loading, error, layout)"
```

---

### Task 25: ReportSummary + DataQualityCallout components

**Files:**
- Create: `src/app/report/[id]/ReportSummary.tsx`
- Create: `src/app/report/[id]/DataQualityCallout.tsx`

- [ ] **Step 1: Write ReportSummary.tsx**

Write `src/app/report/[id]/ReportSummary.tsx`:

```tsx
interface Props {
  summary: string;
  generatedAt: string;
}

export default function ReportSummary({ summary, generatedAt }: Props) {
  const paragraphs = summary.split(/\n\s*\n/).filter((p) => p.trim());
  const date = new Date(generatedAt).toLocaleString();
  return (
    <header className="mb-8">
      <div className="text-center mb-6">
        <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-2">Data Report</h1>
        <p className="text-sm text-gray-400">Generated: {date}</p>
      </div>
      <div className="max-w-3xl mx-auto space-y-4 text-gray-700 leading-relaxed">
        {paragraphs.map((p, i) => (
          <p key={i}>{p}</p>
        ))}
      </div>
    </header>
  );
}
```

- [ ] **Step 2: Write DataQualityCallout.tsx**

Write `src/app/report/[id]/DataQualityCallout.tsx`:

```tsx
interface Props {
  notes: string[];
}

export default function DataQualityCallout({ notes }: Props) {
  return (
    <aside className="max-w-3xl mx-auto bg-yellow-50 border border-yellow-200 rounded-lg p-4 mt-6">
      <h3 className="font-semibold text-yellow-900 mb-2">Data quality notes</h3>
      <ul className="list-disc pl-5 space-y-1 text-yellow-900 text-sm">
        {notes.map((n, i) => (
          <li key={i}>{n}</li>
        ))}
      </ul>
    </aside>
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add src/app/report/[id]/ReportSummary.tsx src/app/report/[id]/DataQualityCallout.tsx
git commit -m "feat: ReportSummary + DataQualityCallout components"
```

---

### Task 26: ChartCard dispatcher + shared chart container

**Files:**
- Create: `src/app/report/[id]/ChartCard.tsx`

- [ ] **Step 1: Write ChartCard.tsx**

Write `src/app/report/[id]/ChartCard.tsx`:

```tsx
'use client';

import dynamic from 'next/dynamic';

const BarChart = dynamic(() => import('./charts/BarChart'), { ssr: false });
const HistogramChart = dynamic(() => import('./charts/HistogramChart'), { ssr: false });
const ScatterChart = dynamic(() => import('./charts/ScatterChart'), { ssr: false });
const LineChart = dynamic(() => import('./charts/LineChart'), { ssr: false });
const PieChart = dynamic(() => import('./charts/PieChart'), { ssr: false });
const BoxPlot = dynamic(() => import('./charts/BoxPlot'), { ssr: false });
const Heatmap = dynamic(() => import('./charts/Heatmap'), { ssr: false });

interface Props {
  spec: any;
  caption: string;
}

export default function ChartCard({ spec, caption }: Props) {
  const renderer = (() => {
    switch (spec.kind) {
      case 'bar': return <BarChart spec={spec} />;
      case 'histogram': return <HistogramChart spec={spec} />;
      case 'scatter': return <ScatterChart spec={spec} />;
      case 'line': return <LineChart spec={spec} />;
      case 'pie': return <PieChart spec={spec} />;
      case 'box': return <BoxPlot spec={spec} />;
      case 'heatmap': return <Heatmap spec={spec} />;
      default: return <p className="text-sm text-red-600">Unsupported chart kind: {String(spec.kind)}</p>;
    }
  })();

  return (
    <section className="bg-white rounded-2xl shadow-md border border-gray-200 p-6 flex flex-col">
      <h2 className="text-lg font-bold text-gray-900 mb-2">{spec.title}</h2>
      <div className="flex-1 min-h-[300px]">{renderer}</div>
      <p className="text-sm text-gray-600 mt-3 italic">{caption}</p>
    </section>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add src/app/report/[id]/ChartCard.tsx
git commit -m "feat: ChartCard dispatcher (lazy-loads each chart kind)"
```

---

### Task 27: BarChart and HistogramChart components

**Files:**
- Create: `src/app/report/[id]/charts/BarChart.tsx`
- Create: `src/app/report/[id]/charts/HistogramChart.tsx`

- [ ] **Step 1: Write BarChart.tsx**

Write `src/app/report/[id]/charts/BarChart.tsx`:

```tsx
'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';

export default function BarChart({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const option = {
    tooltip: { trigger: 'item', formatter: (p: any) => `${p.name}: ${fmtY(p.value)}` },
    grid: { left: 80, right: 40, top: 30, bottom: 80 },
    xAxis: {
      type: 'category',
      data: spec.x,
      name: spec.x_label,
      nameLocation: 'middle',
      nameGap: 50,
      axisLabel: {
        interval: 0,
        rotate: spec.x.length > 10 ? 45 : 0,
        fontSize: spec.x.length > 15 ? 10 : 12,
      },
    },
    yAxis: {
      type: 'value',
      name: spec.y_label,
      nameLocation: 'middle',
      nameGap: 50,
      axisLabel: { formatter: fmtY },
    },
    series: [{
      type: 'bar',
      data: spec.y,
      itemStyle: { color: '#4ECDC4', borderRadius: [4, 4, 0, 0] },
      label: { show: true, position: 'top', formatter: (p: any) => fmtY(p.value) },
    }],
  };
  return <ReactECharts option={option} style={{ width: '100%', height: 360 }} />;
}
```

- [ ] **Step 2: Write HistogramChart.tsx**

Write `src/app/report/[id]/charts/HistogramChart.tsx`:

```tsx
'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';

export default function HistogramChart({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const option = {
    tooltip: { trigger: 'item', formatter: (p: any) => `${p.name}: ${fmtY(p.value)}` },
    grid: { left: 80, right: 40, top: 30, bottom: 100 },
    xAxis: {
      type: 'category',
      data: spec.x,
      name: spec.x_label,
      nameLocation: 'middle',
      nameGap: 70,
      axisLabel: { interval: 0, rotate: 45, fontSize: 10 },
    },
    yAxis: {
      type: 'value',
      name: spec.y_label,
      nameLocation: 'middle',
      nameGap: 50,
      axisLabel: { formatter: fmtY },
    },
    series: [{
      type: 'bar',
      data: spec.y,
      barCategoryGap: '0%',
      itemStyle: { color: '#5470C6' },
    }],
  };
  return <ReactECharts option={option} style={{ width: '100%', height: 360 }} />;
}
```

- [ ] **Step 3: Commit**

```bash
git add src/app/report/[id]/charts/BarChart.tsx src/app/report/[id]/charts/HistogramChart.tsx
git commit -m "feat: BarChart + HistogramChart components"
```

---

### Task 28: ScatterChart and LineChart components

**Files:**
- Create: `src/app/report/[id]/charts/ScatterChart.tsx`
- Create: `src/app/report/[id]/charts/LineChart.tsx`

- [ ] **Step 1: Write ScatterChart.tsx**

Write `src/app/report/[id]/charts/ScatterChart.tsx`:

```tsx
'use client';
import ReactECharts from 'echarts-for-react';

const COLORS = ['#5470C6', '#91CC75', '#FAC858', '#EE6666', '#73C0DE', '#3BA272', '#FC8452', '#9A60B4'];

export default function ScatterChart({ spec }: { spec: any }) {
  const hasSeries = spec.series && Array.isArray(spec.series);
  const series = hasSeries
    ? spec.series.map((s: any, i: number) => ({
        name: s.name,
        type: 'scatter',
        data: s.x.map((x: number, idx: number) => [x, s.y[idx]]),
        itemStyle: { color: COLORS[i % COLORS.length] },
      }))
    : [{
        type: 'scatter',
        data: spec.x.map((x: number, i: number) => [x, spec.y[i]]),
        itemStyle: { color: '#5470C6' },
      }];

  return (
    <ReactECharts
      option={{
        tooltip: { trigger: 'item' },
        legend: hasSeries ? { bottom: 0 } : undefined,
        grid: { left: 80, right: 40, top: 30, bottom: hasSeries ? 60 : 50 },
        xAxis: { type: 'value', name: spec.x_label, nameLocation: 'middle', nameGap: 30 },
        yAxis: { type: 'value', name: spec.y_label, nameLocation: 'middle', nameGap: 50 },
        series,
      }}
      style={{ width: '100%', height: 360 }}
    />
  );
}
```

- [ ] **Step 2: Write LineChart.tsx**

Write `src/app/report/[id]/charts/LineChart.tsx`:

```tsx
'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';

const COLORS = ['#5470C6', '#91CC75', '#FAC858', '#EE6666', '#73C0DE', '#3BA272'];

export default function LineChart({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const hasSeries = spec.series && Array.isArray(spec.series);
  const xData = hasSeries ? spec.series[0].x : spec.x;
  const series = hasSeries
    ? spec.series.map((s: any, i: number) => ({
        name: s.name,
        type: 'line',
        data: s.y,
        smooth: true,
        itemStyle: { color: COLORS[i % COLORS.length] },
      }))
    : [{ type: 'line', data: spec.y, smooth: true, itemStyle: { color: '#5470C6' } }];

  return (
    <ReactECharts
      option={{
        tooltip: { trigger: 'axis', formatter: (p: any[]) => p.map((x) => `${x.seriesName ?? ''}: ${fmtY(x.value)}`).join('<br/>') },
        legend: hasSeries ? { bottom: 0 } : undefined,
        grid: { left: 80, right: 40, top: 30, bottom: hasSeries ? 60 : 50 },
        xAxis: { type: 'category', data: xData, name: spec.x_label, nameLocation: 'middle', nameGap: 30 },
        yAxis: { type: 'value', name: spec.y_label, nameLocation: 'middle', nameGap: 50, axisLabel: { formatter: fmtY } },
        series,
      }}
      style={{ width: '100%', height: 360 }}
    />
  );
}
```

- [ ] **Step 3: Commit**

```bash
git add src/app/report/[id]/charts/ScatterChart.tsx src/app/report/[id]/charts/LineChart.tsx
git commit -m "feat: ScatterChart + LineChart components"
```

---

### Task 29: PieChart, BoxPlot, Heatmap components

**Files:**
- Create: `src/app/report/[id]/charts/PieChart.tsx`
- Create: `src/app/report/[id]/charts/BoxPlot.tsx`
- Create: `src/app/report/[id]/charts/Heatmap.tsx`

- [ ] **Step 1: Write PieChart.tsx**

Write `src/app/report/[id]/charts/PieChart.tsx`:

```tsx
'use client';
import ReactECharts from 'echarts-for-react';
import { getFormatter } from '../../../lib/format';

export default function PieChart({ spec }: { spec: any }) {
  const fmtY = getFormatter(spec.y_display_type);
  const data = spec.x.map((name: string, i: number) => ({ name, value: spec.y[i] }));
  return (
    <ReactECharts
      option={{
        tooltip: { trigger: 'item', formatter: (p: any) => `${p.name}: ${fmtY(p.value)} (${p.percent}%)` },
        legend: { bottom: 0 },
        series: [{
          type: 'pie',
          radius: ['30%', '60%'],
          data,
          label: { formatter: '{b}: {d}%' },
        }],
      }}
      style={{ width: '100%', height: 360 }}
    />
  );
}
```

- [ ] **Step 2: Write BoxPlot.tsx**

Write `src/app/report/[id]/charts/BoxPlot.tsx`:

```tsx
'use client';
import ReactECharts from 'echarts-for-react';

export default function BoxPlot({ spec }: { spec: any }) {
  const series = spec.series ?? [];
  const categories = series.map((s: any) => s.name);
  const boxData = series.map((s: any) => [s.min, s.q1, s.median, s.q3, s.max]);
  const outliers = series.flatMap((s: any, i: number) => (s.outliers ?? []).map((v: number) => [i, v]));
  return (
    <ReactECharts
      option={{
        tooltip: { trigger: 'item' },
        grid: { left: 80, right: 40, top: 30, bottom: 60 },
        xAxis: { type: 'category', data: categories, name: spec.x_label, nameLocation: 'middle', nameGap: 30 },
        yAxis: { type: 'value', name: spec.y_label, nameLocation: 'middle', nameGap: 50 },
        series: [
          { type: 'boxplot', data: boxData, itemStyle: { color: '#73C0DE' } },
          { type: 'scatter', data: outliers, symbolSize: 6, itemStyle: { color: '#EE6666' } },
        ],
      }}
      style={{ width: '100%', height: 360 }}
    />
  );
}
```

- [ ] **Step 3: Write Heatmap.tsx**

Write `src/app/report/[id]/charts/Heatmap.tsx`:

```tsx
'use client';
import ReactECharts from 'echarts-for-react';

export default function Heatmap({ spec }: { spec: any }) {
  const series = spec.series ?? [];
  const xLabels: string[] = spec.x ?? [];
  const yLabels: string[] = spec.y ?? [];
  const data = series.map((s: any) => [xLabels.indexOf(s.col), yLabels.indexOf(s.row), s.value]);
  const values = series.map((s: any) => s.value);
  const vMin = Math.min(...values, 0);
  const vMax = Math.max(...values, 0);
  return (
    <ReactECharts
      option={{
        tooltip: {
          position: 'top',
          formatter: (p: any) => `${yLabels[p.data[1]]} × ${xLabels[p.data[0]]}: ${p.data[2].toFixed(2)}`,
        },
        grid: { left: 100, right: 40, top: 30, bottom: 60 },
        xAxis: { type: 'category', data: xLabels, axisLabel: { rotate: 30 }, name: spec.x_label, nameLocation: 'middle', nameGap: 40 },
        yAxis: { type: 'category', data: yLabels, name: spec.y_label, nameLocation: 'middle', nameGap: 70 },
        visualMap: { min: vMin, max: vMax, calculable: true, orient: 'horizontal', left: 'center', bottom: 0 },
        series: [{
          type: 'heatmap',
          data,
          label: { show: true, formatter: (p: any) => p.data[2].toFixed(2), fontSize: 10 },
        }],
      }}
      style={{ width: '100%', height: 380 }}
    />
  );
}
```

- [ ] **Step 4: Commit**

```bash
git add src/app/report/[id]/charts/PieChart.tsx src/app/report/[id]/charts/BoxPlot.tsx src/app/report/[id]/charts/Heatmap.tsx
git commit -m "feat: PieChart + BoxPlot + Heatmap components"
```

---

### Task 30: Update upload page to navigate to /report/[id]

**Files:**
- Modify: `src/app/page.tsx`

- [ ] **Step 1: Replace the upload page**

Overwrite `src/app/page.tsx`:

```tsx
'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useRouter } from 'next/navigation';
import Papa from 'papaparse';

interface DataPreview {
  columns: string[];
  data: Record<string, any>[];
}

const STEPS = ['Reading file', 'Analyzing with AI', 'Writing report', 'Done'];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<DataPreview | null>(null);
  const [step, setStep] = useState(0);
  const router = useRouter();

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted.length === 0) return;
    const f = accepted[0];
    if (f.size > 10 * 1024 * 1024) {
      setError('File must be under 10MB.');
      return;
    }
    if (!/\.(csv|xlsx)$/i.test(f.name)) {
      setError('Please upload a .csv or .xlsx file.');
      return;
    }
    setFile(f);
    setError(null);
    setPreview(null);

    if (f.name.toLowerCase().endsWith('.csv')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
        if (parsed.errors.length === 0) {
          const cols = (parsed.meta.fields || []).map((c) => c.toLowerCase());
          const rows = parsed.data.slice(0, 10).map((r: any) => {
            const out: Record<string, any> = {};
            for (const k of Object.keys(r)) out[k.toLowerCase()] = r[k];
            return out;
          });
          setPreview({ columns: cols, data: rows });
        }
      };
      reader.readAsText(f);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
    maxFiles: 1,
  });

  async function generate() {
    if (!file) return;
    setIsProcessing(true);
    setError(null);
    setStep(0);
    try {
      setStep(1);
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/generate-report`, {
        method: 'POST',
        body: fd,
      });
      let body: any = null;
      try { body = await res.json(); } catch {}
      if (res.status === 503 && body?.detail?.status === 'busy') {
        setError(body?.detail?.message ?? 'Service busy. Please retry in 30 seconds.');
        setIsProcessing(false);
        return;
      }
      if (!res.ok) {
        const detail = body?.detail;
        throw new Error(typeof detail === 'string' ? detail : 'Failed to generate report.');
      }
      setStep(2);
      router.push(`/report/${body.session_id}`);
    } catch (e: any) {
      setError(e.message || 'Generation failed.');
      setIsProcessing(false);
    }
  }

  return (
    <div className="container mx-auto px-4 py-10 max-w-4xl">
      <h1 className="text-4xl font-bold text-center text-gray-900 mb-2">Turn data into insight</h1>
      <p className="text-center text-lg text-gray-600 mb-10">
        Drop a CSV or Excel file. Get a narrated report with charts in under 10 seconds.
      </p>

      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-10 text-center transition-colors ${
          isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
        }`}
      >
        <input {...getInputProps()} />
        <p className="text-gray-700 mb-1">Drag and drop, or click to select.</p>
        <p className="text-sm text-gray-500">.csv or .xlsx, up to 10MB.</p>
      </div>

      {error && (
        <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-lg">{error}</div>
      )}

      {file && !isProcessing && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg flex justify-between items-center">
          <div>
            <p className="font-medium">{file.name}</p>
            <p className="text-sm text-gray-500">{(file.size / 1024).toFixed(0)} KB</p>
          </div>
          <button
            onClick={generate}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Generate report
          </button>
        </div>
      )}

      {isProcessing && (
        <div className="mt-6 p-6 bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="flex items-center justify-center mb-4">
            <div className="animate-spin rounded-full h-8 w-8 border-4 border-blue-500 border-t-transparent" />
          </div>
          <p className="text-center font-medium text-gray-900">{STEPS[step]}</p>
        </div>
      )}

      {preview && (
        <div className="mt-8 overflow-x-auto bg-white rounded-lg shadow-sm border border-gray-200">
          <h3 className="px-4 py-3 font-semibold text-gray-900 border-b border-gray-200">Preview</h3>
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50">
              <tr>{preview.columns.map((c) => <th key={c} className="px-4 py-2 text-left">{c}</th>)}</tr>
            </thead>
            <tbody>
              {preview.data.map((row, i) => (
                <tr key={i} className="border-t border-gray-100">
                  {preview.columns.map((c) => (
                    <td key={c} className="px-4 py-2 text-gray-700">{row[c]?.toString() ?? ''}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Manual smoke test**

In one terminal:
```bash
cd src/api && uvicorn main:app --reload
```

In another:
```bash
npm run dev
```

Open `http://localhost:3000`, drop `tests/e2e/fixtures/activities.csv` (once it exists), verify the report loads at `/report/<id>`. (You may need to set a real ANTHROPIC_API_KEY for this manual check.)

- [ ] **Step 3: Commit**

```bash
git add src/app/page.tsx
git commit -m "feat: simplify upload page, navigate directly to /report/[id]"
```

---

### Task 31: Delete the old frontend

**Files:**
- Delete: `src/app/visualizations/` (entire dir)
- Delete: `src/app/dashboard/` (empty dir)
- Delete: `src/app/workers/` (empty dir)

- [ ] **Step 1: Verify nothing imports them**

```bash
grep -rn "visualizations\|/dashboard\|/workers" src/app
```

Expected: no matches except possibly inside the route paths in the new code (none should reference these old folders).

- [ ] **Step 2: Delete**

```bash
rm -rf src/app/visualizations
rm -rf src/app/dashboard
rm -rf src/app/workers
```

- [ ] **Step 3: Run dev build to verify**

```bash
npm run build
```

Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
git add -A src/app
git commit -m "chore: delete old visualizations / dashboard / workers routes"
```

---

## Phase 8 — E2E tests, Makefile, docs

### Task 32: E2E fixture CSVs

**Files:**
- Create: `tests/e2e/__init__.py`
- Create: `tests/e2e/fixtures/activities.csv`
- Create: `tests/e2e/fixtures/sales.csv`
- Create: `tests/e2e/fixtures/signups.csv`
- Create: `tests/e2e/fixtures/survey.csv`
- Create: `tests/e2e/fixtures/degenerate.csv`

- [ ] **Step 1: Create the fixtures directory**

```bash
mkdir -p tests/e2e/fixtures
touch tests/e2e/__init__.py
```

- [ ] **Step 2: Write a Python script that produces the fixtures**

Write `tests/e2e/generate_fixtures.py`:

```python
"""Run once to (re)generate the e2e fixture CSVs."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

OUT = Path(__file__).parent / "fixtures"
OUT.mkdir(exist_ok=True)
rng = np.random.RandomState(42)


def activities():
    n = 200
    types = rng.choice(["consultation", "intro_call", "lab_test", "after_hours"], size=n, p=[0.4, 0.3, 0.25, 0.05])
    start = datetime(2024, 1, 1)
    dates = [start + timedelta(days=int(d)) for d in rng.randint(0, 365, size=n)]
    durations = rng.normal(60, 25, size=n).clip(min=5)
    # Inject the anomalies the design spec mentions
    durations[0] = -30.0
    durations[1] = 600.0
    df = pd.DataFrame({
        "activity_id": range(1, n + 1),
        "patient_id": rng.randint(1001, 1100, size=n),
        "activity_type": types,
        "activity_date": dates,
        "duration_minutes": durations,
    })
    df.to_csv(OUT / "activities.csv", index=False)


def sales():
    n = 150
    regions = rng.choice(["north", "south", "east", "west"], size=n)
    rev = rng.lognormal(7, 1, size=n)
    dates = [datetime(2024, 1, 1) + timedelta(days=int(d)) for d in rng.randint(0, 180, size=n)]
    df = pd.DataFrame({
        "order_id": range(1, n + 1),
        "region": regions,
        "revenue": rev,
        "order_date": dates,
        "product_category": rng.choice(["widgets", "gadgets", "doodads"], size=n),
    })
    df.to_csv(OUT / "sales.csv", index=False)


def signups():
    days = 90
    dates = [datetime(2024, 1, 1) + timedelta(days=d) for d in range(days)]
    base = np.linspace(10, 50, days)
    seasonal = 10 * np.sin(np.linspace(0, 6, days))
    noise = rng.normal(0, 5, size=days)
    counts = (base + seasonal + noise).round().clip(min=0).astype(int)
    df = pd.DataFrame({"date": dates, "signups": counts})
    df.to_csv(OUT / "signups.csv", index=False)


def survey():
    n = 300
    df = pd.DataFrame({
        "respondent_id": range(1, n + 1),
        "satisfaction": rng.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.1, 0.2, 0.4, 0.25]),
        "would_recommend": rng.choice(["yes", "no", "maybe"], size=n, p=[0.55, 0.25, 0.2]),
        "channel": rng.choice(["web", "mobile", "store"], size=n),
    })
    df.to_csv(OUT / "survey.csv", index=False)


def degenerate():
    df = pd.DataFrame({"only_column": [1, None, None, 2, None] * 10})
    df.to_csv(OUT / "degenerate.csv", index=False)


if __name__ == "__main__":
    activities(); sales(); signups(); survey(); degenerate()
    print("fixtures written to", OUT)
```

- [ ] **Step 3: Run the generator**

```bash
python tests/e2e/generate_fixtures.py
```

Expected output: `fixtures written to .../tests/e2e/fixtures`.

- [ ] **Step 4: Commit (include the generated CSVs)**

```bash
git add tests/e2e/__init__.py tests/e2e/generate_fixtures.py tests/e2e/fixtures/*.csv
git commit -m "test: e2e fixture CSVs and generator"
```

---

### Task 33: E2E smoke test against real Claude

**Files:**
- Create: `tests/e2e/test_real_claude_smoke.py`

- [ ] **Step 1: Write the test**

Write `tests/e2e/test_real_claude_smoke.py`:

```python
"""End-to-end smoke tests that hit real Claude.

Gated by RUN_E2E=true and a valid ANTHROPIC_API_KEY in the environment.
Run with: RUN_E2E=true pytest tests/e2e/ -m e2e -v
"""
import os
import pandas as pd
import pytest
from pathlib import Path
from claude_client import ClaudeClient
from llm_config import MODEL_SELECTION, MODEL_NARRATIVE
from profile import profile_dataframe
from report_generator import ReportGenerator
from schemas import Report


pytestmark = pytest.mark.e2e


FIXTURES = Path(__file__).parent / "fixtures"


def _skip_unless_enabled():
    if os.environ.get("RUN_E2E") != "true":
        pytest.skip("Set RUN_E2E=true to run e2e tests.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set.")


def _run(fixture_name: str) -> Report:
    _skip_unless_enabled()
    df = pd.read_csv(FIXTURES / fixture_name)
    df.columns = [c.lower() for c in df.columns]
    profile = profile_dataframe(df)
    client = ClaudeClient(api_key=os.environ["ANTHROPIC_API_KEY"])
    gen = ReportGenerator(
        profile=profile, df=df, claude=client,
        model_selection=MODEL_SELECTION, model_narrative=MODEL_NARRATIVE,
    )
    return gen.build_report()


def test_activities_produces_report():
    report = _run("activities.csv")
    assert len(report.charts) >= 3
    assert report.summary
    for c in report.charts:
        assert c.spec.kind in {"bar", "histogram", "scatter", "line", "pie", "box", "heatmap"}
        if c.spec.x is not None and c.spec.y is not None:
            assert len(c.spec.x) == len(c.spec.y)


def test_activities_summary_mentions_anomaly():
    report = _run("activities.csv")
    text = (report.summary + " ".join(report.data_quality)).lower()
    assert "negative" in text or "outlier" in text or "extreme" in text


def test_sales_produces_report():
    report = _run("sales.csv")
    assert len(report.charts) >= 3


def test_signups_produces_line_chart_somewhere():
    report = _run("signups.csv")
    kinds = {c.spec.kind for c in report.charts}
    assert "line" in kinds or "bar" in kinds


def test_survey_produces_report():
    report = _run("survey.csv")
    assert len(report.charts) >= 3


def test_degenerate_does_not_crash():
    report = _run("degenerate.csv")
    assert report.summary or report.data_quality
```

- [ ] **Step 2: Manually verify (optional, costs ~$0.06 total)**

```bash
RUN_E2E=true ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY pytest tests/e2e/ -v
```

Expected: 6 passed (~60s, ~$0.06 on Haiku 4.5).

If skipped by env: that's the default behavior on commit.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/test_real_claude_smoke.py
git commit -m "test: e2e smoke tests against real Claude (opt-in)"
```

---

### Task 34: Makefile

**Files:**
- Create: `Makefile`

- [ ] **Step 1: Write the Makefile**

Write `Makefile`:

```makefile
.PHONY: dev test test-e2e smoke clean

# Backend dev server (run from project root)
dev:
	cd src/api && uvicorn main:app --reload --port 8000

# Fast: unit + integration (no real Claude calls)
test:
	pytest tests/unit tests/integration -v

# Slow: opt-in real Claude smoke tests
test-e2e:
	RUN_E2E=true pytest tests/e2e -v -m e2e

# Boot the API + smoke a single fixture
smoke:
	@cd src/api && (uvicorn main:app --port 8001 &) && sleep 2 && \
		curl -f -X POST http://localhost:8001/generate-report \
			-F "file=@../../tests/e2e/fixtures/sales.csv" \
			| tee /tmp/chartsage_smoke.json \
		&& echo "OK" \
		&& pkill -f "uvicorn main:app --port 8001"

clean:
	rm -rf src/api/__pycache__ src/api/logs/*.log
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

- [ ] **Step 2: Verify**

```bash
make test
```

Expected: all unit + integration tests pass.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "chore: Makefile for dev / test / smoke"
```

---

### Task 35: Update .env.example and add prompts package marker

**Files:**
- Modify: `.env.example`
- Create: `src/api/prompts/__init__.py` (so the dir is a package; safe even though we read files)

- [ ] **Step 1: Rewrite .env.example**

Overwrite `.env.example`:

```bash
# Required: Anthropic API key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000

# Model selection (optional; defaults to haiku-4-5)
# Aliases: haiku-4-5, sonnet-4-6, opus-4-7
# Or pass any full model id like claude-haiku-4-5-20251001
# CLAUDE_MODEL=haiku-4-5

# Per-pass overrides (optional)
# CLAUDE_MODEL_SELECTION=haiku-4-5
# CLAUDE_MODEL_NARRATIVE=sonnet-4-6

# E2E tests (opt-in)
# RUN_E2E=true
```

- [ ] **Step 2: Create empty package marker**

```bash
touch src/api/prompts/__init__.py
```

- [ ] **Step 3: Commit**

```bash
git add .env.example src/api/prompts/__init__.py
git commit -m "chore: .env.example reflects v2 (no Stripe/NextAuth)"
```

---

### Task 36: Update README.md and ChartSage.md

**Files:**
- Modify: `README.md`
- Modify: `ChartSage.md`

- [ ] **Step 1: Rewrite README.md**

Overwrite `README.md`:

```markdown
# ChartSage

Drop a CSV or Excel file. Get a narrated data report with charts in under 10 seconds.

## What it does

ChartSage profiles your data, asks Claude to pick the 5-7 charts that tell the most useful story, renders them with ECharts, and wraps the result in a written executive summary plus a data-quality callout when something looks off in your data.

## Tech Stack

- **Frontend:** Next.js 14, React, TypeScript, Tailwind CSS, ECharts
- **Backend:** FastAPI, Python 3.11+, pandas, Pydantic v2
- **AI:** Claude via Anthropic SDK (Haiku 4.5 default; switchable)
- **Storage:** Redis (24-hour session TTL)

## Getting started

### Prerequisites

- Python 3.11+
- Node.js 18+
- Redis running on localhost:6379 (`brew install redis && brew services start redis`)
- An Anthropic API key

### Setup

```bash
# Backend
cp .env.example .env
# edit .env to add ANTHROPIC_API_KEY
pip install -r requirements.txt

# Frontend
npm install
```

### Run

In one terminal:

```bash
make dev          # FastAPI on :8000
```

In another:

```bash
npm run dev       # Next.js on :3000
```

Open `http://localhost:3000`, drop a CSV, see a report.

## Switching models

Default is `haiku-4-5` (~$0.01 per report). Switch by setting one env var:

```bash
CLAUDE_MODEL=sonnet-4-6 make dev          # ~$0.035/report
CLAUDE_MODEL=opus-4-7 make dev            # ~$0.04/report
```

Per-pass overrides (cheap selection, smarter narrative):

```bash
CLAUDE_MODEL_SELECTION=haiku-4-5
CLAUDE_MODEL_NARRATIVE=sonnet-4-6
```

## Tests

```bash
make test         # unit + integration (~4s, no API calls)
make test-e2e     # real Claude smoke tests (~60s, ~$0.06)
```

## Architecture

See [docs/superpowers/specs/2026-05-23-chartsage-rebuild-design.md](docs/superpowers/specs/2026-05-23-chartsage-rebuild-design.md).

## License

MIT.
```

- [ ] **Step 2: Rewrite ChartSage.md**

Overwrite `ChartSage.md`:

```markdown
# ChartSage — Internal Documentation

## Architecture

CSV/Excel → DataFrame → profile → Claude (selection via parallel tool use) → executors → Claude (narrative via forced tool use) → Report → Redis (24h) → frontend renders.

Full design: [docs/superpowers/specs/2026-05-23-chartsage-rebuild-design.md](docs/superpowers/specs/2026-05-23-chartsage-rebuild-design.md).

## Adding a new chart kind

1. Add an executor to `src/api/chart_executor.py` and register it in `TOOL_EXECUTORS`.
2. Add the matching tool definition to `CHART_TOOLS` in `src/api/chart_tools.py`.
3. Add an executor test file under `tests/unit/`.
4. Add a frontend renderer at `src/app/report/[id]/charts/<NewKind>.tsx` and wire it into `ChartCard.tsx`.

## Logs

- Per-run logs under `src/api/logs/chartsage_run_<timestamp>_<runid>.log`.
- Last 50 runs kept; older auto-deleted.
- Trailer block at end of each log shows the run summary (model, tokens, charts, elapsed).
```

- [ ] **Step 3: Commit**

```bash
git add README.md ChartSage.md
git commit -m "docs: README and internal docs reflect v2 reality"
```

---

### Task 37: Final test sweep

**Files:** none (verification only)

- [ ] **Step 1: Run all non-e2e tests**

```bash
make test
```

Expected: all green.

- [ ] **Step 2: Run e2e tests (if API key available)**

```bash
make test-e2e
```

Expected: 6 e2e tests pass.

- [ ] **Step 3: Run the manual smoke**

```bash
make dev &
sleep 3
curl -f -X POST http://localhost:8000/generate-report \
  -F "file=@tests/e2e/fixtures/sales.csv"
```

Expected: JSON with a `session_id`. Then `GET /report/<id>` returns a populated report.

```bash
pkill -f "uvicorn main:app"
```

- [ ] **Step 4: Commit (no-op if clean)**

```bash
git status
# If anything outstanding, commit with a "polish" message.
```

---

## Spec coverage check

| Spec section | Implemented in |
|---|---|
| Two-pass architecture | Task 18 (pass #1), Task 20 (pass #2) |
| Tool-use protocol (8 tools) | Task 15 |
| profile_dataframe + role detection | Task 6 |
| 7 chart kinds | Tasks 7-14 |
| Trimmed-IQR histogram binning | Task 9 |
| Tool-error retry round | Task 19 |
| Heuristic fallback | Task 17 |
| Forced submit_narrative tool | Task 15 + Task 20 |
| FastAPI /generate-report + /report/{id} | Task 21 |
| Redis 24h session storage | Task 21 |
| Old endpoint removal | Task 21 + Task 22 |
| .env.example cleanup | Task 35 |
| README cleanup | Task 36 |
| Model alias config (env-driven) | Task 4 |
| Per-pass model override | Task 4 |
| Prompt caching for system + tools | Task 5 |
| Retry on transient 5xx, surface 529 as 503 | Task 5 + Task 21 |
| Per-run log file + structured trailer | Task 21 |
| Log rotation (keep last 50) | Task 21 |
| Unit tests per executor (golden values) | Tasks 7-14 |
| Pipeline integration tests (happy / retry / fallback) | Tasks 18-20 |
| API error tests | Task 21 |
| E2E smoke tests against real Claude | Tasks 32-33 |
| Frontend report page | Tasks 24-29 |
| Updated upload page (navigate to /report/[id]) | Task 30 |
| Delete old visualizations/dashboard/workers | Task 31 |
| Makefile targets (test, test-e2e, smoke, dev) | Task 34 |

No spec gaps.

