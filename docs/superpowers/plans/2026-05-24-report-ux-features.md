# Report UX Features (v1.1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four interactivity features to the report page — collapsible insight summary, drag-and-drop chart reordering, sidebar with 5 extra charts (and a "Generate 5 more" button), and a server-rendered PDF export — without disrupting the 10-chart Claude generation pipeline.

**Architecture:** Backend adds three new endpoints to `main.py` (PATCH layout, POST generate-more, GET export.pdf), plus a Playwright PDF module. Report schema gains a `layout` array and per-chart `chart_id`s — the chart specs themselves stay immutable. Frontend wraps the report in a dnd-kit `DndContext`, splits the existing chart grid into a Main + Sidebar two-column layout, and adds a Toolbar with the two action buttons. A print-only route at `/report/[id]/print` is what Playwright renders for the PDF.

**Tech Stack:** Python 3.11+, FastAPI, pandas, Pydantic v2, anthropic SDK, Playwright (new); Next.js 14, React 18, ECharts, @dnd-kit/core + @dnd-kit/sortable (new), Tailwind, Redis. See [the design spec](../specs/2026-05-24-report-ux-features-design.md) (commit `f03b826`) for full context.

---

## File Structure

### Backend (new / modified)

```
src/api/
├── main.py                # +3 endpoints: PATCH layout, POST generate-more, GET export.pdf
├── schemas.py             # +ChartLayoutEntry, +chart_id on ChartWithCaption, +layout on Report
├── report_generator.py    # build_report assigns chart_ids + default layout; new generate_more()
└── pdf_export.py          # NEW — Playwright lifecycle + render_report_pdf()
```

### Frontend (new / modified)

```
src/app/report/[id]/
├── page.tsx               # DndContext + two-column layout
├── ReportSummary.tsx      # collapsible <details>
├── ChartCard.tsx          # drag handle + hide button; index from prop
├── Sidebar.tsx            # NEW — right rail container
├── SidebarCard.tsx        # NEW — compact card with "Move to main"
├── Toolbar.tsx            # NEW — sticky top bar (Export, Generate more)
├── useReportLayout.ts     # NEW — layout state + debounced PATCH
└── print/
    └── page.tsx           # NEW — print-only route
```

### Tests (new)

```
tests/
├── unit/
│   ├── test_schemas_layout.py            # ChartLayoutEntry, chart_id, layout schema
│   ├── test_report_generator_layout.py   # build_report + generate_more semantics
│   └── test_pdf_export.py                # opt-in (RUN_PDF_TESTS=true)
└── integration/
    └── test_api_layout.py                # PATCH layout, POST generate-more
```

### Top-level

```
package.json               # +@dnd-kit/core, +@dnd-kit/sortable
requirements.txt           # +playwright
Makefile                   # +test-pdf target
```

---

## Phase 1 — Backend schemas and default layout

### Task 1: ChartLayoutEntry + Report.layout + ChartWithCaption.chart_id

**Files:**
- Modify: `src/api/schemas.py`
- Create: `tests/unit/test_schemas_layout.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/unit/test_schemas_layout.py`:

```python
import pytest
from pydantic import ValidationError
from schemas import ChartLayoutEntry, ChartSpec, ChartWithCaption, Report


def _spec() -> ChartSpec:
    return ChartSpec(
        kind="bar", title="t", intent="i",
        x=["a"], y=[1],
        x_label="X", y_label="Y",
        x_display_type="category", y_display_type="count",
        source_columns=["c"], data_point_count=1,
    )


def test_layout_entry_minimum():
    e = ChartLayoutEntry(chart_id="abc", position="main", order=0)
    assert e.chart_id == "abc"
    assert e.position == "main"
    assert e.order == 0


def test_layout_entry_rejects_invalid_position():
    with pytest.raises(ValidationError):
        ChartLayoutEntry(chart_id="abc", position="trash", order=0)


def test_chart_with_caption_has_chart_id():
    c = ChartWithCaption(chart_id="abc", spec=_spec(), caption="cap")
    assert c.chart_id == "abc"


def test_report_has_layout_field_with_default_empty():
    r = Report(
        generated_at="2026-05-24T00:00:00",
        summary="...", data_quality=[], charts=[], metadata={},
    )
    assert r.layout == []


def test_report_round_trips_with_layout():
    r = Report(
        generated_at="2026-05-24T00:00:00",
        summary="s", data_quality=[],
        charts=[ChartWithCaption(chart_id="c1", spec=_spec(), caption="cap")],
        layout=[ChartLayoutEntry(chart_id="c1", position="main", order=0)],
        metadata={},
    )
    payload = r.model_dump_json()
    r2 = Report.model_validate_json(payload)
    assert r2.layout[0].chart_id == "c1"
    assert r2.charts[0].chart_id == "c1"
```

- [ ] **Step 2: Run test, expect import errors**

```bash
PYTHONPATH=src/api pytest tests/unit/test_schemas_layout.py -v
```

Expected: `ImportError: cannot import name 'ChartLayoutEntry'`.

- [ ] **Step 3: Add schema models**

Edit `src/api/schemas.py`. After the existing `ChartKind` Literal, add:

```python
LayoutPosition = Literal["main", "sidebar"]


class ChartLayoutEntry(BaseModel):
    chart_id: str
    position: LayoutPosition
    order: int
```

Modify the existing `ChartWithCaption`:

```python
class ChartWithCaption(BaseModel):
    chart_id: str
    spec: ChartSpec
    caption: str
```

Modify the existing `Report` to add `layout` after `charts`:

```python
class Report(BaseModel):
    generated_at: str
    summary: str
    data_quality: list[str]
    charts: list[ChartWithCaption]
    layout: list[ChartLayoutEntry] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Run tests, expect pass**

```bash
PYTHONPATH=src/api pytest tests/unit/test_schemas_layout.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Run full suite to confirm no regressions**

```bash
make test
```

Expected: 114 passed (109 original + 5 new). If existing `test_schemas.py` failed because `ChartWithCaption` now requires `chart_id`, update the existing happy-path test to pass it.

If `test_schemas.py::test_report_round_trips_json` fails because `Report` constructor no longer accepts charts without `chart_id`, fix the test to include `chart_id="c1"` in any `ChartWithCaption` it builds. (The current test builds `Report` with `charts=[]` so this should be safe.)

- [ ] **Step 6: Commit**

```bash
git add src/api/schemas.py tests/unit/test_schemas_layout.py
git commit -m "feat: add ChartLayoutEntry schema, chart_id, Report.layout"
```

---

### Task 2: build_report assigns chart_ids and default layout

**Files:**
- Modify: `src/api/report_generator.py`
- Create: `tests/unit/test_report_generator_layout.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/unit/test_report_generator_layout.py`:

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
    return ReportGenerator(
        profile=profile, df=df, claude=client,
        model_selection="m1", model_narrative="m2",
    )


def _ten_charts_response(activities):
    """Build a FakeClaude with 10 chart tool calls + a narrative call."""
    chart_calls = []
    for i in range(10):
        chart_calls.append(tool_use(
            "frequency_bar_chart",
            {"column": "activity_type", "title": f"Chart {i}", "intent": f"i{i}"},
        ))
    return FakeClaude([
        {"tool_calls": chart_calls},
        {"tool_calls": [tool_use(
            "submit_narrative",
            {"summary": "s", "captions": [f"c{i}" for i in range(10)], "data_quality": []},
        )]},
    ])


def test_chart_ids_unique(activities):
    fake = _ten_charts_response(activities)
    gen = _make_generator(activities, fake)
    report = gen.build_report()
    ids = [c.chart_id for c in report.charts]
    assert len(ids) == len(set(ids)), "chart_ids must be unique"
    assert all(isinstance(i, str) and len(i) > 0 for i in ids)


def test_default_layout_splits_5_main_5_sidebar(activities):
    fake = _ten_charts_response(activities)
    gen = _make_generator(activities, fake)
    report = gen.build_report()

    assert len(report.layout) == 10
    main = [e for e in report.layout if e.position == "main"]
    sidebar = [e for e in report.layout if e.position == "sidebar"]
    assert len(main) == 5
    assert len(sidebar) == 5

    # First 5 charts go to main, next 5 to sidebar
    main_ids_in_order = [e.chart_id for e in sorted(main, key=lambda e: e.order)]
    sidebar_ids_in_order = [e.chart_id for e in sorted(sidebar, key=lambda e: e.order)]
    assert main_ids_in_order == [c.chart_id for c in report.charts[:5]]
    assert sidebar_ids_in_order == [c.chart_id for c in report.charts[5:]]

    # Orders are 0-indexed and dense
    assert [e.order for e in sorted(main, key=lambda e: e.order)] == [0, 1, 2, 3, 4]
    assert [e.order for e in sorted(sidebar, key=lambda e: e.order)] == [0, 1, 2, 3, 4]


def test_fewer_than_5_charts_all_main(activities):
    """Edge case: if only 3 charts come back, all go to main."""
    chart_calls = [tool_use(
        "frequency_bar_chart",
        {"column": "activity_type", "title": f"c{i}", "intent": "i"},
    ) for i in range(3)]
    fake = FakeClaude([
        {"tool_calls": chart_calls},
        {"tool_calls": [tool_use(
            "submit_narrative",
            {"summary": "s", "captions": ["c1", "c2", "c3"], "data_quality": []},
        )]},
    ])
    gen = _make_generator(activities, fake)
    report = gen.build_report()
    assert all(e.position == "main" for e in report.layout)
    assert len(report.layout) == 3
```

- [ ] **Step 2: Run, expect failure**

```bash
PYTHONPATH=src/api pytest tests/unit/test_report_generator_layout.py -v
```

Expected: tests fail because `chart_id` is missing or layout is empty.

- [ ] **Step 3: Modify build_report**

Edit `src/api/report_generator.py`. Find the `build_report` method and replace it with:

```python
    def build_report(self) -> Report:
        from datetime import datetime
        from uuid import uuid4
        from schemas import ChartLayoutEntry

        charts = self.generate_charts()
        narrative = self.generate_narrative(charts)

        captions = narrative.captions
        if len(captions) < len(charts):
            captions = captions + [c.intent for c in charts[len(captions):]]

        # Assign stable chart_ids
        charts_with_caption = [
            ChartWithCaption(chart_id=uuid4().hex, spec=spec, caption=cap)
            for spec, cap in zip(charts, captions)
        ]

        # Default layout: first 5 -> main, next 5 -> sidebar
        layout: list[ChartLayoutEntry] = []
        for i, cwc in enumerate(charts_with_caption[:5]):
            layout.append(ChartLayoutEntry(chart_id=cwc.chart_id, position="main", order=i))
        for i, cwc in enumerate(charts_with_caption[5:]):
            layout.append(ChartLayoutEntry(chart_id=cwc.chart_id, position="sidebar", order=i))

        return Report(
            generated_at=datetime.utcnow().isoformat(),
            summary=narrative.summary or self._narrative_template_fallback(charts).summary,
            data_quality=narrative.data_quality,
            charts=charts_with_caption,
            layout=layout,
            metadata={
                "model_selection": self.model_selection,
                "model_narrative": self.model_narrative,
                "row_count": self.profile.row_count,
                "column_count": len(self.profile.columns),
            },
        )
```

- [ ] **Step 4: Run tests, expect pass**

```bash
PYTHONPATH=src/api pytest tests/unit/test_report_generator_layout.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run full suite — fix any regressions**

```bash
make test
```

Expected: all tests pass. If the existing integration tests in `test_pipeline_happy.py` failed because they assert on `report.charts[0].caption` and now `ChartWithCaption` requires a `chart_id`, they should still pass because we're constructing in `build_report` and the test reads through `report.charts[i].caption`.

If you see a failure on a test that constructed `ChartWithCaption(spec=..., caption=...)` directly without `chart_id`, change that test to include `chart_id="test-id"`. Most likely-affected: `test_pipeline_happy.py`, `test_pipeline_fallback.py`.

- [ ] **Step 6: Commit**

```bash
git add src/api/report_generator.py tests/unit/test_report_generator_layout.py
git commit -m "feat: build_report assigns chart_ids and default 5+5 layout"
```

---

## Phase 2 — Backend endpoints

### Task 3: PATCH /report/{id}/layout endpoint

**Files:**
- Modify: `src/api/main.py`
- Create: `tests/integration/test_api_layout.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/integration/test_api_layout.py`:

```python
import io
import json
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from tests.helpers.fake_claude import FakeClaude, tool_use


def _csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


@pytest.fixture
def client_with_report(sales):
    """Boot the app, create a real report in fake-redis, return (client, session_id, report)."""
    chart_calls = []
    for i in range(10):
        chart_calls.append(tool_use(
            "frequency_bar_chart",
            {"column": "region", "title": f"Chart {i}", "intent": f"intent {i}"},
            id_=f"tu_{i}",
        ))
    fake = FakeClaude([
        {"tool_calls": chart_calls},
        {"tool_calls": [tool_use(
            "submit_narrative",
            {"summary": "Sales.", "captions": [f"cap{i}" for i in range(10)], "data_quality": []},
        )]},
    ])
    fake_redis_data: dict[str, str] = {}

    class FakeRedis:
        def set(self, key, val, ex=None): fake_redis_data[key] = val
        def get(self, key): return fake_redis_data.get(key)

    from main import app, get_claude_client, get_redis
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=fake)
    app.dependency_overrides[get_redis] = lambda: FakeRedis()

    client = TestClient(app)

    resp = client.post(
        "/generate-report",
        files={"file": ("sales.csv", _csv_bytes(sales), "text/csv")},
    )
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    report = json.loads(fake_redis_data[f"report:{session_id}"])

    yield client, session_id, report
    app.dependency_overrides.clear()


def test_patch_layout_valid_returns_204(client_with_report):
    client, session_id, report = client_with_report
    # Swap positions of chart 0 and chart 1
    layout = report["layout"]
    layout[0], layout[1] = layout[1], layout[0]
    layout[0]["order"], layout[1]["order"] = 0, 1
    resp = client.patch(f"/report/{session_id}/layout", json=layout)
    assert resp.status_code == 204


def test_patch_layout_unknown_chart_id_returns_400(client_with_report):
    client, session_id, report = client_with_report
    bad_layout = [{"chart_id": "does-not-exist", "position": "main", "order": 0}]
    resp = client.patch(f"/report/{session_id}/layout", json=bad_layout)
    assert resp.status_code == 400
    assert "does-not-exist" in resp.text or "chart_id" in resp.text.lower()


def test_patch_layout_unknown_session_returns_404(client_with_report):
    client, _, _ = client_with_report
    resp = client.patch("/report/no-such-session/layout", json=[])
    assert resp.status_code == 404


def test_patch_layout_persists_to_redis(client_with_report):
    client, session_id, report = client_with_report
    # Move chart 5 (originally sidebar) to main
    layout = report["layout"]
    chart_5 = layout[5]
    chart_5["position"] = "main"
    chart_5["order"] = 5
    resp = client.patch(f"/report/{session_id}/layout", json=layout)
    assert resp.status_code == 204
    # Re-fetch and verify
    resp2 = client.get(f"/report/{session_id}")
    assert resp2.status_code == 200
    updated = resp2.json()
    moved = next(e for e in updated["layout"] if e["chart_id"] == chart_5["chart_id"])
    assert moved["position"] == "main"
```

- [ ] **Step 2: Run tests, expect 404 on PATCH (endpoint doesn't exist yet)**

```bash
PYTHONPATH=src/api pytest tests/integration/test_api_layout.py -v
```

Expected: All 4 fail with 404 or 405 (Method Not Allowed).

- [ ] **Step 3: Add the PATCH endpoint to main.py**

Edit `src/api/main.py`. After the `GET /report/{session_id}` endpoint, add:

```python
from schemas import ChartLayoutEntry


@app.patch("/report/{session_id}/layout", status_code=204)
async def patch_report_layout(
    session_id: str,
    new_layout: list[ChartLayoutEntry],
    r=Depends(get_redis),
):
    raw = r.get(f"report:{session_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Report not found or expired.")

    report_dict = json.loads(raw)
    known_ids = {c["chart_id"] for c in report_dict.get("charts", [])}

    submitted_ids = {entry.chart_id for entry in new_layout}
    unknown = submitted_ids - known_ids
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown chart_id(s) in layout: {sorted(unknown)}",
        )

    report_dict["layout"] = [entry.model_dump() for entry in new_layout]
    r.set(f"report:{session_id}", json.dumps(report_dict), ex=SESSION_TTL_SECONDS)
    return None
```

- [ ] **Step 4: Run tests, expect pass**

```bash
PYTHONPATH=src/api pytest tests/integration/test_api_layout.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Full sweep**

```bash
make test
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/api/main.py tests/integration/test_api_layout.py
git commit -m "feat: PATCH /report/{id}/layout endpoint"
```

---

### Task 4: POST /report/{id}/generate-more endpoint

**Files:**
- Modify: `src/api/report_generator.py`
- Modify: `src/api/main.py`
- Modify: `tests/integration/test_api_layout.py`

- [ ] **Step 1: Add tests for the new endpoint**

Append to `tests/integration/test_api_layout.py`:

```python
def test_generate_more_appends_charts_to_sidebar(client_with_report, sales, monkeypatch):
    client, session_id, report = client_with_report

    # Build a second FakeClaude that returns 5 NEW charts + a narrative
    new_chart_calls = []
    for i in range(5):
        new_chart_calls.append(tool_use(
            "histogram_chart",
            {"column": "revenue", "title": f"More chart {i}", "intent": f"new intent {i}"},
            id_=f"tu_more_{i}",
        ))
    new_fake = FakeClaude([
        {"tool_calls": new_chart_calls},
        {"tool_calls": [tool_use(
            "submit_narrative",
            {"summary": "Updated.", "captions": [f"new cap {i}" for i in range(5)], "data_quality": []},
        )]},
    ])

    # Swap the claude client to use the new fake
    from main import app, get_claude_client
    app.dependency_overrides[get_claude_client] = lambda: MagicMock(messages_create=new_fake)

    initial_chart_count = len(report["charts"])
    initial_sidebar_count = sum(1 for e in report["layout"] if e["position"] == "sidebar")

    resp = client.post(f"/report/{session_id}/generate-more")
    assert resp.status_code == 200

    updated = resp.json()
    assert len(updated["charts"]) == initial_chart_count + 5
    assert len(updated["layout"]) == len(updated["charts"])

    new_sidebar_count = sum(1 for e in updated["layout"] if e["position"] == "sidebar")
    assert new_sidebar_count == initial_sidebar_count + 5

    # New chart_ids are unique
    all_ids = [c["chart_id"] for c in updated["charts"]]
    assert len(all_ids) == len(set(all_ids))


def test_generate_more_unknown_session(client_with_report):
    client, _, _ = client_with_report
    resp = client.post("/report/no-such-session/generate-more")
    assert resp.status_code == 404
```

- [ ] **Step 2: Run, expect failures**

```bash
PYTHONPATH=src/api pytest tests/integration/test_api_layout.py -v
```

Expected: the two new tests fail (405 or 404).

- [ ] **Step 3: Add `generate_more` to report_generator.py**

Edit `src/api/report_generator.py`. After `build_report`, add a new method on the same class:

```python
    def generate_more(self, existing: list[ChartWithCaption]) -> tuple[list[ChartWithCaption], list[ChartLayoutEntry]]:
        """Run a focused pass-1 to produce 5 additional charts different from `existing`.

        Returns (new charts with fresh chart_ids, new layout entries to append).
        Existing charts and layout are not modified.
        """
        from datetime import datetime  # noqa: F401
        from uuid import uuid4
        from schemas import ChartLayoutEntry

        # Build a focused user message that warns Claude off the existing chart angles
        existing_summary = "\n".join(
            f"- [{c.spec.kind}] {c.spec.title} — {c.spec.intent}"
            for c in existing
        )
        focused_message = (
            f"{self.profile.to_text()}\n\n"
            f"You have already produced these charts:\n{existing_summary}\n\n"
            f"Pick 5 different angles that are NOT repeats of the above. "
            f"Vary kinds; aim for chart types or column combinations not yet covered."
        )

        response = self.claude.messages_create(
            model=self.model_selection,
            max_tokens=4096,
            system=SELECTION_SYSTEM,
            tools=CHART_TOOLS,
            messages=[{"role": "user", "content": focused_message}],
            cache_static=True,
        )
        specs, errors = self._execute_tool_calls(response.content)

        if errors:
            specs2, _ = self._call_selection_retry(response.content, errors)
            specs.extend(specs2)

        # Cap at 5 new charts
        specs = specs[:5]

        # Narrative captions for the new charts only
        narrative = self.generate_narrative(specs) if specs else None
        captions = narrative.captions if narrative else [s.intent for s in specs]
        if len(captions) < len(specs):
            captions = captions + [s.intent for s in specs[len(captions):]]

        new_charts = [
            ChartWithCaption(chart_id=uuid4().hex, spec=spec, caption=cap)
            for spec, cap in zip(specs, captions)
        ]

        # New layout entries always start in sidebar
        return new_charts, [
            ChartLayoutEntry(chart_id=c.chart_id, position="sidebar", order=0)  # order set by caller
            for c in new_charts
        ]
```

- [ ] **Step 4: Add the POST endpoint to main.py**

Edit `src/api/main.py`. At the top of the file, ensure `Report` is imported alongside `ChartLayoutEntry`:

```python
from schemas import ChartLayoutEntry, Report
```

Then after the PATCH layout endpoint, add:

```python
@app.post("/report/{session_id}/generate-more")
async def generate_more(
    session_id: str,
    claude: ClaudeClient = Depends(get_claude_client),
    r=Depends(get_redis),
):
    raw = r.get(f"report:{session_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Report not found or expired.")

    report_dict = json.loads(raw)
    existing_report = Report.model_validate(report_dict)

    # Rebuild a generator scaffold from the stored profile.
    # We don't keep the original df; this is a soft limitation — generate_more
    # uses the same profile + the existing chart list, no df re-execution needed
    # because we're only running pass-1 (chart selection), not the executors.
    # WAIT: we DO need the df to actually execute the new chart tools. Reject if
    # we can't reconstruct.
    #
    # The pragmatic v1.1 approach: store the file bytes alongside the report on
    # initial generate. For now: 501 if df is unavailable.
    df_blob = r.get(f"report:{session_id}:df")
    if not df_blob:
        raise HTTPException(
            status_code=501,
            detail="generate-more requires the original file. Re-upload to enable.",
        )

    import io
    df = pd.read_parquet(io.BytesIO(df_blob.encode("latin1"))) \
        if df_blob.startswith("PAR1") else pd.read_csv(io.StringIO(df_blob))

    profile = profile_dataframe(df)
    gen = ReportGenerator(
        profile=profile, df=df, claude=claude,
        model_selection=MODEL_SELECTION, model_narrative=MODEL_NARRATIVE,
    )

    try:
        new_charts, new_layout = gen.generate_more(existing_report.charts)
    except RetryableBusy:
        raise HTTPException(
            status_code=503,
            detail={"status": "busy", "message": "Claude is busy. Please retry in 30 seconds."},
        )

    if not new_charts:
        return JSONResponse(content=report_dict, status_code=200)

    # Append charts and layout entries with correct sidebar orders
    sidebar_max = max(
        (e["order"] for e in report_dict["layout"] if e["position"] == "sidebar"),
        default=-1,
    )
    for i, (chart, layout_entry) in enumerate(zip(new_charts, new_layout)):
        layout_entry.order = sidebar_max + 1 + i
        report_dict["charts"].append(chart.model_dump())
        report_dict["layout"].append(layout_entry.model_dump())

    r.set(f"report:{session_id}", json.dumps(report_dict), ex=SESSION_TTL_SECONDS)
    return JSONResponse(content=report_dict, status_code=200)
```

- [ ] **Step 5: Store the df alongside the report**

We need to make `generate-more` actually work. Edit `src/api/main.py`'s `/generate-report` endpoint. After the line that writes the report to Redis, add a line that also stores the CSV/XLSX bytes:

Find:
```python
    session_id = uuid.uuid4().hex
    r.set(f"report:{session_id}", report.model_dump_json(), ex=SESSION_TTL_SECONDS)
```

Change to:
```python
    session_id = uuid.uuid4().hex
    r.set(f"report:{session_id}", report.model_dump_json(), ex=SESSION_TTL_SECONDS)
    # Persist the source data so /generate-more can re-execute tool calls
    df_csv = df.to_csv(index=False)
    r.set(f"report:{session_id}:df", df_csv, ex=SESSION_TTL_SECONDS)
```

And change the `/generate-more` endpoint's df-loading block (in step 4 above):

Replace:
```python
    df_blob = r.get(f"report:{session_id}:df")
    if not df_blob:
        raise HTTPException(
            status_code=501,
            detail="generate-more requires the original file. Re-upload to enable.",
        )

    import io
    df = pd.read_parquet(io.BytesIO(df_blob.encode("latin1"))) \
        if df_blob.startswith("PAR1") else pd.read_csv(io.StringIO(df_blob))
```

With:
```python
    df_blob = r.get(f"report:{session_id}:df")
    if not df_blob:
        raise HTTPException(
            status_code=404,
            detail="Source data for this report has expired. Generate a new report.",
        )

    import io
    df = pd.read_csv(io.StringIO(df_blob))
    df.columns = [str(c).lower() for c in df.columns]
```

(Also update the test fixture in `tests/integration/test_api_layout.py` `client_with_report` to seed `fake_redis_data[f"report:{session_id}:df"]` with the CSV string. Edit the fixture after the report is created:)

```python
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    report = json.loads(fake_redis_data[f"report:{session_id}"])

    # NEW: store df alongside (the /generate-report endpoint now does this,
    # but our fake redis bypasses the storage; manually seed it for tests)
    fake_redis_data[f"report:{session_id}:df"] = _csv_bytes(sales).decode("utf-8")
```

(If you completed step 5 fully, the real endpoint sets this already. The fixture-edit above is only needed if the test setup wasn't using the real endpoint code path. Since our fixture goes through `client.post("/generate-report", ...)`, the new code in step 5 will set the key. No manual seed needed. **Skip the manual seed.**)

- [ ] **Step 6: Run tests**

```bash
PYTHONPATH=src/api pytest tests/integration/test_api_layout.py -v
```

Expected: 6 passed (4 from Task 3 + 2 new).

- [ ] **Step 7: Full sweep**

```bash
make test
```

Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add src/api/main.py src/api/report_generator.py tests/integration/test_api_layout.py
git commit -m "feat: POST /report/{id}/generate-more endpoint"
```

---

### Task 5: Playwright PDF export module + endpoint

**Files:**
- Create: `src/api/pdf_export.py`
- Modify: `src/api/main.py`
- Modify: `requirements.txt`
- Modify: `Makefile`
- Create: `tests/unit/test_pdf_export.py`

- [ ] **Step 1: Install Playwright**

```bash
echo "playwright==1.42.0" >> requirements.txt
pip install playwright==1.42.0
playwright install chromium
```

- [ ] **Step 2: Add the pdf_export module**

Write `src/api/pdf_export.py`:

```python
"""Playwright-driven PDF export.

Owns a single long-lived Browser instance, started lazily on first use.
Each export opens a fresh Page, navigates to the print route, waits for the
data-charts-ready flag, captures as A4 PDF.
"""
import asyncio
import logging
import os
from typing import Optional


_FRONTEND_BASE = os.getenv("FRONTEND_BASE_URL", "http://localhost:3000")
_CHARTS_READY_TIMEOUT_MS = 10_000


_browser_lock = asyncio.Lock()
_browser = None
_playwright = None


async def _ensure_browser():
    """Lazily start Playwright + Chromium. Reused across exports."""
    global _browser, _playwright
    async with _browser_lock:
        if _browser is None:
            from playwright.async_api import async_playwright
            _playwright = await async_playwright().start()
            _browser = await _playwright.chromium.launch(headless=True)
            logging.info("[PDF] Started Chromium")
    return _browser


async def render_report_pdf(session_id: str) -> bytes:
    """Render the /report/{session_id}/print route as an A4 PDF and return bytes."""
    browser = await _ensure_browser()
    page = await browser.new_page(viewport={"width": 1240, "height": 1754})
    try:
        url = f"{_FRONTEND_BASE}/report/{session_id}/print"
        await page.goto(url, wait_until="networkidle", timeout=30_000)
        # Wait for the print page to signal "all charts mounted"
        try:
            await page.wait_for_selector(
                'body[data-charts-ready="true"]',
                timeout=_CHARTS_READY_TIMEOUT_MS,
            )
        except Exception:
            logging.warning("[PDF] charts-ready flag not seen in %dms — proceeding", _CHARTS_READY_TIMEOUT_MS)
        pdf_bytes = await page.pdf(
            format="A4",
            print_background=True,
            margin={"top": "50px", "bottom": "50px", "left": "40px", "right": "40px"},
        )
        return pdf_bytes
    finally:
        await page.close()


async def shutdown():
    """Close the Browser and Playwright instance. Used on app shutdown."""
    global _browser, _playwright
    if _browser is not None:
        await _browser.close()
        _browser = None
    if _playwright is not None:
        await _playwright.stop()
        _playwright = None
```

- [ ] **Step 3: Wire the endpoint**

Edit `src/api/main.py`. After the generate-more endpoint, add:

```python
from fastapi.responses import StreamingResponse
import io as _io


@app.get("/report/{session_id}/export.pdf")
async def export_pdf(session_id: str, r=Depends(get_redis)):
    if not r.get(f"report:{session_id}"):
        raise HTTPException(status_code=404, detail="Report not found or expired.")

    from pdf_export import render_report_pdf
    try:
        pdf_bytes = await render_report_pdf(session_id)
    except Exception as e:
        logging.exception("[PDF] export failed")
        raise HTTPException(status_code=500, detail=f"PDF export failed: {e}")

    short = session_id[:8]
    return StreamingResponse(
        _io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="chartsage-{short}.pdf"'},
    )
```

Add a shutdown hook just before `if __name__ == "__main__":` at the bottom:

```python
@app.on_event("shutdown")
async def _shutdown_event():
    from pdf_export import shutdown as pdf_shutdown
    await pdf_shutdown()
```

- [ ] **Step 4: Write the opt-in test**

Write `tests/unit/test_pdf_export.py`:

```python
"""Opt-in: requires Playwright + Chromium installed and a running frontend.

Run with: RUN_PDF_TESTS=true pytest tests/unit/test_pdf_export.py -v
"""
import asyncio
import os
import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_PDF_TESTS") != "true",
    reason="Opt-in: set RUN_PDF_TESTS=true",
)


def test_pdf_export_returns_pdf_bytes():
    """Smoke: render a stub report and assert we get valid PDF bytes."""
    # This test requires a real frontend at http://localhost:3000 to be running.
    # It's primarily here for CI; in practice, manually drive the UI.
    from pdf_export import render_report_pdf

    # Use a known-good session_id from a manual generate-report run.
    session_id = os.getenv("PDF_TEST_SESSION_ID")
    if not session_id:
        pytest.skip("Set PDF_TEST_SESSION_ID to a valid session_id")

    pdf_bytes = asyncio.run(render_report_pdf(session_id))
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 10_000
```

- [ ] **Step 5: Add Makefile target**

Edit `Makefile`. Add:

```makefile
test-pdf:
	RUN_PDF_TESTS=true pytest tests/unit/test_pdf_export.py -v
```

- [ ] **Step 6: Verify default test suite still passes (PDF tests skip)**

```bash
make test
```

Expected: 121 passed (109 original + 8 new from layout/PATCH/generate-more/schemas/build_report). PDF tests skip with reason.

- [ ] **Step 7: Commit**

```bash
git add src/api/pdf_export.py src/api/main.py requirements.txt Makefile tests/unit/test_pdf_export.py
git commit -m "feat: Playwright-driven PDF export endpoint"
```

---

## Phase 3 — Frontend foundation

### Task 6: Install dnd-kit

**Files:**
- Modify: `package.json`

- [ ] **Step 1: Install**

```bash
npm install @dnd-kit/core@^6.1.0 @dnd-kit/sortable@^8.0.0 @dnd-kit/utilities@^3.2.2
```

- [ ] **Step 2: Verify build still works**

```bash
npm run build
```

Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add package.json package-lock.json
git commit -m "chore: install @dnd-kit/core, /sortable, /utilities"
```

---

### Task 7: useReportLayout hook

**Files:**
- Create: `src/app/report/[id]/useReportLayout.ts`

- [ ] **Step 1: Implement the hook**

Write `src/app/report/[id]/useReportLayout.ts`:

```typescript
'use client';
import { useCallback, useEffect, useRef, useState } from 'react';

export interface ChartLayoutEntry {
  chart_id: string;
  position: 'main' | 'sidebar';
  order: number;
}

export interface ChartWithCaption {
  chart_id: string;
  spec: any;
  caption: string;
}

export interface Report {
  generated_at: string;
  summary: string;
  data_quality: string[];
  charts: ChartWithCaption[];
  layout: ChartLayoutEntry[];
  metadata: Record<string, any>;
}

const PATCH_DEBOUNCE_MS = 500;

export function useReportLayout(initial: Report, sessionId: string) {
  const [report, setReport] = useState<Report>(initial);
  const [saveError, setSaveError] = useState<string | null>(null);
  const patchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastSavedLayoutRef = useRef<ChartLayoutEntry[]>(initial.layout);

  const queuePatch = useCallback((nextLayout: ChartLayoutEntry[]) => {
    if (patchTimerRef.current) clearTimeout(patchTimerRef.current);
    patchTimerRef.current = setTimeout(async () => {
      try {
        const res = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL}/report/${sessionId}/layout`,
          {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(nextLayout),
          },
        );
        if (!res.ok) throw new Error(`Save failed (${res.status})`);
        lastSavedLayoutRef.current = nextLayout;
        setSaveError(null);
      } catch (e: any) {
        setSaveError(e.message || 'Could not save layout');
        // Revert to last saved state
        setReport((r) => ({ ...r, layout: lastSavedLayoutRef.current }));
      }
    }, PATCH_DEBOUNCE_MS);
  }, [sessionId]);

  useEffect(() => () => {
    if (patchTimerRef.current) clearTimeout(patchTimerRef.current);
  }, []);

  /** Reorder a chart within its current position to a new index. */
  const reorder = useCallback((chartId: string, newIndex: number) => {
    setReport((r) => {
      const entry = r.layout.find((e) => e.chart_id === chartId);
      if (!entry) return r;
      const others = r.layout
        .filter((e) => e.position === entry.position && e.chart_id !== chartId)
        .sort((a, b) => a.order - b.order);
      others.splice(newIndex, 0, entry);
      const reordered = others.map((e, i) => ({ ...e, order: i }));
      const next: ChartLayoutEntry[] = [
        ...r.layout.filter((e) => e.position !== entry.position),
        ...reordered,
      ];
      queuePatch(next);
      return { ...r, layout: next };
    });
  }, [queuePatch]);

  /** Move a chart to a different position (main <-> sidebar). */
  const move = useCallback((chartId: string, newPosition: 'main' | 'sidebar') => {
    setReport((r) => {
      const entry = r.layout.find((e) => e.chart_id === chartId);
      if (!entry || entry.position === newPosition) return r;

      // Remove from old position and compact
      const oldSiblings = r.layout
        .filter((e) => e.position === entry.position && e.chart_id !== chartId)
        .sort((a, b) => a.order - b.order)
        .map((e, i) => ({ ...e, order: i }));

      // Append to new position at the end
      const newSiblings = r.layout
        .filter((e) => e.position === newPosition)
        .sort((a, b) => a.order - b.order);
      const moved: ChartLayoutEntry = {
        ...entry,
        position: newPosition,
        order: newSiblings.length,
      };
      const updatedNew = [...newSiblings, moved];

      const next = [...oldSiblings, ...updatedNew];
      queuePatch(next);
      return { ...r, layout: next };
    });
  }, [queuePatch]);

  /** Replace the report wholesale (used after generate-more). */
  const replaceReport = useCallback((newReport: Report) => {
    if (patchTimerRef.current) clearTimeout(patchTimerRef.current);
    lastSavedLayoutRef.current = newReport.layout;
    setSaveError(null);
    setReport(newReport);
  }, []);

  // Partition for the UI
  const mainCharts = report.layout
    .filter((e) => e.position === 'main')
    .sort((a, b) => a.order - b.order)
    .map((e) => report.charts.find((c) => c.chart_id === e.chart_id))
    .filter((c): c is ChartWithCaption => !!c);

  const sidebarCharts = report.layout
    .filter((e) => e.position === 'sidebar')
    .sort((a, b) => a.order - b.order)
    .map((e) => report.charts.find((c) => c.chart_id === e.chart_id))
    .filter((c): c is ChartWithCaption => !!c);

  return {
    report,
    mainCharts,
    sidebarCharts,
    reorder,
    move,
    replaceReport,
    saveError,
  };
}
```

- [ ] **Step 2: Type-check**

```bash
npm run build
```

Expected: build succeeds (the hook is unused so far, just type-checked).

- [ ] **Step 3: Commit**

```bash
git add src/app/report/'[id]'/useReportLayout.ts
git commit -m "feat: useReportLayout hook with optimistic + debounced PATCH"
```

---

### Task 8: Collapsible ReportSummary

**Files:**
- Modify: `src/app/report/[id]/ReportSummary.tsx`

- [ ] **Step 1: Rewrite the component**

Overwrite `src/app/report/[id]/ReportSummary.tsx`:

```typescript
'use client';
import { useState } from 'react';

interface Props {
  summary: string;
  generatedAt: string;
}

export default function ReportSummary({ summary, generatedAt }: Props) {
  const [open, setOpen] = useState(true);
  const paragraphs = summary.split(/\n\s*\n/).filter((p) => p.trim());
  const date = new Date(generatedAt).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  return (
    <header className="mb-10">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-baseline justify-between pb-6 border-b border-stone-200 group cursor-pointer"
        aria-expanded={open}
      >
        <div className="flex items-baseline gap-3">
          <svg
            className={`w-4 h-4 text-stone-400 transition-transform ${open ? 'rotate-90' : ''}`}
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          <h1 className="text-3xl md:text-4xl font-semibold tracking-tight text-stone-900">
            Insights
          </h1>
        </div>
        <span className="text-xs uppercase tracking-widest text-stone-400">{date}</span>
      </button>
      <div
        className={`overflow-hidden transition-all duration-300 ${open ? 'max-h-[2000px] opacity-100 mt-6' : 'max-h-0 opacity-0'}`}
      >
        <div className="max-w-3xl space-y-4 text-stone-700 leading-relaxed text-[15px]">
          {paragraphs.map((p, i) => (
            <p key={i}>{p}</p>
          ))}
        </div>
      </div>
    </header>
  );
}
```

- [ ] **Step 2: Build + smoke**

```bash
npm run build
```

Expected: success.

- [ ] **Step 3: Manual smoke**

Reload the existing report URL in the browser. Click the title — body collapses with animation. Click again — expands.

- [ ] **Step 4: Commit**

```bash
git add src/app/report/'[id]'/ReportSummary.tsx
git commit -m "feat: collapsible insight summary with chevron"
```

---

## Phase 4 — Frontend interactivity

### Task 9: ChartCard with drag handle, hide button, prop-based index

**Files:**
- Modify: `src/app/report/[id]/ChartCard.tsx`

- [ ] **Step 1: Rewrite ChartCard**

Overwrite `src/app/report/[id]/ChartCard.tsx`:

```typescript
'use client';

import dynamic from 'next/dynamic';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

const BarChart = dynamic(() => import('./charts/BarChart'), { ssr: false });
const HistogramChart = dynamic(() => import('./charts/HistogramChart'), { ssr: false });
const ScatterChart = dynamic(() => import('./charts/ScatterChart'), { ssr: false });
const LineChart = dynamic(() => import('./charts/LineChart'), { ssr: false });
const PieChart = dynamic(() => import('./charts/PieChart'), { ssr: false });
const BoxPlot = dynamic(() => import('./charts/BoxPlot'), { ssr: false });
const Heatmap = dynamic(() => import('./charts/Heatmap'), { ssr: false });

interface Props {
  index: number;
  spec: any;
  caption: string;
  chartId: string;
  onHide?: (chartId: string) => void;
}

const KIND_LABEL: Record<string, string> = {
  bar: 'Bar', histogram: 'Histogram', scatter: 'Scatter',
  line: 'Trend', pie: 'Composition', box: 'Distribution', heatmap: 'Heatmap',
};

function ChartContent({ spec }: { spec: any }) {
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
}

export default function ChartCard({ index, spec, caption, chartId, onHide }: Props) {
  const sortable = useSortable({ id: chartId });
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = sortable;

  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.4 : 1,
  };

  const kindLabel = KIND_LABEL[spec.kind] ?? spec.kind;

  return (
    <section
      ref={setNodeRef}
      style={style}
      className="bg-white rounded-2xl ring-1 ring-stone-200/80 shadow-[0_1px_3px_rgba(0,0,0,0.04)] hover:shadow-[0_4px_12px_rgba(0,0,0,0.06)] transition-shadow p-6 flex flex-col"
    >
      <div className="flex items-start justify-between mb-3 gap-2">
        <div className="flex items-baseline gap-3 min-w-0 flex-1">
          <button
            type="button"
            className="text-stone-300 hover:text-stone-600 cursor-grab active:cursor-grabbing -ml-1 px-1 leading-none focus:outline-none focus:ring-2 focus:ring-stone-400 rounded"
            aria-label="Drag to reorder"
            {...attributes}
            {...listeners}
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <circle cx="6" cy="5" r="1.5" /><circle cx="10" cy="5" r="1.5" /><circle cx="14" cy="5" r="1.5" />
              <circle cx="6" cy="10" r="1.5" /><circle cx="10" cy="10" r="1.5" /><circle cx="14" cy="10" r="1.5" />
              <circle cx="6" cy="15" r="1.5" /><circle cx="10" cy="15" r="1.5" /><circle cx="14" cy="15" r="1.5" />
            </svg>
          </button>
          <span className="text-xs font-mono text-stone-400 tabular-nums shrink-0">
            {String(index).padStart(2, '0')}
          </span>
          <h2 className="text-base font-semibold text-stone-900 leading-snug tracking-tight truncate">
            {spec.title}
          </h2>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className="text-[10px] uppercase tracking-widest text-stone-400 mt-1">
            {kindLabel}
          </span>
          {onHide && (
            <button
              type="button"
              onClick={() => onHide(chartId)}
              className="text-stone-300 hover:text-stone-700 transition-colors w-6 h-6 flex items-center justify-center rounded hover:bg-stone-100 focus:outline-none focus:ring-2 focus:ring-stone-400"
              aria-label="Move to sidebar"
              title="Move to sidebar"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>
      <div className="flex-1 min-h-[300px]">
        <ChartContent spec={spec} />
      </div>
      {caption && (
        <p className="text-sm text-stone-600 mt-4 pt-4 border-t border-stone-100 leading-relaxed">
          {caption}
        </p>
      )}
    </section>
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
git add src/app/report/'[id]'/ChartCard.tsx
git commit -m "feat: drag handle + hide button on ChartCard"
```

---

### Task 10: Sidebar + SidebarCard

**Files:**
- Create: `src/app/report/[id]/SidebarCard.tsx`
- Create: `src/app/report/[id]/Sidebar.tsx`

- [ ] **Step 1: Write SidebarCard**

Write `src/app/report/[id]/SidebarCard.tsx`:

```typescript
'use client';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

interface Props {
  chartId: string;
  title: string;
  kind: string;
  onPromote: (chartId: string) => void;
}

const KIND_LABEL: Record<string, string> = {
  bar: 'Bar', histogram: 'Histogram', scatter: 'Scatter',
  line: 'Trend', pie: 'Composition', box: 'Distribution', heatmap: 'Heatmap',
};

export default function SidebarCard({ chartId, title, kind, onPromote }: Props) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id: chartId });

  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.4 : 1,
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      className="bg-white rounded-xl ring-1 ring-stone-200/80 p-3 mb-2 flex items-start gap-2"
    >
      <button
        type="button"
        className="text-stone-300 hover:text-stone-600 cursor-grab active:cursor-grabbing px-0.5 leading-none focus:outline-none focus:ring-2 focus:ring-stone-400 rounded shrink-0"
        aria-label="Drag to reorder"
        {...attributes}
        {...listeners}
      >
        <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
          <circle cx="6" cy="5" r="1.5" /><circle cx="10" cy="5" r="1.5" /><circle cx="14" cy="5" r="1.5" />
          <circle cx="6" cy="10" r="1.5" /><circle cx="10" cy="10" r="1.5" /><circle cx="14" cy="10" r="1.5" />
          <circle cx="6" cy="15" r="1.5" /><circle cx="10" cy="15" r="1.5" /><circle cx="14" cy="15" r="1.5" />
        </svg>
      </button>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-stone-800 leading-snug truncate">{title}</p>
        <p className="text-[10px] uppercase tracking-widest text-stone-400 mt-0.5">
          {KIND_LABEL[kind] ?? kind}
        </p>
      </div>
      <button
        type="button"
        onClick={() => onPromote(chartId)}
        className="text-stone-400 hover:text-stone-900 transition-colors w-6 h-6 flex items-center justify-center rounded hover:bg-stone-100 shrink-0 focus:outline-none focus:ring-2 focus:ring-stone-400"
        aria-label="Move to main report"
        title="Move to main"
      >
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M11 17l-5-5m0 0l5-5m-5 5h12" />
        </svg>
      </button>
    </div>
  );
}
```

- [ ] **Step 2: Write Sidebar**

Write `src/app/report/[id]/Sidebar.tsx`:

```typescript
'use client';
import { useState } from 'react';
import SidebarCard from './SidebarCard';
import type { ChartWithCaption } from './useReportLayout';

interface Props {
  charts: ChartWithCaption[];
  onPromote: (chartId: string) => void;
}

export default function Sidebar({ charts, onPromote }: Props) {
  const [open, setOpen] = useState(true);

  if (!open) {
    return (
      <aside className="w-10 shrink-0">
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="w-10 h-10 rounded-lg bg-white ring-1 ring-stone-200/80 flex items-center justify-center text-stone-500 hover:text-stone-900 hover:bg-stone-50"
          aria-label="Expand sidebar"
          title={`${charts.length} extra charts`}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M11 19l-7-7 7-7M19 19l-7-7 7-7" />
          </svg>
        </button>
      </aside>
    );
  }

  return (
    <aside className="w-72 shrink-0">
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="text-xs uppercase tracking-widest text-stone-400">
          More charts · {charts.length}
        </h3>
        <button
          type="button"
          onClick={() => setOpen(false)}
          className="text-stone-400 hover:text-stone-700 text-sm"
          aria-label="Collapse sidebar"
        >
          ›
        </button>
      </div>
      {charts.length === 0 ? (
        <p className="text-sm text-stone-400 italic">
          Drag a chart here or click ×, or hit "Generate 5 more" above.
        </p>
      ) : (
        <div>
          {charts.map((c) => (
            <SidebarCard
              key={c.chart_id}
              chartId={c.chart_id}
              title={c.spec.title}
              kind={c.spec.kind}
              onPromote={onPromote}
            />
          ))}
        </div>
      )}
    </aside>
  );
}
```

- [ ] **Step 3: Build check**

```bash
npm run build
```

Expected: success.

- [ ] **Step 4: Commit**

```bash
git add src/app/report/'[id]'/Sidebar.tsx src/app/report/'[id]'/SidebarCard.tsx
git commit -m "feat: Sidebar + SidebarCard components"
```

---

### Task 11: Toolbar with Export PDF + Generate 5 more

**Files:**
- Create: `src/app/report/[id]/Toolbar.tsx`

- [ ] **Step 1: Write Toolbar**

Write `src/app/report/[id]/Toolbar.tsx`:

```typescript
'use client';
import { useState } from 'react';
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
    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/report/${sessionId}/generate-more`,
        { method: 'POST' },
      );
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
    const url = `${process.env.NEXT_PUBLIC_API_URL}/report/${sessionId}/export.pdf`;
    window.open(url, '_blank');
    // We can't reliably detect download completion, so reset the spinner after a short delay
    setTimeout(() => setExporting(false), 1500);
  }

  return (
    <div className="sticky top-0 z-10 -mx-4 sm:-mx-6 lg:-mx-8 px-4 sm:px-6 lg:px-8 py-3 mb-6 bg-stone-50/90 backdrop-blur border-b border-stone-200 flex items-center justify-end gap-3">
      {error && (
        <span className="text-sm text-red-600 mr-auto">{error}</span>
      )}
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
git commit -m "feat: Toolbar with Generate-more + Export PDF buttons"
```

---

### Task 12: Wire it all together in page.tsx

**Files:**
- Modify: `src/app/report/[id]/page.tsx`

- [ ] **Step 1: Rewrite page.tsx**

Overwrite `src/app/report/[id]/page.tsx`:

```typescript
'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import {
  DndContext,
  closestCenter,
  DragEndEvent,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
} from '@dnd-kit/core';
import {
  SortableContext,
  arrayMove,
  sortableKeyboardCoordinates,
  rectSortingStrategy,
} from '@dnd-kit/sortable';
import { useReportLayout, type Report } from './useReportLayout';

const ChartCard = dynamic(() => import('./ChartCard'), { ssr: false });
const ReportSummary = dynamic(() => import('./ReportSummary'));
const DataQualityCallout = dynamic(() => import('./DataQualityCallout'));
const Sidebar = dynamic(() => import('./Sidebar'), { ssr: false });
const Toolbar = dynamic(() => import('./Toolbar'), { ssr: false });

function Loading() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-stone-50">
      <div className="animate-spin rounded-full h-9 w-9 border-2 border-stone-300 border-t-stone-900 mb-4" />
      <p className="text-stone-600 text-sm">Loading report…</p>
    </div>
  );
}

function ErrorView({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-stone-50">
      <div className="text-center">
        <h2 className="text-xl font-semibold text-stone-900">Could not load report</h2>
        <p className="mt-2 text-stone-600">{message}</p>
        <a href="/" className="mt-6 inline-block px-5 py-2.5 bg-stone-900 text-white text-sm rounded-lg hover:bg-stone-800 transition-colors">
          Back to upload
        </a>
      </div>
    </div>
  );
}

export default function ReportPage({ params }: { params: { id: string } }) {
  const [initial, setInitial] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/report/${params.id}`)
      .then(async (r) => {
        if (r.status === 404) throw new Error('This report has expired. Generate a new one.');
        if (!r.ok) throw new Error('Failed to load report');
        return r.json();
      })
      .then(setInitial)
      .catch((e) => setError(e.message));
  }, [params.id]);

  if (error) return <ErrorView message={error} />;
  if (!initial) return <Loading />;

  return <ReportView sessionId={params.id} initialReport={initial} />;
}

function ReportView({ sessionId, initialReport }: { sessionId: string; initialReport: Report }) {
  const { report, mainCharts, sidebarCharts, reorder, move, replaceReport, saveError } =
    useReportLayout(initialReport, sessionId);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event;
    if (!over) return;
    if (active.id === over.id) return;

    const activeId = String(active.id);
    const overId = String(over.id);

    const activeEntry = report.layout.find((e) => e.chart_id === activeId);
    const overEntry = report.layout.find((e) => e.chart_id === overId);
    if (!activeEntry || !overEntry) return;

    if (activeEntry.position === overEntry.position) {
      // Reorder within the same position
      const siblings = report.layout
        .filter((e) => e.position === activeEntry.position)
        .sort((a, b) => a.order - b.order)
        .map((e) => e.chart_id);
      const fromIdx = siblings.indexOf(activeId);
      const toIdx = siblings.indexOf(overId);
      if (fromIdx === -1 || toIdx === -1) return;
      reorder(activeId, toIdx);
    } else {
      // Cross-position: move to the other side
      move(activeId, overEntry.position);
    }
  }

  return (
    <div className="min-h-screen bg-stone-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Toolbar sessionId={sessionId} onReportUpdated={replaceReport} />

        <ReportSummary summary={report.summary} generatedAt={report.generated_at} />
        {report.data_quality && report.data_quality.length > 0 && (
          <DataQualityCallout notes={report.data_quality} />
        )}

        {saveError && (
          <div className="mt-4 p-3 bg-amber-50 border border-amber-200 text-amber-900 text-sm rounded-lg">
            {saveError}
          </div>
        )}

        <DndContext
          sensors={sensors}
          collisionDetection={closestCenter}
          onDragEnd={handleDragEnd}
        >
          <div className="flex gap-6 mt-10">
            <main className="flex-1 min-w-0">
              <SortableContext
                items={mainCharts.map((c) => c.chart_id)}
                strategy={rectSortingStrategy}
              >
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {mainCharts.map((c, idx) => (
                    <ChartCard
                      key={c.chart_id}
                      chartId={c.chart_id}
                      index={idx + 1}
                      spec={c.spec}
                      caption={c.caption}
                      onHide={(id) => move(id, 'sidebar')}
                    />
                  ))}
                </div>
              </SortableContext>
            </main>
            <SortableContext
              items={sidebarCharts.map((c) => c.chart_id)}
              strategy={rectSortingStrategy}
            >
              <Sidebar
                charts={sidebarCharts}
                onPromote={(id) => move(id, 'main')}
              />
            </SortableContext>
          </div>
        </DndContext>

        <footer className="mt-16 pt-6 border-t border-stone-200 text-xs text-stone-400 flex justify-between">
          <span>Report id: {sessionId.slice(0, 8)}</span>
          <a href="/" className="hover:text-stone-600">New report →</a>
        </footer>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Build check**

```bash
npm run build
```

Expected: success.

- [ ] **Step 3: Manual smoke**

Start backend (`make dev`), start frontend (`npm run dev`). Generate a new report. Verify:

- Main grid shows 5 charts, sidebar shows 5
- Drag chart 02 onto chart 04 → order updates, layout saves (no error)
- Click the × on a main chart → chart animates to sidebar
- Click the ← on a sidebar chart → chart appears in main
- Reload → layout persists

If the layout PATCH fails silently, open browser devtools network tab. The PATCH request should return 204.

- [ ] **Step 4: Commit**

```bash
git add src/app/report/'[id]'/page.tsx
git commit -m "feat: DndContext wiring + main/sidebar layout"
```

---

## Phase 5 — Print route + PDF export

### Task 13: Print route with print CSS

**Files:**
- Create: `src/app/report/[id]/print/page.tsx`

- [ ] **Step 1: Write the print page**

Write `src/app/report/[id]/print/page.tsx`:

```typescript
'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';

const ChartCard = dynamic(() => import('../ChartCard'), { ssr: false });
const ReportSummary = dynamic(() => import('../ReportSummary'));
const DataQualityCallout = dynamic(() => import('../DataQualityCallout'));

interface ChartLayoutEntry {
  chart_id: string;
  position: 'main' | 'sidebar';
  order: number;
}
interface ChartWithCaption { chart_id: string; spec: any; caption: string; }
interface Report {
  generated_at: string;
  summary: string;
  data_quality: string[];
  charts: ChartWithCaption[];
  layout: ChartLayoutEntry[];
  metadata: Record<string, any>;
}

const CHARTS_PER_PAGE = 2;

export default function PrintReportPage({ params }: { params: { id: string } }) {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/report/${params.id}`)
      .then(async (r) => {
        if (!r.ok) throw new Error('Failed to load report');
        return r.json();
      })
      .then(setReport)
      .catch((e) => setError(e.message));
  }, [params.id]);

  useEffect(() => {
    if (!report) return;
    // Signal readiness after the charts have had time to render.
    // ECharts mounts asynchronously; this rAF + small timeout is enough.
    const t = setTimeout(() => {
      document.body.setAttribute('data-charts-ready', 'true');
    }, 1500);
    return () => clearTimeout(t);
  }, [report]);

  if (error) return <p style={{ padding: 40 }}>Error: {error}</p>;
  if (!report) return <p style={{ padding: 40 }}>Loading…</p>;

  const mainCharts = report.layout
    .filter((e) => e.position === 'main')
    .sort((a, b) => a.order - b.order)
    .map((e) => report.charts.find((c) => c.chart_id === e.chart_id))
    .filter((c): c is ChartWithCaption => !!c);

  // Group charts into pages of CHARTS_PER_PAGE
  const pages: ChartWithCaption[][] = [];
  for (let i = 0; i < mainCharts.length; i += CHARTS_PER_PAGE) {
    pages.push(mainCharts.slice(i, i + CHARTS_PER_PAGE));
  }

  return (
    <>
      <style jsx global>{`
        @media print {
          @page { size: A4; margin: 50px 40px; }
          body { background: white !important; }
          .print-page-break { page-break-after: always; }
          .no-print { display: none !important; }
        }
        body { background: white; }
        .print-container { font-family: Inter, system-ui, sans-serif; }
      `}</style>
      <div className="print-container max-w-[700px] mx-auto p-8">
        {/* Cover page: title + summary + data quality */}
        <header className="mb-8">
          <p className="text-xs uppercase tracking-widest text-stone-400 mb-2">ChartSage Report</p>
          <h1 className="text-4xl font-semibold tracking-tight text-stone-900 mb-3">Insights</h1>
          <p className="text-xs text-stone-500">
            Generated {new Date(report.generated_at).toLocaleDateString(undefined, {
              year: 'numeric', month: 'long', day: 'numeric',
            })}
          </p>
        </header>

        <article className="prose prose-sm max-w-none text-stone-700 leading-relaxed mb-8">
          {report.summary.split(/\n\s*\n/).filter(Boolean).map((p, i) => (
            <p key={i}>{p}</p>
          ))}
        </article>

        {report.data_quality && report.data_quality.length > 0 && (
          <DataQualityCallout notes={report.data_quality} />
        )}

        <div className="print-page-break" />

        {/* Charts: CHARTS_PER_PAGE per page */}
        {pages.map((pageCharts, pageIdx) => (
          <div key={pageIdx} className={pageIdx < pages.length - 1 ? 'print-page-break' : ''}>
            {pageCharts.map((c, i) => (
              <div key={c.chart_id} className="mb-6">
                <ChartCard
                  chartId={c.chart_id}
                  index={pageIdx * CHARTS_PER_PAGE + i + 1}
                  spec={c.spec}
                  caption={c.caption}
                />
              </div>
            ))}
          </div>
        ))}

        <footer className="mt-12 pt-4 border-t border-stone-200 text-xs text-stone-400 text-center">
          ChartSage · Report {params.id.slice(0, 8)}
        </footer>
      </div>
    </>
  );
}
```

- [ ] **Step 2: Build check**

```bash
npm run build
```

Expected: success.

- [ ] **Step 3: Manual smoke**

Open `http://localhost:3000/report/{your-session-id}/print` in the browser. The page should render with no toolbar, no sidebar, no drag handles (well, the drag handles exist on the ChartCard but they're inactive without a DndContext — visually harmless). Use Cmd+P (Mac) / Ctrl+P (Windows) → Save as PDF to preview the PDF flow that Playwright will run.

- [ ] **Step 4: Commit**

```bash
git add src/app/report/'[id]'/print/page.tsx
git commit -m "feat: print-only route with paginated charts"
```

---

### Task 14: End-to-end manual smoke

**Files:** none (verification only)

- [ ] **Step 1: Confirm everything is running**

```bash
# Terminal 1
make dev

# Terminal 2
npm run dev
```

- [ ] **Step 2: Walk through every feature**

Open `http://localhost:3000`:

1. Drop a CSV (use `tests/e2e/fixtures/sales.csv` or your own)
2. Verify 10 charts generate, 5 in main + 5 in sidebar
3. Click the **Insights** title — body collapses; click again — expands
4. Drag chart 02 onto 04 — order swaps; reload page — order persists
5. Click `×` on a main chart — it animates into the sidebar
6. Click the back-arrow on a sidebar chart — it moves to main
7. Click **Generate 5 more** — spinner runs, 5 new cards appear in sidebar
8. Click **Export PDF** — new tab opens with a downloadable PDF; open the PDF and verify:
   - Cover page has title + summary + data-quality (if any)
   - Subsequent pages have 2 charts each
   - Page break between cover and charts works
   - Footer reads `ChartSage · Report {id}`

If anything is broken, fix it now before moving on.

- [ ] **Step 3: Run the full test suite**

```bash
make test
```

Expected: all green.

- [ ] **Step 4: Commit any polish fixes**

If you fixed bugs during the smoke test:

```bash
git add -p   # selectively stage
git commit -m "polish: <specific fix>"
```

---

## Spec coverage check

| Spec requirement | Implemented in |
|---|---|
| Collapsible insight summary | Task 8 |
| Drag-and-drop reorder within main | Tasks 6, 9, 10, 12 (DndContext + useSortable + page wiring) |
| Drag-and-drop cross-context (main ↔ sidebar) | Tasks 9, 10, 12 (move() in hook + DndContext logic) |
| Sidebar listing extra charts | Tasks 10, 12 |
| Hide chart (× button) | Task 9 |
| Move chart back to main (← button) | Task 10 |
| Default 5 main + 5 sidebar | Task 2 (build_report) |
| Layout persists across reloads | Tasks 3, 7 (PATCH endpoint + debounced hook sync) |
| Stable chart_id per chart | Tasks 1, 2 |
| "Generate 5 more" button | Tasks 4, 11 |
| Generate-more produces non-duplicate angles | Task 4 (`generate_more` method's prompt) |
| Server-side PDF export | Task 5, 13 |
| Print route at /report/[id]/print | Task 13 |
| PDF has cover page + paginated charts | Task 13 (print CSS + grouping) |
| Playwright launches lazily, reuses Browser | Task 5 (`_ensure_browser`) |
| 503 surface on Claude busy in generate-more | Task 4 |
| Toast on layout save failure | Task 7 (saveError) + Task 12 (rendering) |
| Persist df alongside report so generate-more works | Task 4 (step 5) |
| Backend test for build_report layout | Task 2 |
| Backend tests for PATCH layout | Task 3 |
| Backend tests for generate-more | Task 4 |
| Opt-in PDF export test | Task 5 |
| @dnd-kit install | Task 6 |
| Playwright install | Task 5 |
| Makefile test-pdf target | Task 5 |
| Manual smoke checklist | Task 14 |

No spec gaps.

