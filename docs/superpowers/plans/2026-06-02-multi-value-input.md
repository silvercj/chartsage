# Smarter Input Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recognize and chart multi-value/multi-label columns (split → top-N), and accept larger files via deterministic row-sampling.

**Architecture:** A new shared `multi_value.py` (`detect_multi_value`, `explode_multi_value`) is imported by both `profile.py` (to classify a delimited column as a usable multi-value categorical) and `chart_executor.py` (so the bar/treemap/pie executors split→explode→top-N when the column they're charting is multi-value — executors take only `(df, params)`, so they re-detect locally). A `sample_for_analysis` helper bounds rows for large files; the analyzed (sampled) CSV is what gets stored.

**Tech Stack:** Python + pandas (backend), Next.js (frontend upload page).

**Branch:** `multi-value-input` (off main, carries the spec). **Forbid subagents from `git checkout`/`switch`/`reset`/`stash`.** Interpreter: `venv/bin/python`.

---

## File Structure
- `src/api/multi_value.py` (create) — `detect_multi_value(series) -> str | None`, `explode_multi_value(series, delimiter) -> pd.Series`, the `_MV_*` constants.
- `src/api/schemas.py` (modify) — `ColumnInfo.multi_value`/`delimiter`; `DataProfile` prompt rendering shows multi-value columns.
- `src/api/profile.py` (modify) — `_profile_column` calls `detect_multi_value` before the `unusable` return.
- `src/api/chart_executor.py` (modify) — frequency_bar/treemap/pie split→explode→top-N when multi-value.
- `prompts/selection_system.txt` (modify) — one guidance line.
- `src/api/main.py` (modify) — 50 MB cap, `MAX_ANALYSIS_ROWS`, `sample_for_analysis`, store the sampled CSV, data-quality note.
- `src/app/app/page.tsx` (modify) — 50 MB check + copy + large-file note.
- Tests: `tests/unit/test_multi_value.py`, additions to `tests/unit/test_profile.py`, `tests/unit/test_chart_executor*` (or a new `test_multi_value_charts.py`), `tests/unit/test_sampling.py`.

---

## Phase 1 — Detection + schema (TDD)

### Task 1: `multi_value.py` shared helpers

**Files:** Create `src/api/multi_value.py`, `tests/unit/test_multi_value.py`.

- [ ] **Step 1: Failing test** — `tests/unit/test_multi_value.py`:
```python
import pandas as pd
from multi_value import detect_multi_value, explode_multi_value


def _col(values, reps):
    return pd.Series(values * reps)


def test_detects_comma_space_multi_value():
    s = pd.Series(["United States, India", "India, UK", "United States", "UK, Japan, India"] * 30)
    assert detect_multi_value(s) == ", "


def test_detects_pipe():
    s = pd.Series(["Drama|Comedy", "Comedy|Action", "Drama", "Action|Drama"] * 30)
    assert detect_multi_value(s) == "|"


def test_rejects_free_text_sentences():
    s = pd.Series([f"A unique, long descriptive sentence number {i} about a title." for i in range(200)])
    assert detect_multi_value(s) is None  # long atoms + near-unique


def test_rejects_too_many_atoms():
    # cast-like: thousands of distinct names -> exploded cardinality way over the cap
    s = pd.Series([f"Actor{i}, Actor{i+1}, Actor{i+2}" for i in range(400)])
    assert detect_multi_value(s) is None


def test_rejects_single_value_categorical():
    s = pd.Series(["TV-MA", "PG-13", "R", "TV-14"] * 50)
    assert detect_multi_value(s) is None


def test_explode_counts_atoms():
    s = pd.Series(["A, B", "B, C", "A"])
    atoms = explode_multi_value(s, ", ")
    assert atoms.tolist() == ["A", "B", "B", "C", "A"]
    assert atoms.value_counts()["A"] == 2 and atoms.value_counts()["B"] == 2
```

- [ ] **Step 2: Run → FAIL** (`venv/bin/python -m pytest tests/unit/test_multi_value.py -q`) — ImportError.

- [ ] **Step 3: Implement** `src/api/multi_value.py`:
```python
"""Detect and explode delimited multi-value / multi-label columns (e.g. a
Netflix `country`="United States, India" or `listed_in`="Dramas, Comedies").
Shared by the profiler (to mark the column usable) and the chart executors
(to split → count individual atoms). Conservative on purpose: never treat a
free-text column as multi-value."""
import pandas as pd

# Try more specific (whitespace-padded) delimiters first.
_MV_DELIMITERS = ["; ", " | ", "|", ", ", ","]
_MV_MAX_ATOMS = 50           # distinct atoms must be few enough to chart as top-N
_MV_MAX_ATOM_LEN = 30        # mean atom length — tags, not sentences
_MV_MAX_ATOM_LEN_HARD = 60   # no single atom may exceed this
_MV_MIN_MULTI_FRAC = 0.6     # >=60% of non-null cells must split into >=2 parts
_MV_MAX_ATOM_RATIO = 0.5     # distinct atoms / total atom occurrences (atoms must repeat)
_MV_MIN_ROWS = 10            # too few rows to judge


def detect_multi_value(series: pd.Series) -> str | None:
    s = series.dropna().astype(str)
    if len(s) < _MV_MIN_ROWS:
        return None
    for delim in _MV_DELIMITERS:
        parts = s.str.split(delim, regex=False)
        multi_frac = (parts.str.len() >= 2).mean()
        if multi_frac < _MV_MIN_MULTI_FRAC:
            continue
        atoms = parts.explode().str.strip()
        atoms = atoms[atoms != ""]
        total = len(atoms)
        if total == 0:
            continue
        distinct = atoms.nunique()
        if distinct == 0 or distinct > _MV_MAX_ATOMS:
            continue
        lengths = atoms.str.len()
        if lengths.mean() > _MV_MAX_ATOM_LEN or lengths.max() > _MV_MAX_ATOM_LEN_HARD:
            continue
        if (distinct / total) > _MV_MAX_ATOM_RATIO:
            continue
        return delim
    return None


def explode_multi_value(series: pd.Series, delimiter: str) -> pd.Series:
    atoms = series.dropna().astype(str).str.split(delimiter, regex=False).explode().str.strip()
    return atoms[atoms != ""]
```

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit**
```bash
git add src/api/multi_value.py tests/unit/test_multi_value.py
git commit -m "feat(input): multi_value detect + explode helpers"
```

---

### Task 2: Schema fields + prompt rendering

**Files:** Modify `src/api/schemas.py`. Test: `tests/unit/test_profile.py` (assertion added in Task 3).

- [ ] **Step 1: Add fields.** In `ColumnInfo` (after `unusable_reason`):
```python
    multi_value: bool = False
    delimiter: Optional[str] = None
```

- [ ] **Step 2: Render multi-value in the profile prompt.** In `DataProfile`'s prompt-building method (the loop that appends `f"- {c.name}: role={c.role}, ..."` with a per-role detail; the `categorical`+`top_values` branch is the model), make the categorical branch note multi-value. Change the categorical rendering so that when `c.multi_value` is true it reads, e.g.:
```python
            elif c.role == "categorical" and c.top_values:
                top = ", ".join(f"{v}={n}" for v, n in c.top_values[:5])
                if c.multi_value:
                    parts.append(f"  multi-value (split on '{c.delimiter}') — top values: {top}")
                else:
                    parts.append(f"  top values: {top}")
```
(Match the file's actual formatting; the point is the model sees a "(multi-value)" marker + the atom top values.)

- [ ] **Step 3: Commit** (with Task 3, since the assertion lives there). Proceed to Task 3.

---

### Task 3: Profiler classifies multi-value columns

**Files:** Modify `src/api/profile.py`. Test: `tests/unit/test_profile.py`.

- [ ] **Step 1: Failing test** — add to `tests/unit/test_profile.py`:
```python
def test_multi_value_country_is_usable_categorical():
    import pandas as pd
    from profile import profile_dataframe
    n = 400
    countries = ["United States", "India", "United Kingdom", "Japan", "Canada", "France", "Spain", "Mexico"]
    import random; random.seed(1)
    def multi(pool, lo, hi): return ", ".join(random.sample(pool, random.randint(lo, hi)))
    df = pd.DataFrame({
        "country": [multi(countries, 1, 3) for _ in range(n)],
        "cast": [", ".join(f"Actor{random.randint(0,3000)}" for _ in range(4)) for _ in range(n)],
        "description": [f"A unique, long sentence number {i} describing the title at length." for i in range(n)],
        "rating": [random.choice(["TV-MA", "PG-13", "R", "TV-14"]) for _ in range(n)],
    })
    prof = profile_dataframe(df)
    by = {c.name: c for c in prof.columns}
    assert by["country"].role == "categorical" and by["country"].multi_value is True
    assert by["country"].delimiter == ", "
    assert by["cast"].role == "unusable"          # thousands of distinct atoms
    assert by["description"].role == "unusable"   # long near-unique sentences
    assert by["rating"].role == "categorical" and by["rating"].multi_value is False
```

- [ ] **Step 2: Run → FAIL** (`country` is `unusable`, no `multi_value`).

- [ ] **Step 3: Implement.** In `src/api/profile.py`: add `from multi_value import detect_multi_value, explode_multi_value` at the top. In `_profile_column`, **between** the high-cardinality-categorical branch (ends line ~111) and the final `unusable` return (line ~113), insert:
```python
    # Multi-value / multi-label column (e.g. "United States, India" or "Drama|Comedy"):
    # split into atoms and treat as a usable categorical charted as top-N of the atoms.
    delim = detect_multi_value(series)
    if delim is not None:
        atoms = explode_multi_value(series, delim)
        top = atoms.value_counts().head(5)
        return ColumnInfo(
            name=name, dtype=dtype, role="categorical",
            cardinality=int(atoms.nunique()), null_count=null_count,
            top_values=[(k, int(v)) for k, v in top.items()],
            multi_value=True, delimiter=delim,
        )
```

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Full suite + commit**
```bash
venv/bin/python -m pytest tests/unit tests/integration -p no:warnings -q   # all pass
git add src/api/schemas.py src/api/profile.py tests/unit/test_profile.py
git commit -m "feat(input): profiler classifies delimited multi-value columns as usable categoricals"
```

---

## Phase 2 — Multi-value charting (TDD)

### Task 4: bar / treemap / pie executors split multi-value columns

**Files:** Modify `src/api/chart_executor.py`. Test: `tests/unit/test_multi_value_charts.py` (create).

- [ ] **Step 1: Failing test** — `tests/unit/test_multi_value_charts.py`:
```python
import pandas as pd
from chart_executor import execute_frequency_bar_chart


def test_frequency_bar_explodes_multi_value():
    # 'United States' appears in 3 of 4 rows; raw value_counts would never see it as 3.
    df = pd.DataFrame({"country": [
        "United States, India", "United States, UK", "India", "United States, Japan, India",
    ] * 30})
    spec = execute_frequency_bar_chart(df, {"column": "country", "title": "Top countries", "intent": "x"})
    assert not hasattr(spec, "reason"), getattr(spec, "reason", "")
    pairs = dict(zip(spec.x, spec.y))
    assert pairs["United States"] == 90   # 3 per block * 30
    assert pairs["India"] == 90
    assert pairs["Japan"] == 30
```

- [ ] **Step 2: Run → FAIL** (raw value_counts sees combos like "United States, India", and `United States` alone = 30, not 90).

- [ ] **Step 3: Implement.** In `src/api/chart_executor.py`: add `from multi_value import detect_multi_value, explode_multi_value` at the top. In `execute_frequency_bar_chart`, replace the `series = df[column].dropna()` / `counts = series.value_counts()` / `> MAX_CATEGORIES` block with multi-value awareness:
```python
    raw = df[column].dropna()
    if len(raw) == 0:
        return _err(f"'{column}' has no non-null values.")

    delim = detect_multi_value(df[column])
    if delim is not None:
        atoms = explode_multi_value(df[column], delim)
        counts = atoms.value_counts().head(MAX_CATEGORIES)   # top-N atoms; no error
        series = atoms
    else:
        series = raw
        counts = series.value_counts()
        if len(counts) > MAX_CATEGORIES:
            return _err(
                f"'{column}' has {len(counts)} unique values, more than the max ({MAX_CATEGORIES}). "
                f"Use frequency charts on lower-cardinality columns; this column may be an identifier."
            )
```
(The rest — building `x`, `y`, the `ChartSpec` — is unchanged; `data_point_count=int(len(series))`.) Apply the **same pattern** to the treemap executor and the pie executor (find them in the file: locate the `value_counts()` call in each, and gate it on `detect_multi_value(df[column])` → `explode_multi_value` → `head(MAX_CATEGORIES)` for treemap / `head(MAX_PIE_SLICES)` rollup for pie). Do NOT change aggregation_bar / scatter / histogram / etc.

- [ ] **Step 4: Run → PASS** (the bar test; add an analogous treemap test if quick).

- [ ] **Step 5: Full suite + commit**
```bash
venv/bin/python -m pytest tests/unit tests/integration -p no:warnings -q
git add src/api/chart_executor.py tests/unit/test_multi_value_charts.py
git commit -m "feat(input): bar/treemap/pie explode multi-value columns into top-N atoms"
```

---

## Phase 3 — Selection prompt

### Task 5: Prompt guidance line

**Files:** Modify `prompts/selection_system.txt`.

- [ ] **Step 1:** After the high-cardinality / "rich-but-messy columns" guidance, add one line:
```
- Some categorical columns are multi-value / tag columns (marked "multi-value" in the profile — e.g. genres, countries, tags, cast). Chart them with frequency_bar_chart or treemap_chart; the values shown will be the top individual tags, not the raw combined cells.
```
Leave all other rules intact.

- [ ] **Step 2: Full suite (any prompt-structure test stays green) + commit**
```bash
venv/bin/python -m pytest tests/unit tests/integration -p no:warnings -q
git add prompts/selection_system.txt
git commit -m "feat(input): prompt guidance for charting multi-value columns"
```

---

## Phase 4 — Larger files + row-sampling

### Task 6: `sample_for_analysis` + 50 MB cap + store the sampled CSV

**Files:** Modify `src/api/main.py`. Test: `tests/unit/test_sampling.py` (create).

- [ ] **Step 1: Failing test** — `tests/unit/test_sampling.py`:
```python
import pandas as pd
from main import sample_for_analysis, MAX_ANALYSIS_ROWS


def test_large_frame_sampled_deterministically():
    df = pd.DataFrame({"a": range(120_000), "b": range(120_000)})
    out, sampled, total = sample_for_analysis(df)
    assert sampled is True and total == 120_000
    assert len(out) == MAX_ANALYSIS_ROWS
    # deterministic
    out2, _, _ = sample_for_analysis(df)
    assert out.equals(out2)


def test_small_frame_untouched():
    df = pd.DataFrame({"a": range(100)})
    out, sampled, total = sample_for_analysis(df)
    assert sampled is False and total == 100 and len(out) == 100
```

- [ ] **Step 2: Run → FAIL** (no `sample_for_analysis`).

- [ ] **Step 3: Implement.** In `src/api/main.py`: add `MAX_ANALYSIS_ROWS = int(os.environ.get("MAX_ANALYSIS_ROWS", "50000"))` near the other limits, and raise the upload cap to 50 MB (`MAX_UPLOAD_BYTES = 50 * 1024 * 1024` — update wherever it's defined). Add:
```python
def sample_for_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, int]:
    """Bound rows for analysis. Returns (frame, was_sampled, original_row_count).
    Deterministic so generate-more / deepen (which re-read the stored CSV) match."""
    total = len(df)
    if total > MAX_ANALYSIS_ROWS:
        return df.sample(n=MAX_ANALYSIS_ROWS, random_state=0).reset_index(drop=True), True, total
    return df, False, total
```
Then in `generate_report`, right after the DataFrame is loaded + columns lowercased (and before profiling), apply it and store the **sampled** CSV:
```python
    df, was_sampled, total_rows = sample_for_analysis(df)
    # Store what we analyzed (so generate-more/deepen are consistent + storage bounded).
    if was_sampled:
        content = df.to_csv(index=False).encode("utf-8")
```
(`content` is the bytes uploaded to storage later — confirm the storage call uploads `content`; if it uploads the original file bytes under a different variable, re-serialize there instead.) After the report is built, if `was_sampled`, prepend a data-quality note:
```python
    if was_sampled:
        note = f"Analyzed a representative random sample of {MAX_ANALYSIS_ROWS:,} of {total_rows:,} rows."
        report.data_quality = [note] + list(report.data_quality or [])
```
(Place this where `report` exists and before it's serialized/saved; match the actual attribute — `report.data_quality`.)

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Full suite + commit**
```bash
venv/bin/python -m pytest tests/unit tests/integration -p no:warnings -q
git add src/api/main.py tests/unit/test_sampling.py
git commit -m "feat(input): 50MB cap + deterministic row-sampling for large files"
```

---

### Task 7: Frontend — 50 MB cap + large-file note

**Files:** Modify `src/app/app/page.tsx`.

- [ ] **Step 1:** Change the size check (`~line 54`) from `10 * 1024 * 1024` to `50 * 1024 * 1024` and the message to `'File must be under 50MB.'`. Update the helper copy (`~line 203`) `up to 10 MB` → `up to 50 MB`. Add a note shown when a selected file is large (e.g. `f.size > 10 * 1024 * 1024`): a small line "Large file — we'll analyze a representative sample." (use a state flag + render near the preview, semantic classes consistent with the page).

- [ ] **Step 2: tsc + commit**
```bash
npx tsc --noEmit   # exit 0
git add src/app/app/page.tsx
git commit -m "feat(input): raise upload cap to 50MB + large-file sample note"
```

---

## Phase 5 — Build, QA, deploy (production — requires explicit user authorization)

### Task 8: Verify on real Netflix data + deploy

**Files:** none (ops).

- [ ] **Step 1: Full verification**
```bash
cd /Users/chrissilver/Documents/ChartSage
venv/bin/python -m pytest tests/unit tests/integration -p no:warnings -q   # all pass
npx tsc --noEmit && npm run build                                          # clean
```

- [ ] **Step 2: Independent review (subagent)** over `main..multi-value-input`, focused on the **false-positive risk** in `detect_multi_value` (could a free-text / address / notes column be wrongly flagged multi-value and produce a junk chart?), the executor top-N correctness, and the sampling determinism + that the stored CSV is the sampled one.

- [ ] **Step 3: Local real-data check.** Obtain an actual `netflix_titles.csv` — try `curl -sL -o /tmp/netflix.csv "https://raw.githubusercontent.com/datasciencedojo/datasets/master/netflix_titles.csv"` (or any reachable mirror); if unreachable, reconstruct faithfully (genuine comma'd `country`/`listed_in`/`cast`, near-unique `title`/`description`). Profile it locally (`sys.path.insert(0,'src/api'); from profile import profile_dataframe`) and confirm `country` + `listed_in` are `multi_value` categorical, `cast`/`title`/`description` `unusable`.

- [ ] **Step 4: Deploy backend** (user-authorized):
```bash
SUPA=$(grep -E '^SUPABASE_URL=' .env | cut -d= -f2-)
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_SUPABASE_URL="$SUPA",_TAG="$(git rev-parse --short HEAD)"
```
(Do NOT source `FRONTEND_BASE_URL` — the cloudbuild default is correct.)

- [ ] **Step 5: Live verify** against the deployed backend with the real `netflix_titles.csv`: `POST /generate-report` (fresh anon id + X-Forwarded-For), fetch `/report/{id}`, confirm chart count reaches ~8–10 and that country/genre charts are present. Also a >50k-row CSV → confirm the sample note appears.

- [ ] **Step 6: Deploy frontend** — merge `multi-value-input` → main + push (Vercel); verify the build goes Ready.

---

## Self-Review

**Spec coverage:** multi_value detect/explode (Task 1) ✓ · schema fields + prompt render (Task 2) ✓ · profiler classification before unusable (Task 3) ✓ · bar/treemap/pie split→top-N (Task 4) ✓ · prompt line (Task 5) ✓ · 50 MB cap + deterministic sampling + store sampled CSV + data-quality note (Task 6) ✓ · frontend cap+note (Task 7) ✓ · real-netflix verification + deploy + false-positive review (Task 8) ✓.

**Placeholder scan:** concrete code + thresholds throughout; the two "match the file's actual formatting/attribute" notes (the profile prompt branch in Task 2, the storage-variable + `data_quality` attribute in Task 6) are explicit verify-against-real-code instructions, not vague hand-waving — the implementer reads the file and matches.

**Type consistency:** `detect_multi_value(series)->str|None`, `explode_multi_value(series, delimiter)->Series`, `ColumnInfo.multi_value`/`delimiter`, `sample_for_analysis(df)->(df,bool,int)`, `MAX_ANALYSIS_ROWS`, `MAX_CATEGORIES`/`MAX_PIE_SLICES` are used identically across `multi_value.py`, `profile.py`, `chart_executor.py`, `main.py`, and the tests.
