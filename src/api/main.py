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
from schemas import ChartLayoutEntry, Report


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
    # Persist the source data so /generate-more can re-execute tool calls
    df_csv = df.to_csv(index=False)
    r.set(f"report:{session_id}:df", df_csv, ex=SESSION_TTL_SECONDS)

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

    df_blob = r.get(f"report:{session_id}:df")
    if not df_blob:
        raise HTTPException(
            status_code=404,
            detail="Source data for this report has expired. Generate a new report.",
        )

    import io as _io
    df = pd.read_csv(_io.StringIO(df_blob))
    df.columns = [str(c).lower() for c in df.columns]

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


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
