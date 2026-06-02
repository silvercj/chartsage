"""FastAPI app — /generate-report, /report/{id}/*, /export.pdf."""
import hashlib
import io
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

import pandas as pd
import stripe
from anthropic import APIStatusError
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from alerting import report_alert
from claude_client import ClaudeClient, RetryableBusy
from credits import ADD_CHART_COST, DEEP_ANALYSIS_COST, GENERATE_MORE_COST, REPORT_COST, SIGNUP_GRANT, InsufficientCredits
from billing import public_catalogue
from db import SupabaseDB
from deps import Identity, get_identity, require_admin
from pydantic import BaseModel
from llm_config import MODEL_NARRATIVE, MODEL_SELECTION, estimate_cost_usd
from posthog_server import PostHogServer
from profile import profile_dataframe
import report_export
from report_generator import ReportGenerator
from chart_tools import CHART_TOOLS
from schemas import ChartLayoutEntry, ChartWithCaption, Report
from storage import StorageError, SupabaseStorage


load_dotenv()


MAX_UPLOAD_BYTES = 50 * 1024 * 1024
# Above this row count we analyze a deterministic random sample (the analysis is
# column-driven, so a representative sample is statistically faithful while keeping
# memory/latency bounded). Env-overridable.
MAX_ANALYSIS_ROWS = int(os.environ.get("MAX_ANALYSIS_ROWS", "50000"))
ALLOWED_EXTENSIONS = (".csv", ".xlsx")
# Free reports per anonymous visitor (env-overridable).
ANON_REPORT_LIMIT = int(os.environ.get("ANON_REPORT_LIMIT", "1"))
# Anon IDs that bypass the limit entirely (comma-separated) — for owner/QA testing.
UNLIMITED_ANON_IDS = {
    x.strip() for x in os.environ.get("UNLIMITED_ANON_IDS", "").split(",") if x.strip()
}
# Soft-launch daily abuse/cost guards for anonymous reports (env-overridable).
ANON_IP_DAILY_CAP = int(os.environ.get("ANON_IP_DAILY_CAP", "5"))
ANON_GLOBAL_DAILY_CAP = int(os.environ.get("ANON_GLOBAL_DAILY_CAP", "200"))


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


# ---- Sentry (no-op unless SENTRY_DSN is set) -------------------------------

_SENTRY_DSN = os.environ.get("SENTRY_DSN", "")
if _SENTRY_DSN:
    import sentry_sdk
    sentry_sdk.init(dsn=_SENTRY_DSN, traces_sample_rate=0.0, send_default_pii=False)


# ---- Stripe (payments / credit packs) --------------------------------------
stripe.api_key = os.environ.get("STRIPE_SECRET_KEY") or None
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
_FRONTEND_BASE = os.environ.get("FRONTEND_BASE_URL", "http://localhost:3000")
# The signature-verification error moved to the top level in newer stripe-python;
# import it compatibly so the webhook catch clause works across versions.
try:
    from stripe import SignatureVerificationError
except ImportError:  # pragma: no cover - older stripe layout
    from stripe.error import SignatureVerificationError


# ---- App -------------------------------------------------------------------

app = FastAPI(title="ChartSage v2")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://chartsage(-[a-z0-9-]+)?\.vercel\.app|https://(www\.)?chartsage\.app|http://localhost:3000|http://localhost:3001",
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


def sample_for_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, int]:
    """Bound rows for analysis. Returns (frame, was_sampled, original_row_count).
    Deterministic (fixed seed) so generate-more / deepen — which re-read the stored
    CSV — analyze the same rows."""
    total = len(df)
    if total > MAX_ANALYSIS_ROWS:
        return df.sample(n=MAX_ANALYSIS_ROWS, random_state=0).reset_index(drop=True), True, total
    return df, False, total


def _title_from_summary(summary: str) -> str:
    first_sentence = summary.split(".")[0].strip() if summary else ""
    return first_sentence[:200] if first_sentence else "Untitled report"


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else ""


def _client_fingerprint(request: Request) -> str:
    ua = request.headers.get("user-agent", "")
    lang = request.headers.get("accept-language", "")
    return hashlib.sha256(f"{ua}|{lang}".encode()).hexdigest()[:16]


def _ensure_profile_tracked(db: SupabaseDB, posthog: PostHogServer, identity: Identity) -> int:
    """Ensure the user's profile + one-time starter grant, firing the PostHog
    `credits_granted` event exactly once (on first creation). Race-tolerant: a
    rare concurrent first-request could double-fire the event, which is harmless."""
    is_new = not db.profile_exists(identity.user_id)
    balance = db.ensure_profile(identity.user_id, SIGNUP_GRANT)
    if is_new:
        posthog.capture(identity.distinct_id, "credits_granted",
                        {"amount": SIGNUP_GRANT, "balance": balance})
    return balance


# ---- Endpoints -------------------------------------------------------------

@app.post("/generate-report")
async def generate_report(
    request: Request,
    file: UploadFile = File(...),
    custom_prompt: str | None = Form(None),
    deep: bool = Form(False),
    identity: Identity = Depends(get_identity),
    claude: ClaudeClient = Depends(get_claude_client),
    db: SupabaseDB = Depends(get_db),
    storage: SupabaseStorage = Depends(get_storage),
    posthog: PostHogServer = Depends(get_posthog),
):
    run_id, _ = setup_run_logging()
    started = time.perf_counter()
    client_ip = _client_ip(request)
    fingerprint = _client_fingerprint(request)

    # Deep analysis is a paid feature: it always costs DEEP_ANALYSIS_COST, requires
    # auth (anon gets the upsell, never the free-report path), and debits under the
    # "deep_analysis" reason. Non-deep keeps the existing REPORT_COST/"report" flow.
    report_cost = DEEP_ANALYSIS_COST if deep else REPORT_COST
    spend_reason = "deep_analysis" if deep else "report"

    # Identity-aware gating: anonymous visitors keep the 1-free-report cap
    # (with an owner/QA allowlist bypass); authenticated users are credit-gated.
    if not identity.is_authenticated:
        if deep:
            posthog.capture(identity.distinct_id, "upgrade_required",
                            {"action": "deep_analysis"})
            raise HTTPException(
                status_code=402,
                detail={"code": "UPGRADE_REQUIRED",
                        "message": "Create a free account to run a deep analysis."},
            )
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
    else:
        try:
            balance = _ensure_profile_tracked(db, posthog, identity)
        except Exception:
            logging.exception("ensure_profile failed")
            raise HTTPException(status_code=503, detail={
                "code": "CREDITS_UNAVAILABLE",
                "message": "Couldn't check your credits right now. Please retry."})
        if balance < report_cost:
            posthog.capture(identity.distinct_id, "out_of_credits",
                            {"action": spend_reason, "balance": balance})
            raise HTTPException(status_code=402, detail={
                "code": "OUT_OF_CREDITS",
                "message": "You're out of credits.",
                "balance": balance, "cost": report_cost,
            })

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

    df, was_sampled, total_rows = sample_for_analysis(df)
    if was_sampled:
        content = df.to_csv(index=False).encode("utf-8")   # store what we analyzed

    if df.shape[1] < 2:
        raise HTTPException(status_code=422, detail="File must have at least 2 columns to chart.")
    if df.shape[0] < 1:
        raise HTTPException(status_code=422, detail="File has no data rows.")

    posthog.capture(identity.distinct_id, "report_generation_started", {
        "rowCount": int(df.shape[0]),
        "columnCount": int(df.shape[1]),
        "filename": file.filename,
        "sizeBytes": len(content),
        "deep": deep,
        "customPrompt": bool(custom_prompt),
    })

    try:
        profile = profile_dataframe(df)
        gen = ReportGenerator(
            profile=profile, df=df, claude=claude,
            model_selection=MODEL_SELECTION, model_narrative=MODEL_NARRATIVE,
            custom_prompt=custom_prompt,
        )
        report = gen.build_report(deep=deep)
    except RetryableBusy:
        posthog.capture(identity.distinct_id, "claude_overloaded", {"stage": "selection"})
        raise HTTPException(status_code=503, detail={
            "code": "BUSY",
            "message": "Claude is busy. Please retry in 30 seconds.",
        })
    except APIStatusError as e:
        logging.exception("Claude API error")
        posthog.capture(identity.distinct_id, "report_generation_failed", {
            "reason": "claude_api_status_error",
            "errorClass": type(e).__name__,
            "httpStatus": getattr(getattr(e, "response", None), "status_code", 0),
            "elapsedMs": int((time.perf_counter() - started) * 1000),
        })
        raise HTTPException(status_code=502, detail=f"Upstream model error: {e}")
    except Exception as e:
        logging.exception("Report generation failed")
        posthog.capture(identity.distinct_id, "report_generation_failed", {
            "reason": "internal",
            "errorClass": type(e).__name__,
            "httpStatus": 500,
            "elapsedMs": int((time.perf_counter() - started) * 1000),
        })
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    if was_sampled:
        _note = f"Analyzed a representative random sample of {MAX_ANALYSIS_ROWS:,} of {total_rows:,} rows."
        report.data_quality = [_note] + list(report.data_quality or [])

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
            anon_id=identity.anon_id,
            user_id=identity.user_id,
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

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    in_tok = report.metadata.get("input_tokens_total", 0) if isinstance(report.metadata, dict) else 0
    out_tok = report.metadata.get("output_tokens_total", 0) if isinstance(report.metadata, dict) else 0
    cache_tok = report.metadata.get("cache_read_input_tokens_total", 0) if isinstance(report.metadata, dict) else 0

    posthog.capture(identity.distinct_id, "report_generation_succeeded", {
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
        "deep": deep,
        "customPrompt": bool(custom_prompt),
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


class UpgradeIntentIn(BaseModel):
    email: str | None = None


@app.get("/me")
async def me(
    identity: Identity = Depends(get_identity),
    db: SupabaseDB = Depends(get_db),
    posthog: PostHogServer = Depends(get_posthog),
):
    if not identity.is_authenticated:
        raise HTTPException(status_code=401, detail={
            "code": "AUTH_REQUIRED", "message": "Sign in required."})
    try:
        balance = _ensure_profile_tracked(db, posthog, identity)
    except Exception:
        logging.exception("ensure_profile failed")
        raise HTTPException(status_code=503, detail={
            "code": "CREDITS_UNAVAILABLE",
            "message": "Couldn't check your credits right now. Please retry."})
    return {"credits_balance": balance}


@app.get("/credits/history")
async def credits_history(
    identity: Identity = Depends(get_identity),
    db: SupabaseDB = Depends(get_db),
):
    if not identity.is_authenticated:
        raise HTTPException(status_code=401, detail={
            "code": "AUTH_REQUIRED", "message": "Sign in required."})
    return db.list_transactions(identity.user_id)


@app.post("/upgrade-intent")
async def upgrade_intent(
    body: UpgradeIntentIn,
    identity: Identity = Depends(get_identity),
    db: SupabaseDB = Depends(get_db),
    posthog: PostHogServer = Depends(get_posthog),
):
    if not identity.is_authenticated:
        raise HTTPException(status_code=401, detail={
            "code": "AUTH_REQUIRED", "message": "Sign in required."})
    db.record_upgrade_intent(identity.user_id, body.email)
    posthog.capture(identity.distinct_id, "upgrade_intent_captured", {})
    return {"ok": True}


@app.post("/report/{session_id}/generate-more")
async def generate_more(
    session_id: str,
    identity: Identity = Depends(get_identity),
    claude: ClaudeClient = Depends(get_claude_client),
    db: SupabaseDB = Depends(get_db),
    storage: SupabaseStorage = Depends(get_storage),
    posthog: PostHogServer = Depends(get_posthog),
):
    if not identity.is_authenticated:
        raise HTTPException(
            status_code=402,
            detail={"code": "UPGRADE_REQUIRED",
                    "message": "Create a free account to generate more charts."},
        )
    try:
        balance = _ensure_profile_tracked(db, posthog, identity)
    except Exception:
        logging.exception("ensure_profile failed")
        raise HTTPException(status_code=503, detail={
            "code": "CREDITS_UNAVAILABLE",
            "message": "Couldn't check your credits right now. Please retry."})
    if balance < GENERATE_MORE_COST:
        posthog.capture(identity.distinct_id, "out_of_credits",
                        {"action": "generate_more", "balance": balance})
        raise HTTPException(status_code=402, detail={
            "code": "OUT_OF_CREDITS",
            "message": "You're out of credits.",
            "balance": balance, "cost": GENERATE_MORE_COST,
        })
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
        csv_bytes = storage.download_by_key(csv_key)
    except StorageError:
        raise HTTPException(status_code=404, detail={
            "code": "SOURCE_DATA_UNAVAILABLE",
            "message": "Source data for this report is no longer available.",
        })

    df = pd.read_csv(io.StringIO(csv_bytes.decode("utf-8")))
    df.columns = [str(c).lower() for c in df.columns]

    posthog.capture(identity.distinct_id, "generate_more_started", {
        "reportId": session_id,
        "existingChartCount": len(existing_report.charts),
    })

    profile = profile_dataframe(df)
    persisted_prompt = row["report_json"].get("metadata", {}).get("custom_prompt")
    gen = ReportGenerator(
        profile=profile, df=df, claude=claude,
        model_selection=MODEL_SELECTION, model_narrative=MODEL_NARRATIVE,
        custom_prompt=persisted_prompt,
    )

    try:
        new_charts, new_layout = gen.generate_more(existing_report.charts)
    except RetryableBusy:
        posthog.capture(identity.distinct_id, "generate_more_failed", {
            "reportId": session_id, "reason": "busy", "httpStatus": 503,
            "elapsedMs": int((time.perf_counter() - started) * 1000),
        })
        raise HTTPException(status_code=503, detail={
            "code": "BUSY",
            "message": "Claude is busy. Try again in 30s.",
        })
    except Exception as e:
        posthog.capture(identity.distinct_id, "generate_more_failed", {
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
    try:
        new_balance = db.spend_credits(identity.user_id, GENERATE_MORE_COST, "generate_more", session_id)
        posthog.capture(identity.distinct_id, "credits_spent",
                        {"amount": GENERATE_MORE_COST, "balance": new_balance, "reason": "generate_more"})
    except InsufficientCredits:
        logging.warning("Credit spend lost a race on generate-more %s; serving free.", session_id)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    posthog.capture(identity.distinct_id, "generate_more_succeeded", {
        "reportId": session_id,
        "newChartCount": len(new_charts),
        "inputTokens": 0,    # tokens from sub-Claude call aren't surfaced through gen yet
        "outputTokens": 0,
        "estCostUsd": 0.005, # rough — 1 selection pass on Haiku
        "elapsedMs": elapsed_ms,
    })

    return JSONResponse(content=report_dict, status_code=200)


class AddChartIn(BaseModel):
    mode: Literal["type", "describe"]
    chart_type: str | None = None
    prompt: str | None = None


# Tool names a user may force via mode="type": every chart-producing tool
# (key_metrics is a stat band, not a chart, so it's excluded).
_ADD_CHART_TYPES = {t["name"] for t in CHART_TOOLS if t["name"] != "key_metrics"}


@app.post("/report/{session_id}/add-chart")
async def add_chart(
    session_id: str,
    body: AddChartIn,
    identity: Identity = Depends(get_identity),
    claude: ClaudeClient = Depends(get_claude_client),
    db: SupabaseDB = Depends(get_db),
    storage: SupabaseStorage = Depends(get_storage),
    posthog: PostHogServer = Depends(get_posthog),
):
    if not identity.is_authenticated:
        raise HTTPException(
            status_code=402,
            detail={"code": "UPGRADE_REQUIRED",
                    "message": "Create a free account to add a chart."},
        )
    try:
        balance = _ensure_profile_tracked(db, posthog, identity)
    except Exception:
        logging.exception("ensure_profile failed")
        raise HTTPException(status_code=503, detail={
            "code": "CREDITS_UNAVAILABLE",
            "message": "Couldn't check your credits right now. Please retry."})
    if balance < ADD_CHART_COST:
        posthog.capture(identity.distinct_id, "out_of_credits",
                        {"action": "add_chart", "balance": balance})
        raise HTTPException(status_code=402, detail={
            "code": "OUT_OF_CREDITS",
            "message": "You're out of credits.",
            "balance": balance, "cost": ADD_CHART_COST,
        })

    if body.mode == "type" and body.chart_type not in _ADD_CHART_TYPES:
        raise HTTPException(status_code=422, detail={
            "code": "INVALID_CHART_TYPE",
            "message": f"Unknown chart type '{body.chart_type}'.",
        })

    started = time.perf_counter()
    row = db.get_report(session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Report not found.")

    csv_key = row.get("csv_storage_key")
    if not csv_key:
        raise HTTPException(status_code=404, detail={
            "code": "SOURCE_DATA_UNAVAILABLE",
            "message": "Source data for this report is no longer available.",
        })
    try:
        csv_bytes = storage.download_by_key(csv_key)
    except StorageError:
        raise HTTPException(status_code=404, detail={
            "code": "SOURCE_DATA_UNAVAILABLE",
            "message": "Source data for this report is no longer available.",
        })

    df = pd.read_csv(io.StringIO(csv_bytes.decode("utf-8")))
    df.columns = [str(c).lower() for c in df.columns]

    posthog.capture(identity.distinct_id, "add_chart_started", {
        "reportId": session_id, "mode": body.mode,
        "chartType": body.chart_type if body.mode == "type" else None,
    })

    profile = profile_dataframe(df)
    persisted_prompt = row["report_json"].get("metadata", {}).get("custom_prompt")
    gen = ReportGenerator(
        profile=profile, df=df, claude=claude,
        model_selection=MODEL_SELECTION, model_narrative=MODEL_NARRATIVE,
        custom_prompt=persisted_prompt,
    )

    try:
        cwc = gen.add_chart(mode=body.mode, chart_type=body.chart_type, prompt=body.prompt)
    except RetryableBusy:
        posthog.capture(identity.distinct_id, "add_chart_failed", {
            "reportId": session_id, "reason": "busy", "httpStatus": 503,
            "elapsedMs": int((time.perf_counter() - started) * 1000),
        })
        raise HTTPException(status_code=503, detail={
            "code": "BUSY",
            "message": "Claude is busy. Try again in 30s.",
        })
    except Exception as e:
        posthog.capture(identity.distinct_id, "add_chart_failed", {
            "reportId": session_id, "reason": "internal",
            "errorClass": type(e).__name__,
            "elapsedMs": int((time.perf_counter() - started) * 1000),
        })
        raise

    # No chart produced -> 422, and no debit (mirrors the no-debit-on-failure rule).
    if cwc is None:
        posthog.capture(identity.distinct_id, "add_chart_failed", {
            "reportId": session_id, "reason": "no_chart", "httpStatus": 422,
            "elapsedMs": int((time.perf_counter() - started) * 1000),
        })
        raise HTTPException(status_code=422, detail={
            "code": "NO_CHART",
            "message": "Couldn't build that chart — try another type or description.",
        })

    report_dict = row["report_json"]
    sidebar_max = max(
        (e["order"] for e in report_dict["layout"] if e["position"] == "sidebar"),
        default=-1,
    )
    layout_entry = ChartLayoutEntry(chart_id=cwc.chart_id, position="sidebar", order=sidebar_max + 1)
    report_dict["charts"].append(cwc.model_dump())
    report_dict["layout"].append(layout_entry.model_dump())

    db.update_report_json(session_id, report_dict)
    try:
        new_balance = db.spend_credits(identity.user_id, ADD_CHART_COST, "add_chart", session_id)
        posthog.capture(identity.distinct_id, "credits_spent",
                        {"amount": ADD_CHART_COST, "balance": new_balance, "reason": "add_chart"})
    except InsufficientCredits:
        logging.warning("Credit spend lost a race on add-chart %s; serving free.", session_id)

    posthog.capture(identity.distinct_id, "add_chart_succeeded", {
        "reportId": session_id, "mode": body.mode,
        "chartKind": cwc.spec.kind,
        "elapsedMs": int((time.perf_counter() - started) * 1000),
    })

    return JSONResponse(content=report_dict, status_code=200)


@app.post("/report/{session_id}/deepen")
async def deepen_report(
    session_id: str,
    identity: Identity = Depends(get_identity),
    claude: ClaudeClient = Depends(get_claude_client),
    db: SupabaseDB = Depends(get_db),
    storage: SupabaseStorage = Depends(get_storage),
    posthog: PostHogServer = Depends(get_posthog),
):
    """Deep-analysis on an existing report: run the iterative-deepening loop over the
    current charts, append any new ones, and re-narrate over the full enriched set.
    Mirrors generate_more's gate -> work -> append -> debit-after-save pattern.
    Paid feature: auth required, costs DEEP_ANALYSIS_COST, debited only on success.
    """
    if not identity.is_authenticated:
        raise HTTPException(
            status_code=402,
            detail={"code": "UPGRADE_REQUIRED",
                    "message": "Create a free account to run a deep analysis."},
        )
    try:
        balance = _ensure_profile_tracked(db, posthog, identity)
    except Exception:
        logging.exception("ensure_profile failed")
        raise HTTPException(status_code=503, detail={
            "code": "CREDITS_UNAVAILABLE",
            "message": "Couldn't check your credits right now. Please retry."})
    if balance < DEEP_ANALYSIS_COST:
        posthog.capture(identity.distinct_id, "out_of_credits",
                        {"action": "deep_analysis", "balance": balance})
        raise HTTPException(status_code=402, detail={
            "code": "OUT_OF_CREDITS",
            "message": "You're out of credits.",
            "balance": balance, "cost": DEEP_ANALYSIS_COST,
        })

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
        csv_bytes = storage.download_by_key(csv_key)
    except StorageError:
        raise HTTPException(status_code=404, detail={
            "code": "SOURCE_DATA_UNAVAILABLE",
            "message": "Source data for this report is no longer available.",
        })

    df = pd.read_csv(io.StringIO(csv_bytes.decode("utf-8")))
    df.columns = [str(c).lower() for c in df.columns]

    posthog.capture(identity.distinct_id, "deepen_started", {
        "reportId": session_id,
        "existingChartCount": len(existing_report.charts),
    })

    profile = profile_dataframe(df)
    persisted_prompt = row["report_json"].get("metadata", {}).get("custom_prompt")
    gen = ReportGenerator(
        profile=profile, df=df, claude=claude,
        model_selection=MODEL_SELECTION, model_narrative=MODEL_NARRATIVE,
        custom_prompt=persisted_prompt,
    )

    try:
        extra = gen.deepen([c.spec for c in existing_report.charts])
    except RetryableBusy:
        posthog.capture(identity.distinct_id, "deepen_failed", {
            "reportId": session_id, "reason": "busy", "httpStatus": 503,
            "elapsedMs": int((time.perf_counter() - started) * 1000),
        })
        raise HTTPException(status_code=503, detail={
            "code": "BUSY",
            "message": "Claude is busy. Try again in 30s.",
        })
    except Exception as e:
        posthog.capture(identity.distinct_id, "deepen_failed", {
            "reportId": session_id, "reason": "internal",
            "errorClass": type(e).__name__,
            "elapsedMs": int((time.perf_counter() - started) * 1000),
        })
        raise

    # Loop added nothing (the AI's "done" signal) -> serve the report unchanged, no debit.
    if not extra:
        posthog.capture(identity.distinct_id, "deepen_succeeded", {
            "reportId": session_id, "newChartCount": 0,
            "elapsedMs": int((time.perf_counter() - started) * 1000),
        })
        return JSONResponse(content=row["report_json"], status_code=200)

    from uuid import uuid4
    new_charts = [
        ChartWithCaption(chart_id=uuid4().hex, spec=spec, caption=spec.intent)
        for spec in extra
    ]

    # Re-narrate over the FULL enriched set so the summary + captions cover the new charts.
    # Covered by the same busy/error handling as deepen() above: a 529 overload during the
    # narrative pass must surface as a clean 503 BUSY (not a generic 500), fire deepen_failed,
    # and — critically — leave the report unsaved and the user uncharged.
    all_specs = [c.spec for c in existing_report.charts] + [c.spec for c in new_charts]
    all_charts = list(existing_report.charts) + new_charts
    try:
        narrative = gen.generate_narrative(all_specs)
    except RetryableBusy:
        posthog.capture(identity.distinct_id, "deepen_failed", {
            "reportId": session_id, "reason": "busy", "httpStatus": 503,
            "elapsedMs": int((time.perf_counter() - started) * 1000),
        })
        raise HTTPException(status_code=503, detail={
            "code": "BUSY",
            "message": "Claude is busy. Try again in 30s.",
        })
    except Exception as e:
        posthog.capture(identity.distinct_id, "deepen_failed", {
            "reportId": session_id, "reason": "internal",
            "errorClass": type(e).__name__,
            "elapsedMs": int((time.perf_counter() - started) * 1000),
        })
        raise
    captions = list(narrative.captions)
    if len(captions) < len(all_charts):
        captions = captions + [c.spec.intent for c in all_charts[len(captions):]]

    report_dict = row["report_json"]
    report_dict["summary"] = narrative.summary or report_dict.get("summary", "")
    if narrative.data_quality:
        report_dict["data_quality"] = narrative.data_quality
    if isinstance(report_dict.get("metadata"), dict):
        report_dict["metadata"]["deep"] = True

    # Apply refreshed captions to existing charts and append the new ones.
    for cwc, cap in zip(all_charts, captions):
        cwc.caption = cap
    report_dict["charts"] = [c.model_dump() for c in all_charts]

    sidebar_max = max(
        (e["order"] for e in report_dict["layout"] if e["position"] == "sidebar"),
        default=-1,
    )
    for i, c in enumerate(new_charts):
        report_dict["layout"].append(
            ChartLayoutEntry(chart_id=c.chart_id, position="sidebar", order=sidebar_max + 1 + i).model_dump()
        )

    db.update_report_json(session_id, report_dict)
    try:
        new_balance = db.spend_credits(identity.user_id, DEEP_ANALYSIS_COST, "deep_analysis", session_id)
        posthog.capture(identity.distinct_id, "credits_spent",
                        {"amount": DEEP_ANALYSIS_COST, "balance": new_balance, "reason": "deep_analysis"})
    except InsufficientCredits:
        logging.warning("Credit spend lost a race on deepen %s; serving free.", session_id)

    posthog.capture(identity.distinct_id, "deepen_succeeded", {
        "reportId": session_id,
        "newChartCount": len(new_charts),
        "elapsedMs": int((time.perf_counter() - started) * 1000),
    })

    return JSONResponse(content=report_dict, status_code=200)


@app.get("/report/{session_id}/export.pdf")
async def export_pdf(
    session_id: str,
    identity: Identity = Depends(get_identity),
    db: SupabaseDB = Depends(get_db),
    posthog: PostHogServer = Depends(get_posthog),
):
    started = time.perf_counter()
    if not db.get_report(session_id):
        raise HTTPException(status_code=404, detail="Report not found.")

    import pdf_export
    cold_start = pdf_export._browser is None
    posthog.capture(identity.distinct_id, "pdf_export_started", {
        "reportId": session_id, "coldStart": cold_start,
    })

    try:
        pdf_bytes = await pdf_export.render_report_pdf(session_id)
    except Exception as e:
        logging.exception("PDF export failed")
        posthog.capture(identity.distinct_id, "pdf_export_failed", {
            "reportId": session_id,
            "reason": "internal",
            "errorClass": type(e).__name__,
        })
        raise HTTPException(status_code=500, detail=f"PDF export failed: {e}")

    posthog.capture(identity.distinct_id, "pdf_export_succeeded", {
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


def _export_response(payload: bytes, session_id: str, ext: str, media_type: str) -> StreamingResponse:
    short = session_id[:8]
    return StreamingResponse(
        io.BytesIO(payload),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="chartsage-{short}.{ext}"'},
    )


@app.get("/report/{session_id}/export.pptx")
async def export_pptx(
    session_id: str,
    identity: Identity = Depends(get_identity),
    db: SupabaseDB = Depends(get_db),
    posthog: PostHogServer = Depends(get_posthog),
):
    row = db.get_report(session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Report not found.")
    report = Report.model_validate(row["report_json"])

    import pdf_export
    try:
        images = await pdf_export.render_chart_images(session_id)
        payload = report_export.build_pptx(report, images)
    except Exception as e:
        logging.exception("PPTX export failed")
        raise HTTPException(status_code=500, detail=f"PPTX export failed: {e}")

    posthog.capture(identity.distinct_id, "report_exported",
                    {"reportId": session_id, "format": "pptx"})
    return _export_response(
        payload, session_id, "pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )


@app.get("/report/{session_id}/export.xlsx")
async def export_xlsx(
    session_id: str,
    identity: Identity = Depends(get_identity),
    db: SupabaseDB = Depends(get_db),
    storage: SupabaseStorage = Depends(get_storage),
    posthog: PostHogServer = Depends(get_posthog),
):
    row = db.get_report(session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Report not found.")
    report = Report.model_validate(row["report_json"])

    csv_key = row.get("csv_storage_key")
    if not csv_key:
        raise HTTPException(status_code=404, detail={
            "code": "SOURCE_DATA_UNAVAILABLE",
            "message": "Source data for this report is no longer available.",
        })
    try:
        csv_bytes = storage.download_by_key(csv_key)
    except StorageError:
        raise HTTPException(status_code=404, detail={
            "code": "SOURCE_DATA_UNAVAILABLE",
            "message": "Source data for this report is no longer available.",
        })

    try:
        payload = report_export.build_xlsx(report, csv_bytes)
    except Exception as e:
        logging.exception("XLSX export failed")
        raise HTTPException(status_code=500, detail=f"XLSX export failed: {e}")

    posthog.capture(identity.distinct_id, "report_exported",
                    {"reportId": session_id, "format": "xlsx"})
    return _export_response(
        payload, session_id, "xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/report/{session_id}/export.zip")
async def export_zip(
    session_id: str,
    identity: Identity = Depends(get_identity),
    db: SupabaseDB = Depends(get_db),
    posthog: PostHogServer = Depends(get_posthog),
):
    row = db.get_report(session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Report not found.")
    report = Report.model_validate(row["report_json"])

    import pdf_export
    try:
        images = await pdf_export.render_chart_images(session_id)
        payload = report_export.build_png_zip(images, report)
    except Exception as e:
        logging.exception("PNG zip export failed")
        raise HTTPException(status_code=500, detail=f"PNG zip export failed: {e}")

    posthog.capture(identity.distinct_id, "report_exported",
                    {"reportId": session_id, "format": "zip"})
    return _export_response(payload, session_id, "zip", "application/zip")


@app.get("/report/{session_id}/export.md")
async def export_md(
    session_id: str,
    identity: Identity = Depends(get_identity),
    db: SupabaseDB = Depends(get_db),
    posthog: PostHogServer = Depends(get_posthog),
):
    row = db.get_report(session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Report not found.")
    report = Report.model_validate(row["report_json"])

    import pdf_export
    try:
        images = await pdf_export.render_chart_images(session_id)
        payload = report_export.build_markdown(report, images).encode("utf-8")
    except Exception as e:
        logging.exception("Markdown export failed")
        raise HTTPException(status_code=500, detail=f"Markdown export failed: {e}")

    posthog.capture(identity.distinct_id, "report_exported",
                    {"reportId": session_id, "format": "md"})
    return _export_response(payload, session_id, "md", "text/markdown")


@app.get("/report/{session_id}/export.html")
async def export_html(
    session_id: str,
    identity: Identity = Depends(get_identity),
    db: SupabaseDB = Depends(get_db),
    posthog: PostHogServer = Depends(get_posthog),
):
    row = db.get_report(session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Report not found.")
    report = Report.model_validate(row["report_json"])

    import pdf_export
    try:
        images = await pdf_export.render_chart_images(session_id)
        payload = report_export.build_html(report, images)
    except Exception as e:
        logging.exception("HTML export failed")
        raise HTTPException(status_code=500, detail=f"HTML export failed: {e}")

    posthog.capture(identity.distinct_id, "report_exported",
                    {"reportId": session_id, "format": "html"})
    return _export_response(payload, session_id, "html", "text/html")


@app.get("/billing/packages")
async def billing_packages():
    """Public catalogue of purchasable credit packs (single source of truth)."""
    return public_catalogue()


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
    if body.company and body.company.strip():     # honeypot -> silently accept, store nothing
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


@app.on_event("shutdown")
async def _shutdown_event():
    try:
        from pdf_export import shutdown as pdf_shutdown
        await pdf_shutdown()
    except Exception:
        pass


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
