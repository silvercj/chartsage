"""Supabase Postgres client.

The interface mirrors the FakeDB helper used in tests so we can swap
between real and fake implementations through dependency injection.
"""
import os
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from supabase import create_client, Client

from credits import InsufficientCredits


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

    def set_report_visibility(self, report_id: str, is_public: bool,
                              og_image_key: str | None = None,
                              published_at: str | None = None) -> bool:
        patch: dict = {"is_public": is_public}
        if og_image_key is not None:
            patch["og_image_key"] = og_image_key
        if published_at is not None:
            patch["published_at"] = published_at
        res = self.client.table("reports").update(patch).eq("id", report_id).execute()
        return len(res.data) > 0

    def list_public_reports(self, limit: int = 5000) -> list[dict]:
        res = (self.client.table("reports").select("id, updated_at")
               .eq("is_public", True).order("updated_at", desc=True).limit(limit).execute())
        return res.data or []

    def count_anon_reports(self, anon_id: UUID) -> int:
        res = (self.client.table("reports")
               .select("id", count="exact")
               .eq("anon_id", str(anon_id))
               .is_("user_id", "null")
               .execute())
        return res.count or 0

    def claim_anon_reports(self, anon_id: UUID, user_id: UUID) -> int:
        """Reassign an anon's unclaimed reports to a user. Idempotent."""
        res = (self.client.table("reports")
               .update({"user_id": str(user_id), "anon_id": None, "updated_at": "now()"})
               .eq("anon_id", str(anon_id))
               .is_("user_id", "null")
               .execute())
        return len(res.data or [])

    def list_user_reports(self, user_id: UUID) -> list[dict]:
        """Compact summaries for the My Reports page, newest first."""
        res = (self.client.table("reports")
               .select("id, title, report_json, created_at")
               .eq("user_id", str(user_id))
               .order("created_at", desc=True)
               .execute())
        return [_summarize_report_row(r) for r in (res.data or [])]

    # --- credits (SP3) ---
    def ensure_profile(self, user_id, grant_amount: int) -> int:
        res = self.client.rpc("ensure_profile",
                              {"p_user": str(user_id), "p_grant": grant_amount}).execute()
        return int(res.data)

    def get_balance(self, user_id) -> int:
        res = (self.client.table("profiles").select("credits_balance")
               .eq("user_id", str(user_id)).limit(1).execute())
        return res.data[0]["credits_balance"] if res.data else 0

    def profile_exists(self, user_id) -> bool:
        res = (self.client.table("profiles").select("user_id")
               .eq("user_id", str(user_id)).limit(1).execute())
        return bool(res.data)

    def grant_credits(self, user_id, amount: int, reason: str, ref=None) -> int:
        res = self.client.rpc("grant_credits",
                              {"p_user": str(user_id), "p_amount": amount,
                               "p_reason": reason, "p_ref": ref}).execute()
        return int(res.data)

    def process_stripe_purchase(self, event_id: str, user_id, credits: int, ref: str) -> dict:
        """Atomic + idempotent (server-side fn). Returns {'granted': bool, 'balance': int}."""
        res = self.client.rpc("process_stripe_purchase", {
            "p_event": event_id, "p_user": str(user_id),
            "p_credits": credits, "p_ref": ref,
        }).execute()
        return res.data

    def spend_credits(self, user_id, amount: int, reason: str, ref=None) -> int:
        try:
            res = self.client.rpc("spend_credits",
                                  {"p_user": str(user_id), "p_amount": amount,
                                   "p_reason": reason, "p_ref": ref}).execute()
        except Exception as e:
            if "INSUFFICIENT_CREDITS" in str(e):
                raise InsufficientCredits()
            raise
        return int(res.data)

    def list_transactions(self, user_id, limit: int = 50) -> list[dict]:
        res = (self.client.table("credit_transactions")
               .select("delta, reason, ref, created_at")
               .eq("user_id", str(user_id))
               .order("created_at", desc=True).limit(limit).execute())
        return res.data or []

    def record_upgrade_intent(self, user_id, email) -> None:
        (self.client.table("upgrade_intent")
         .upsert({"user_id": str(user_id), "email": email}, on_conflict="user_id")
         .execute())

    def save_support_message(self, email, message, user_id, anon_id) -> None:
        self.client.table("support_messages").insert({
            "email": email,
            "message": message,
            "user_id": str(user_id) if user_id else None,
            "anon_id": str(anon_id) if anon_id else None,
        }).execute()

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


def _summarize_report_row(row: dict) -> dict:
    charts = (row.get("report_json") or {}).get("charts", [])
    kinds: list[str] = []
    for c in charts:
        kind = (c.get("spec") or {}).get("kind")
        if kind and kind not in kinds:
            kinds.append(kind)
    return {
        "id": row["id"],
        "title": row.get("title") or "Untitled report",
        "chartCount": len(charts),
        "kinds": kinds,
        "createdAt": row.get("created_at"),
    }
