"""In-memory db matching the db.py interface, for integration tests."""
from copy import deepcopy
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from credits import InsufficientCredits


class FakeDB:
    def __init__(self):
        self._rows: dict[str, dict] = {}   # report_id -> row dict
        self._seq = 0
        self._profiles: dict[str, int] = {}   # user_id -> balance
        self._users: list[dict] = []          # fake auth user directory
        self._txns: list[dict] = []           # ledger
        self._intent: dict[str, str | None] = {}
        self._anon_log: list[dict] = []
        self._support_messages: list[dict] = []

    def save_report(
        self,
        report_id: str,
        anon_id: Optional[UUID],
        user_id: Optional[UUID],
        report_json: dict,
        csv_storage_key: Optional[str],
        title: str,
    ) -> None:
        self._seq += 1
        self._rows[report_id] = {
            "id": report_id,
            "anon_id": str(anon_id) if anon_id else None,
            "user_id": str(user_id) if user_id else None,
            "report_json": deepcopy(report_json),
            "csv_storage_key": csv_storage_key,
            "title": title,
            "_seq": self._seq,
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

    def claim_anon_reports(self, anon_id: UUID, user_id: UUID) -> int:
        n = 0
        for r in self._rows.values():
            if r["anon_id"] == str(anon_id) and r["user_id"] is None:
                r["user_id"] = str(user_id)
                r["anon_id"] = None
                n += 1
        return n

    def list_user_reports(self, user_id: UUID) -> list[dict]:
        rows = [r for r in self._rows.values() if r["user_id"] == str(user_id)]
        rows.sort(key=lambda r: r.get("_seq", 0), reverse=True)
        out = []
        for r in rows:
            charts = (r["report_json"] or {}).get("charts", [])
            kinds: list[str] = []
            for c in charts:
                kind = (c.get("spec") or {}).get("kind")
                if kind and kind not in kinds:
                    kinds.append(kind)
            out.append({
                "id": r["id"],
                "title": r["title"] or "Untitled report",
                "chartCount": len(charts),
                "kinds": kinds,
                "createdAt": r.get("_seq"),
            })
        return out

    # --- credits (SP3) ---
    def ensure_profile(self, user_id, grant_amount: int) -> int:
        uid = str(user_id)
        self._profiles.setdefault(uid, 0)
        if not any(t["user_id"] == uid and t["reason"] == "signup_grant" for t in self._txns):
            self.grant_credits(uid, grant_amount, "signup_grant", None)
        return self._profiles[uid]

    def get_balance(self, user_id) -> int:
        return self._profiles.get(str(user_id), 0)

    def profile_exists(self, user_id) -> bool:
        return str(user_id) in self._profiles

    def grant_credits(self, user_id, amount: int, reason: str, ref=None) -> int:
        uid = str(user_id)
        self._profiles[uid] = self._profiles.get(uid, 0) + amount
        self._txns.append({"user_id": uid, "delta": amount, "reason": reason, "ref": ref})
        return self._profiles[uid]

    def spend_credits(self, user_id, amount: int, reason: str, ref=None) -> int:
        uid = str(user_id)
        if self._profiles.get(uid, 0) < amount:
            raise InsufficientCredits()
        self._profiles[uid] -= amount
        self._txns.append({"user_id": uid, "delta": -amount, "reason": reason, "ref": ref})
        return self._profiles[uid]

    def list_transactions(self, user_id, limit: int = 50) -> list[dict]:
        rows = [t for t in self._txns if t["user_id"] == str(user_id)]
        return list(reversed(rows))[:limit]

    def record_upgrade_intent(self, user_id, email) -> None:
        self._intent[str(user_id)] = email

    def save_support_message(self, email, message, user_id, anon_id) -> None:
        self._support_messages.append({
            "email": email, "message": message,
            "user_id": str(user_id) if user_id else None,
            "anon_id": str(anon_id) if anon_id else None,
        })

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

    # --- admin (fake user directory) ---
    def add_user(self, user_id, email, created_at: str = "2026-01-01T00:00:00Z") -> None:
        self._users.append({"id": str(user_id), "email": email, "created_at": created_at})

    def _find_user(self, user_id):
        return next((u for u in self._users if u["id"] == str(user_id)), None)

    def search_accounts(self, query: str, limit: int = 50) -> list[dict]:
        q = (query or "").strip().lower()
        out = []
        for u in self._users:
            if q and q not in (u["email"] or "").lower():
                continue
            out.append({
                "user_id": u["id"],
                "email": u["email"],
                "credits_balance": self.get_balance(u["id"]),
                "created_at": u["created_at"],
            })
            if len(out) >= limit:
                break
        return out

    def get_account_detail(self, user_id) -> dict | None:
        u = self._find_user(user_id)
        if u is None:
            return None
        return {
            "user_id": u["id"],
            "email": u["email"],
            "credits_balance": self.get_balance(u["id"]),
            "transactions": self.list_transactions(u["id"]),
        }
