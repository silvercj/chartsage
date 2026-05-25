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
