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
