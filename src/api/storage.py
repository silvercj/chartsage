"""Supabase Storage wrapper.

Bucket name is fixed at csv-inputs. Keys are {report_id}.csv.
"""
import os
from typing import Optional
from supabase import create_client, Client


BUCKET = "csv-inputs"
OG_BUCKET = "og-images"


class StorageError(Exception):
    pass


def _client() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
    return create_client(url, key)


class SupabaseStorage:
    def __init__(self, client: Optional[Client] = None):
        self.client = client or _client()

    def upload_csv(self, report_id: str, csv_bytes: bytes) -> str:
        key = f"{report_id}.csv"
        try:
            self.client.storage.from_(BUCKET).upload(
                path=key,
                file=csv_bytes,
                file_options={"content-type": "text/csv", "upsert": "true"},
            )
        except Exception as e:
            raise StorageError(f"upload failed: {e}") from e
        return key

    def upload_public_image(self, key: str, png_bytes: bytes) -> str:
        """Upload to the public og-images bucket; returns the storage key."""
        try:
            self.client.storage.from_(OG_BUCKET).upload(
                path=key, file=png_bytes,
                file_options={"content-type": "image/png", "upsert": "true"},
            )
        except Exception as e:
            raise StorageError(f"og upload failed: {e}") from e
        return key

    def download_by_key(self, key: str) -> bytes:
        """Download the exact stored object key. Prefer this over download_csv when
        a csv_storage_key is recorded: report ids are uuid4().hex at upload time but
        Postgres normalizes the reports.id column to dashed form, so rebuilding
        {report_id}.csv from a dashed url id would miss the hex-keyed object."""
        try:
            return self.client.storage.from_(BUCKET).download(key)
        except Exception as e:
            raise StorageError(f"download failed: {e}") from e

    def download_csv(self, report_id: str) -> bytes:
        return self.download_by_key(f"{report_id}.csv")

    def delete_csv(self, report_id: str) -> None:
        key = f"{report_id}.csv"
        try:
            self.client.storage.from_(BUCKET).remove([key])
        except Exception:
            pass   # delete is best-effort
