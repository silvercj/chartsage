"""In-memory storage matching the storage.py interface.

Raises storage.StorageError (the real class) so that main.py's
`except StorageError` catches failures from both real and fake storage.
"""
from storage import StorageError  # noqa: F401 — re-exported for test imports


class FakeStorage:
    def __init__(self):
        self._objects: dict[str, bytes] = {}
        self._fail_next_upload = False

    def upload_csv(self, report_id: str, csv_bytes: bytes) -> str:
        if self._fail_next_upload:
            self._fail_next_upload = False
            raise StorageError("simulated upload failure")
        key = f"{report_id}.csv"
        self._objects[key] = csv_bytes
        return key

    def upload_public_image(self, key: str, png_bytes: bytes) -> str:
        if self._fail_next_upload:
            self._fail_next_upload = False
            raise StorageError("simulated upload failure")
        self._objects[key] = png_bytes
        return key

    def download_by_key(self, key: str) -> bytes:
        if key not in self._objects:
            raise StorageError(f"missing object: {key}")
        return self._objects[key]

    def download_csv(self, report_id: str) -> bytes:
        return self.download_by_key(f"{report_id}.csv")

    def delete_csv(self, report_id: str) -> None:
        self._objects.pop(f"{report_id}.csv", None)

    def fail_next_upload(self) -> None:
        self._fail_next_upload = True
