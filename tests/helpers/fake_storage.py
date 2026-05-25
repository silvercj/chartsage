"""In-memory storage matching the storage.py interface."""


class StorageError(Exception):
    pass


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

    def download_csv(self, report_id: str) -> bytes:
        key = f"{report_id}.csv"
        if key not in self._objects:
            raise StorageError(f"missing object: {key}")
        return self._objects[key]

    def delete_csv(self, report_id: str) -> None:
        self._objects.pop(f"{report_id}.csv", None)

    def fail_next_upload(self) -> None:
        self._fail_next_upload = True
