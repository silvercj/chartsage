import pytest
from tests.helpers.fake_storage import FakeStorage, StorageError


def test_upload_returns_key():
    s = FakeStorage()
    key = s.upload_csv("abc", b"col1,col2\n1,2\n")
    assert key == "abc.csv"


def test_download_returns_uploaded_bytes():
    s = FakeStorage()
    s.upload_csv("abc", b"col1,col2\n1,2\n")
    assert s.download_csv("abc") == b"col1,col2\n1,2\n"


def test_download_missing_raises():
    s = FakeStorage()
    with pytest.raises(StorageError) as exc:
        s.download_csv("nope")
    assert "missing" in str(exc.value)


def test_delete_csv():
    s = FakeStorage()
    s.upload_csv("abc", b"x")
    s.delete_csv("abc")
    with pytest.raises(StorageError):
        s.download_csv("abc")


def test_delete_missing_is_noop():
    s = FakeStorage()
    s.delete_csv("never_existed")   # must not raise


def test_fail_next_upload_hook():
    s = FakeStorage()
    s.fail_next_upload()
    with pytest.raises(StorageError):
        s.upload_csv("abc", b"x")
    # subsequent uploads work again
    s.upload_csv("def", b"y")
    assert s.download_csv("def") == b"y"
