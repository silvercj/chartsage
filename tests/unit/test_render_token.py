"""Short-lived, report-scoped render token (authorizes one server-side render
of a PRIVATE report via GET /report)."""
from render_token import make_render_token, verify_render_token


def test_roundtrip_valid():
    assert verify_render_token(make_render_token("abc"), "abc") is True


def test_wrong_report_id_rejected():
    assert verify_render_token(make_render_token("abc"), "xyz") is False


def test_tamper_rejected():
    assert verify_render_token(make_render_token("abc") + "x", "abc") is False


def test_expired_rejected():
    assert verify_render_token(make_render_token("abc", ttl=-1), "abc") is False


def test_garbage_rejected():
    assert verify_render_token("not.a.token", "abc") is False
    assert verify_render_token("", "abc") is False
    assert verify_render_token(None, "abc") is False
