"""Short-lived, report-scoped HMAC token that authorizes ONE server-side render
(the Playwright print/og pages) to read a PRIVATE report via GET /report without
user auth. Safe even if it leaks: scoped to a single report id and expires in ~2
minutes. The secret is server-only and never sent to the browser by us.
"""
import hashlib
import hmac
import os
import time

_TTL_DEFAULT = 120


def _secret() -> bytes:
    # Reuse the app's service-role key (always set in prod) so the token is
    # unforgeable. An explicit RENDER_TOKEN_SECRET overrides if set. Never logged.
    s = os.environ.get("RENDER_TOKEN_SECRET") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or ""
    return s.encode()


def make_render_token(report_id: str, ttl: int = _TTL_DEFAULT) -> str:
    exp = str(int(time.time()) + ttl)
    sig = hmac.new(_secret(), f"{report_id}.{exp}".encode(), hashlib.sha256).hexdigest()
    return f"{exp}.{sig}"


def verify_render_token(token: str, report_id: str) -> bool:
    try:
        exp_s, sig = token.split(".", 1)
        if int(exp_s) < int(time.time()):
            return False
        expected = hmac.new(_secret(), f"{report_id}.{exp_s}".encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, expected)
    except (ValueError, AttributeError):
        return False
