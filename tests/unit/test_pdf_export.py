"""Opt-in: requires Playwright + Chromium installed and a running frontend.

Run with: RUN_PDF_TESTS=true pytest tests/unit/test_pdf_export.py -v
"""
import asyncio
import os
import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_PDF_TESTS") != "true",
    reason="Opt-in: set RUN_PDF_TESTS=true",
)


def test_pdf_export_returns_pdf_bytes():
    """Smoke: render a stub report and assert we get valid PDF bytes."""
    # This test requires a real frontend at http://localhost:3000 to be running.
    # It's primarily here for CI; in practice, manually drive the UI.
    from pdf_export import render_report_pdf

    # Use a known-good session_id from a manual generate-report run.
    session_id = os.getenv("PDF_TEST_SESSION_ID")
    if not session_id:
        pytest.skip("Set PDF_TEST_SESSION_ID to a valid session_id")

    pdf_bytes = asyncio.run(render_report_pdf(session_id))
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 10_000
