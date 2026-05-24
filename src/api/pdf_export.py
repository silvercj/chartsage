"""Playwright-driven PDF export.

Owns a single long-lived Browser instance, started lazily on first use.
Each export opens a fresh Page, navigates to the print route, waits for the
data-charts-ready flag, captures as A4 PDF.
"""
import asyncio
import logging
import os
from typing import Optional


_FRONTEND_BASE = os.getenv("FRONTEND_BASE_URL", "http://localhost:3000")
_CHARTS_READY_TIMEOUT_MS = 10_000


_browser_lock = asyncio.Lock()
_browser = None
_playwright = None


async def _ensure_browser():
    """Lazily start Playwright + Chromium. Reused across exports."""
    global _browser, _playwright
    async with _browser_lock:
        if _browser is None:
            from playwright.async_api import async_playwright
            _playwright = await async_playwright().start()
            _browser = await _playwright.chromium.launch(headless=True)
            logging.info("[PDF] Started Chromium")
    return _browser


async def render_report_pdf(session_id: str) -> bytes:
    """Render the /report/{session_id}/print route as an A4 PDF and return bytes."""
    browser = await _ensure_browser()
    page = await browser.new_page(viewport={"width": 1240, "height": 1754})
    try:
        url = f"{_FRONTEND_BASE}/report/{session_id}/print"
        await page.goto(url, wait_until="networkidle", timeout=30_000)
        # Wait for the print page to signal "all charts mounted"
        try:
            await page.wait_for_selector(
                'body[data-charts-ready="true"]',
                timeout=_CHARTS_READY_TIMEOUT_MS,
            )
        except Exception:
            logging.warning("[PDF] charts-ready flag not seen in %dms — proceeding", _CHARTS_READY_TIMEOUT_MS)
        pdf_bytes = await page.pdf(
            format="A4",
            print_background=True,
            margin={"top": "50px", "bottom": "50px", "left": "40px", "right": "40px"},
        )
        return pdf_bytes
    finally:
        await page.close()


async def shutdown():
    """Close the Browser and Playwright instance. Used on app shutdown."""
    global _browser, _playwright
    if _browser is not None:
        await _browser.close()
        _browser = None
    if _playwright is not None:
        await _playwright.stop()
        _playwright = None
