#!/usr/bin/env python3
"""Render an UNPUBLISHED ChartSage report for QA, by injecting the owner's anon-id cookie
(chartsage_anon_pub) so the page loads it as the owner — letting us eyeball a report we
generated *before* it's ever published. Shoots a chrome-stripped hero card + a full-page
review screenshot.

    qa_render.py <report-url> [hero-out.png] [full-out.png]

The owner anon id comes from CHARTSAGE_QA_ANON_ID / ~/.chartsage/qa-anon-id.
Outputs default to /tmp/qa_hero.png and /tmp/qa_full.png.
Needs the project venv (Playwright + chromium): ~/.venvs/chartsage/bin/python.
"""
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

APP = os.environ.get("CHARTSAGE_APP_URL", "https://chartsage.app").rstrip("/")

# Hide the card chrome (drag handle, index, kind badge, toggles, caption) — title + chart only.
STRIP_CHROME = """() => {
  const card = document.querySelector('section.card');
  if (!card) return;
  card.querySelectorAll('button, span').forEach(e => (e.style.display = 'none'));
}"""


def qa_anon_id() -> str:
    v = os.environ.get("CHARTSAGE_ANON_ID") or os.environ.get("CHARTSAGE_QA_ANON_ID")
    if v:
        return v.strip()
    f = Path.home() / ".chartsage" / "qa-anon-id"
    return f.read_text().strip() if f.is_file() else ""


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: qa_render.py <report-url> [hero-out.png] [full-out.png]")
    url = sys.argv[1]
    hero_out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/qa_hero.png"
    full_out = sys.argv[3] if len(sys.argv) > 3 else "/tmp/qa_full.png"
    anon = qa_anon_id()  # owner id from CHARTSAGE_QA_ANON_ID / ~/.chartsage/qa-anon-id
    if not anon:
        print("! no anon id (pass one or set ~/.chartsage/qa-anon-id); an unpublished report "
              "won't load as owner.", file=sys.stderr)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 1600}, device_scale_factor=1)
        if anon:
            ctx.add_cookies([
                {"name": "chartsage_anon_pub", "value": anon, "url": APP},
                {"name": "chartsage_anon", "value": anon, "url": APP},
            ])
        page = ctx.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        try:
            page.wait_for_selector("canvas", timeout=30000)  # ECharts renders to canvas
        except Exception:
            print("! no canvas appeared — did the report load as owner?", file=sys.stderr)
        page.wait_for_timeout(4200)  # let it finish drawing
        cards = page.query_selector_all("section.card")
        print("cards found:", len(cards))
        page.screenshot(path=full_out, full_page=True)
        page.evaluate(STRIP_CHROME)
        page.wait_for_timeout(400)
        if cards:
            cards[0].screenshot(path=hero_out)
        browser.close()
    print("saved", hero_out, "+", full_out)


if __name__ == "__main__":
    main()
