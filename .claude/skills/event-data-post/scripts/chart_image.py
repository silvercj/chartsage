#!/usr/bin/env python3
"""Render a clean ChartSage hero-chart image for a social post.

Loads a published report, strips the card chrome (buttons, index, kind badge,
caption), and screenshots the first chart card at 2x. Run it with a Python env
that has Playwright + chromium (the project's venv does):

    ~/.venvs/chartsage/bin/python scripts/chart_image.py \
        https://chartsage.app/report/<id> ~/Downloads/<event>_chart.png

The first chart card is the report's hero. The report must be public/published
for the URL to render without auth.
"""
import os
import sys

from playwright.sync_api import sync_playwright

# JS run in the page: hide every button + span (drag handle, index, kind badge,
# toggle, hide-X) and the caption paragraph, leaving just the title + chart.
STRIP_CHROME = """() => {
  const card = document.querySelector('section.card');
  if (!card) return;
  card.querySelectorAll('button, span').forEach(e => (e.style.display = 'none'));
  const cap = card.querySelector('p');
  if (cap) cap.style.display = 'none';
}"""


def render(url: str, out_path: str) -> str:
    out_path = os.path.expanduser(out_path)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1320, "height": 1800}, device_scale_factor=2)
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_selector("canvas", timeout=30000)  # ECharts renders to canvas
        page.wait_for_timeout(3800)                       # let it finish drawing
        page.evaluate(STRIP_CHROME)
        page.wait_for_timeout(400)
        cards = page.query_selector_all("section.card")
        if not cards:
            browser.close()
            raise SystemExit("No chart card found — is the report public + rendered?")
        cards[0].screenshot(path=out_path)
        browser.close()
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: chart_image.py <report-url> <output-path>")
    print("saved", render(sys.argv[1], sys.argv[2]))
