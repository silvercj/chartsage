'use client';
import { apiFetch } from './api';
import { posthog } from './posthog';

// Must match SAMPLE_REPORT_ID in src/api/main.py — the fixed id of the public showcase
// report. The frontend uses it to recognise when a visitor is viewing the sample.
export const SAMPLE_REPORT_ID = '5a3b1ec05a3b1ec05a3b1ec05a3b1ec0';

/**
 * Open the public showcase report. Asks the backend to (lazily) ensure it exists, then
 * navigates to it. The very first call after a deploy generates it (~30s); every call
 * after is instant. No account, no file, no spent credit — built for first-touch / mobile
 * visitors who don't have a spreadsheet on hand.
 *
 * Returns false if it couldn't be opened (caller may surface a fallback).
 */
export async function openSampleReport(
  navigate: (href: string) => void,
  location: string,
): Promise<boolean> {
  posthog.capture?.('sample_report_clicked', { location });
  try {
    const res = await apiFetch('/sample-report');
    const body = await res.json().catch(() => null);
    if (res.ok && body?.session_id) {
      navigate(`/report/${body.session_id}`);
      return true;
    }
  } catch {
    /* fall through */
  }
  return false;
}
