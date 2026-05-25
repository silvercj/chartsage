'use client';

/**
 * The chartsage_anon cookie is httpOnly so JS can't read it directly.
 * The Next.js middleware also writes a parallel chartsage_anon_pub cookie
 * (NOT httpOnly) so the browser SDK can identify users in PostHog.
 *
 * Both contain the same UUID. The httpOnly one is sent automatically with
 * fetch credentials; we mirror it client-side for header injection too.
 */
const COOKIE_NAME = 'chartsage_anon_pub';

export function getAnonId(): string | null {
  if (typeof document === 'undefined') return null;
  const match = document.cookie.match(/(?:^|; )chartsage_anon_pub=([^;]+)/);
  return match ? decodeURIComponent(match[1]) : null;
}
