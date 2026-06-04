'use client';

// X (Twitter) advertising pixel (UWT). Loaded app-wide via <XPixelInit>, EXCEPT on the
// chrome-less /embed, /print and /og routes. Per an explicit product decision it loads
// without a cookie-consent gate (matching how the analytics tool currently loads); the
// privacy policy discloses advertising cookies. Conversion events fire on funnel steps.

const PIXEL_ID = 'rcru2';
let initialized = false;

export const X_EVENTS = {
  purchase: 'tw-rcru2-rcruj',
  signup: 'tw-rcru2-rcruk',
} as const;

export function initXPixel(): void {
  if (typeof window === 'undefined') return;
  const w = window as any;
  if (initialized || w.twq) {
    initialized = true;
    return;
  }
  const twq: any = function (...args: any[]) {
    if (twq.exe) {
      twq.exe.apply(twq, args);
    } else {
      twq.queue.push(args);
    }
  };
  twq.version = '1.1';
  twq.queue = [];
  w.twq = twq;
  const u = document.createElement('script');
  u.async = true;
  u.src = 'https://static.ads-twitter.com/uwt.js';
  const first = document.getElementsByTagName('script')[0];
  first?.parentNode?.insertBefore(u, first);
  w.twq('config', PIXEL_ID);
  initialized = true;
}

export function xtwqEvent(eventId: string, params?: Record<string, unknown>): void {
  if (typeof window === 'undefined') return;
  (window as any).twq?.('event', eventId, params ?? {});
}
