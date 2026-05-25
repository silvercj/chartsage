'use client';
import posthog from 'posthog-js';
import { getAnonId } from './anon';

let initialized = false;

export function initPostHog(): void {
  if (initialized || typeof window === 'undefined') return;
  const key = process.env.NEXT_PUBLIC_POSTHOG_KEY;
  const host = process.env.NEXT_PUBLIC_POSTHOG_HOST || 'https://us.i.posthog.com';
  if (!key) return;

  posthog.init(key, {
    api_host: host,
    capture_pageview: true,
    autocapture: false,
    person_profiles: 'identified_only',
  });

  const anonId = getAnonId();
  if (anonId) {
    posthog.identify(anonId);
  }
  initialized = true;
}

export { posthog };
