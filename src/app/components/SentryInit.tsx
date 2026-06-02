'use client';

import { useEffect } from 'react';
import * as Sentry from '@sentry/nextjs';

let initialized = false;

export default function SentryInit() {
  useEffect(() => {
    const dsn = process.env.NEXT_PUBLIC_SENTRY_DSN;
    if (dsn && !initialized) {
      Sentry.init({ dsn, tracesSampleRate: 0 });
      initialized = true;
    }
  }, []);
  return null;
}
