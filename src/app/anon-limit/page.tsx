'use client';
import { useEffect } from 'react';
import { posthog } from '../lib/posthog';

export default function AnonLimitPage() {
  useEffect(() => {
    posthog.capture?.('anon_limit_page_viewed', { entryPoint: 'afterUpload' });
  }, []);

  return (
    <div className="min-h-screen bg-canvas flex items-center justify-center px-6">
      <div className="card shadow-card-lg rounded-2xl p-8 max-w-md w-full text-center">
        <p className="eyebrow mb-3">Free tier</p>
        <h1 className="font-display text-2xl font-medium text-ink mb-3">
          You've used your free report.
        </h1>
        <p className="text-ink-2 text-sm leading-relaxed mb-6">
          Create a free account to do more — 300 starter credits, generate
          additional charts, and save your reports. It only takes a few seconds.
        </p>
        <a
          href="/login"
          onClick={() => posthog.capture?.('signin_cta_clicked', { from: 'anonLimit' })}
          className="btn btn-primary w-full"
        >
          Sign up free
        </a>
        <p className="mt-6 text-sm text-ink-2">
          <a href="/" className="hover:text-ink transition-colors">← Back to home</a>
        </p>
      </div>
    </div>
  );
}
