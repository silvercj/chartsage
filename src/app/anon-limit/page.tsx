'use client';
import { useEffect } from 'react';
import { posthog } from '../lib/posthog';

export default function AnonLimitPage() {
  useEffect(() => {
    posthog.capture?.('anon_limit_page_viewed', { entryPoint: 'afterUpload' });
  }, []);

  return (
    <div className="min-h-screen bg-stone-50 flex items-center justify-center px-4">
      <div className="max-w-md text-center">
        <p className="text-xs uppercase tracking-widest text-stone-400 mb-2">Free tier</p>
        <h1 className="text-3xl font-semibold tracking-tight text-stone-900 mb-3">
          You've used your free report.
        </h1>
        <p className="text-stone-600 leading-relaxed mb-6">
          Create a free account to do more — 300 starter credits, generate
          additional charts, and save your reports. It only takes a few seconds.
        </p>
        <a
          href="/login"
          onClick={() => posthog.capture?.('signin_cta_clicked', { from: 'anonLimit' })}
          className="inline-block px-5 py-2.5 bg-stone-900 text-white text-sm font-medium rounded-lg hover:bg-stone-800 transition-colors"
        >
          Sign up free
        </a>
        <p className="mt-6 text-sm text-stone-400">
          <a href="/" className="hover:text-stone-700">← Back to home</a>
        </p>
      </div>
    </div>
  );
}
