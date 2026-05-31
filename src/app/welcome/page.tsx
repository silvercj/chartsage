'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { posthog } from '../lib/posthog';

const FLAG = 'chartsage_onboarded';

function safeNext(raw: string | null): string {
  if (raw && raw.startsWith('/') && !raw.startsWith('//')) return raw;
  return '/reports';
}

const UNLOCKS = [
  {
    title: '300 credits to start',
    body: 'Enough for about 3 reports — and more charts whenever you need them.',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" className="w-5 h-5">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6A2.25 2.25 0 013.75 8.25V6zM3.75 15.75A2.25 2.25 0 016 13.5h2.25a2.25 2.25 0 012.25 2.25V18a2.25 2.25 0 01-2.25 2.25H6A2.25 2.25 0 013.75 18v-2.25zM13.5 6a2.25 2.25 0 012.25-2.25H18A2.25 2.25 0 0120.25 6v2.25A2.25 2.25 0 0118 10.5h-2.25A2.25 2.25 0 0113.5 8.25V6zM13.5 15.75a2.25 2.25 0 012.25-2.25H18a2.25 2.25 0 012.25 2.25V18A2.25 2.25 0 0118 20.25h-2.25a2.25 2.25 0 01-2.25-2.25v-2.25z" />
      </svg>
    ),
  },
  {
    title: 'Generate more charts',
    body: 'Spend credits to add more angles to any report, on demand.',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" className="w-5 h-5">
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456z" />
      </svg>
    ),
  },
  {
    title: 'Saved & revisitable',
    body: 'Every report is kept in My Reports for later.',
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" className="w-5 h-5">
        <path strokeLinecap="round" strokeLinejoin="round" d="M17.593 3.322c1.1.128 1.907 1.077 1.907 2.185V21L12 17.25 4.5 21V5.507c0-1.108.806-2.057 1.907-2.185a48.507 48.507 0 0111.186 0z" />
      </svg>
    ),
  },
];

export default function WelcomePage() {
  const router = useRouter();
  const [ready, setReady] = useState(false);
  const [next, setNext] = useState('/');

  useEffect(() => {
    const dest = safeNext(new URLSearchParams(window.location.search).get('next'));
    setNext(dest);
    if (localStorage.getItem(FLAG)) {
      router.replace(dest);
      return;
    }
    posthog.capture?.('onboarding_viewed', {});
    setReady(true);
  }, [router]);

  function finish() {
    localStorage.setItem(FLAG, '1');
    posthog.capture?.('onboarding_completed', {});
    router.replace(next);
  }

  if (!ready) return null;

  return (
    <div className="min-h-screen bg-gradient-to-b from-stone-50 to-stone-100 flex items-center justify-center px-4 py-12">
      <div className="w-full max-w-lg bg-white rounded-3xl ring-1 ring-stone-200/70 shadow-sm overflow-hidden">
        <div className="h-1.5 bg-gradient-to-r from-teal-400 to-teal-600" />
        <div className="p-8 sm:p-10">
          <div className="flex items-center gap-2.5 mb-7">
            <span className="inline-flex items-center justify-center w-8 h-8 rounded-lg bg-stone-900 text-white text-sm font-semibold">C</span>
            <p className="text-xs uppercase tracking-widest text-stone-400">Welcome to ChartSage</p>
          </div>

          <h1 className="text-3xl font-semibold tracking-tight text-stone-900 mb-2">You're in.</h1>
          <p className="text-stone-500 mb-8">Here's what your account unlocks.</p>

          <ul className="space-y-5 mb-9">
            {UNLOCKS.map((u) => (
              <li key={u.title} className="flex gap-4">
                <span className="flex-shrink-0 inline-flex items-center justify-center w-10 h-10 rounded-xl bg-teal-50 text-teal-600 ring-1 ring-teal-100">
                  {u.icon}
                </span>
                <div className="pt-0.5">
                  <p className="font-medium text-stone-900">{u.title}</p>
                  <p className="text-sm text-stone-500 leading-relaxed">{u.body}</p>
                </div>
              </li>
            ))}
          </ul>

          <button
            type="button"
            onClick={finish}
            className="w-full sm:w-auto px-6 py-2.5 bg-stone-900 text-white text-sm font-medium rounded-lg hover:bg-stone-800 transition-colors"
          >
            Get started →
          </button>
        </div>
      </div>
    </div>
  );
}
