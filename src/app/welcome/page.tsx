'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { posthog } from '../lib/posthog';

const FLAG = 'chartsage_onboarded';

function safeNext(raw: string | null): string {
  if (raw && raw.startsWith('/') && !raw.startsWith('//')) return raw;
  return '/';
}

const UNLOCKS = [
  { title: 'Unlimited reports', body: 'Turn as many CSVs into dashboards as you like.' },
  { title: 'Generate more charts', body: 'Ask for extra angles on any report, on demand.' },
  { title: 'Saved & revisitable', body: 'Every report is kept in My Reports for later.' },
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
    <div className="min-h-screen bg-stone-50 flex items-center justify-center px-4">
      <div className="w-full max-w-lg">
        <p className="text-xs uppercase tracking-widest text-stone-400 mb-2">Welcome to ChartSage</p>
        <h1 className="text-3xl font-semibold tracking-tight text-stone-900 mb-6">
          You're in. Here's what your account unlocks.
        </h1>
        <ul className="space-y-4 mb-8">
          {UNLOCKS.map((u) => (
            <li key={u.title} className="flex gap-3">
              <span className="mt-1 h-2 w-2 rounded-full bg-teal-500 flex-shrink-0" />
              <div>
                <p className="font-medium text-stone-900">{u.title}</p>
                <p className="text-sm text-stone-600">{u.body}</p>
              </div>
            </li>
          ))}
        </ul>
        <button
          type="button"
          onClick={finish}
          className="px-5 py-2.5 bg-stone-900 text-white text-sm font-medium rounded-lg hover:bg-stone-800 transition-colors"
        >
          Get started →
        </button>
      </div>
    </div>
  );
}
