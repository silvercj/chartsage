'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { apiFetch } from '../lib/api';
import { useCredits } from '../lib/useCredits';
import { REPORT_COST, GENERATE_MORE_COST } from '../lib/credits';
import OutOfCreditsModal from '../components/OutOfCreditsModal';

interface Txn { delta: number; reason: string; ref: string | null; created_at: string | number | null; }

const LABELS: Record<string, string> = {
  signup_grant: 'Welcome credits',
  report: 'New report',
  generate_more: 'Generate more charts',
  adjustment: 'Adjustment',
};

export default function CreditsPage() {
  const router = useRouter();
  const { balance, session, authLoading } = useCredits();
  const [txns, setTxns] = useState<Txn[] | null>(null);
  const [showNotify, setShowNotify] = useState(false);

  useEffect(() => {
    if (authLoading) return;                 // wait until auth state is known
    if (!session) { router.replace('/login?next=/credits'); return; }
    (async () => {
      try {
        const res = await apiFetch('/credits/history');
        if (res.status === 401) { router.replace('/login?next=/credits'); return; }
        setTxns(res.ok ? await res.json() : []);
      } catch {
        setTxns([]);   // never hang on "Loading…"
      }
    })();
  }, [authLoading, session, router]);

  return (
    <div className="min-h-screen bg-canvas">
      <div className="max-w-2xl mx-auto px-6 py-12">
        <h1 className="font-display text-3xl font-medium text-ink mb-1">Credits</h1>
        <p className="text-ink-2 mb-8">What you have, and where it went.</p>

        <div className="card shadow-card p-6 rounded-2xl mb-6">
          <p className="eyebrow mb-2">Balance</p>
          <p className="font-mono text-4xl font-semibold text-ink">{balance ?? '—'} <span className="text-base font-normal text-ink-3">credits</span></p>
          <div className="mt-4 text-sm text-ink-2 flex gap-6">
            <span>New report · <strong className="font-mono text-ink">{REPORT_COST}</strong></span>
            <span>Generate 5 more · <strong className="font-mono text-ink">{GENERATE_MORE_COST}</strong></span>
          </div>
          <button
            type="button"
            onClick={() => setShowNotify(true)}
            className="btn btn-ghost mt-5"
          >
            Need more? Get notified about top-ups
          </button>
        </div>

        <h2 className="eyebrow mb-3">History</h2>
        {!txns ? (
          <p className="text-ink-3 text-sm">Loading…</p>
        ) : txns.length === 0 ? (
          <p className="text-ink-3 text-sm">No activity yet.</p>
        ) : (
          <ul className="card divide-y divide-line overflow-hidden">
            {txns.map((t, i) => (
              <li key={i} className="px-5 py-3 flex items-center justify-between text-sm">
                <span className="text-ink">{LABELS[t.reason] ?? t.reason}</span>
                <span className={t.delta >= 0 ? 'font-mono text-accent font-medium' : 'font-mono text-ink-2'}>
                  {t.delta >= 0 ? `+${t.delta}` : t.delta}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
      <OutOfCreditsModal open={showNotify} onClose={() => setShowNotify(false)} />
    </div>
  );
}
