'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { apiFetch } from '../lib/api';
import { getSupabaseBrowser } from '../lib/supabase';
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
  const { balance } = useCredits();
  const [txns, setTxns] = useState<Txn[] | null>(null);
  const [showNotify, setShowNotify] = useState(false);

  useEffect(() => {
    (async () => {
      const { data } = await getSupabaseBrowser().auth.getSession();
      if (!data.session) { router.replace('/login?next=/credits'); return; }
      const res = await apiFetch('/credits/history');
      if (res.status === 401) { router.replace('/login?next=/credits'); return; }
      if (res.ok) setTxns(await res.json());
      else setTxns([]);
    })();
  }, [router]);

  return (
    <div className="min-h-screen bg-stone-50">
      <div className="max-w-2xl mx-auto px-4 sm:px-6 py-12">
        <h1 className="text-3xl font-semibold tracking-tight text-stone-900 mb-1">Credits</h1>
        <p className="text-stone-500 mb-8">What you have, and where it went.</p>

        <div className="p-6 bg-white border border-stone-200 rounded-2xl mb-6">
          <p className="text-xs uppercase tracking-widest text-stone-400 mb-1">Balance</p>
          <p className="text-4xl font-semibold text-stone-900">{balance ?? '—'} <span className="text-base font-normal text-stone-400">credits</span></p>
          <div className="mt-4 text-sm text-stone-500 flex gap-6">
            <span>New report · <strong className="text-stone-700">{REPORT_COST}</strong></span>
            <span>Generate 5 more · <strong className="text-stone-700">{GENERATE_MORE_COST}</strong></span>
          </div>
          <button
            type="button"
            onClick={() => setShowNotify(true)}
            className="mt-5 px-4 py-2 text-sm font-medium text-stone-700 bg-white ring-1 ring-stone-300 rounded-lg hover:bg-stone-50"
          >
            Need more? Get notified about top-ups
          </button>
        </div>

        <h2 className="text-sm font-medium text-stone-500 mb-3">History</h2>
        {!txns ? (
          <p className="text-stone-400 text-sm">Loading…</p>
        ) : txns.length === 0 ? (
          <p className="text-stone-400 text-sm">No activity yet.</p>
        ) : (
          <ul className="divide-y divide-stone-100 bg-white border border-stone-200 rounded-2xl overflow-hidden">
            {txns.map((t, i) => (
              <li key={i} className="px-5 py-3 flex items-center justify-between text-sm">
                <span className="text-stone-700">{LABELS[t.reason] ?? t.reason}</span>
                <span className={t.delta >= 0 ? 'text-teal-600 font-medium' : 'text-stone-500'}>
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
