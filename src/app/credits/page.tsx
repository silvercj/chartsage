'use client';
import { Suspense, useCallback, useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { apiFetch } from '../lib/api';
import { useCredits } from '../lib/useCredits';
import { REPORT_COST, GENERATE_MORE_COST } from '../lib/credits';
import { posthog } from '../lib/posthog';

interface Txn { delta: number; reason: string; ref: string | null; created_at: string | number | null; }
interface Pack { id: string; label: string; credits: number; price_display: string; }

const LABELS: Record<string, string> = {
  signup_grant: 'Welcome credits',
  report: 'New report',
  generate_more: 'Generate more charts',
  add_chart: 'Added chart',
  deep_analysis: 'Deep analysis',
  stripe_purchase: 'Credit purchase',
  admin_grant: 'Adjustment',
  adjustment: 'Adjustment',
};

function CreditsInner() {
  const router = useRouter();
  const params = useSearchParams();
  const { balance, session, authLoading, refetch } = useCredits();
  const [txns, setTxns] = useState<Txn[] | null>(null);
  const [packs, setPacks] = useState<Pack[]>([]);
  const [buying, setBuying] = useState<string | null>(null);
  const [showSuccess, setShowSuccess] = useState(false);
  const [successDone, setSuccessDone] = useState(false);
  const purchase = params.get('purchase');

  const loadHistory = useCallback(async () => {
    try {
      const res = await apiFetch('/credits/history');
      if (res.status === 401) { router.replace('/login?next=/credits'); return; }
      setTxns(res.ok ? await res.json() : []);
    } catch { setTxns([]); }
  }, [router]);

  // Auth guard + history load
  useEffect(() => {
    if (authLoading) return;
    if (!session) { router.replace('/login?next=/credits'); return; }
    loadHistory();
  }, [authLoading, session, router, loadHistory]);

  // Purchasable packs
  useEffect(() => {
    (async () => {
      try {
        const res = await apiFetch('/billing/packages');
        if (res.ok) setPacks(await res.json());
      } catch { /* leave empty; the section just hides */ }
    })();
  }, []);

  // After returning from Stripe success the webhook credits asynchronously —
  // poll the balance a few times so the new total appears without a refresh,
  // then confirm and auto-dismiss the banner (and drop ?purchase from the URL so
  // a refresh won't re-show it). All timers are tracked + cleared on unmount.
  useEffect(() => {
    if (purchase !== 'success') return;
    setShowSuccess(true);
    let n = 0;
    const timers: ReturnType<typeof setTimeout>[] = [];
    const tick = () => {
      refetch();
      n += 1;
      if (n < 3) {
        timers.push(setTimeout(tick, 2000));
      } else {
        loadHistory();
        setSuccessDone(true);
        timers.push(setTimeout(() => { setShowSuccess(false); router.replace('/credits'); }, 4000));
      }
    };
    tick();
    return () => timers.forEach(clearTimeout);
  }, [purchase, refetch, loadHistory, router]);

  // Track checkout abandonment (returned from Stripe without paying).
  useEffect(() => {
    if (purchase === 'cancelled') posthog.capture?.('checkout_cancelled', {});
  }, [purchase]);

  async function buy(pkgId: string) {
    setBuying(pkgId);
    posthog.capture?.('buy_pack_clicked', { package_id: pkgId });
    try {
      const res = await apiFetch('/billing/checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ package_id: pkgId }),
      });
      if (!res.ok) { setBuying(null); return; }
      const { url } = await res.json();
      window.location.href = url;
    } catch { setBuying(null); }
  }

  return (
    <div className="min-h-screen bg-canvas">
      <div className="max-w-2xl mx-auto px-6 py-12">
        <h1 className="font-display text-3xl font-medium text-ink mb-1">Credits</h1>
        <p className="text-ink-2 mb-8">What you have, and where it went.</p>

        {showSuccess && (
          <div className="card shadow-card p-4 rounded-2xl mb-6 border border-accent/40">
            <p className="text-sm text-ink">
              {successDone ? 'Credits added ✓' : 'Payment received — adding your credits…'}
            </p>
          </div>
        )}
        {purchase === 'cancelled' && (
          <div className="card shadow-card p-4 rounded-2xl mb-6">
            <p className="text-sm text-ink-2">Checkout cancelled — you weren't charged.</p>
          </div>
        )}

        <div className="card shadow-card p-6 rounded-2xl mb-6">
          <p className="eyebrow mb-2">Balance</p>
          <p className="font-mono text-4xl font-semibold text-ink">
            {balance ?? '—'} <span className="text-base font-normal text-ink-3">credits</span>
          </p>
          <div className="mt-4 text-sm text-ink-2 flex gap-6">
            <span>New report · <strong className="font-mono text-ink">{REPORT_COST}</strong></span>
            <span>Generate 5 more · <strong className="font-mono text-ink">{GENERATE_MORE_COST}</strong></span>
          </div>
        </div>

        {packs.length > 0 && (
          <>
            <h2 className="eyebrow mb-3">Buy more</h2>
            <div className="grid sm:grid-cols-3 gap-4 mb-8">
              {packs.map((p) => (
                <div key={p.id} className="card shadow-card p-5 rounded-2xl flex flex-col">
                  {p.id === 'standard'
                    ? <span className="eyebrow text-accent mb-1">Best value</span>
                    : <span className="eyebrow text-ink-3 mb-1">&nbsp;</span>}
                  <p className="font-display text-lg text-ink">{p.label}</p>
                  <p className="font-mono text-2xl font-semibold text-ink mt-1">{p.price_display}</p>
                  <p className="text-sm text-ink-2 mt-1 mb-4">{p.credits.toLocaleString()} credits</p>
                  <button
                    type="button"
                    onClick={() => buy(p.id)}
                    disabled={buying !== null}
                    className="btn btn-primary mt-auto w-full"
                  >
                    {buying === p.id ? 'Redirecting…' : 'Buy'}
                  </button>
                </div>
              ))}
            </div>
          </>
        )}

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
    </div>
  );
}

export default function CreditsPage() {
  // useSearchParams requires a Suspense boundary during prerender (Next 14).
  return (
    <Suspense fallback={<div className="min-h-screen bg-canvas" />}>
      <CreditsInner />
    </Suspense>
  );
}
