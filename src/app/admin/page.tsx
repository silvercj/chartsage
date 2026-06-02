'use client';

import { useEffect, useState } from 'react';
import { adminFetch, getAdminToken, setAdminToken, clearAdminToken } from '../lib/adminApi';

interface Account { user_id: string; email: string; credits_balance: number; created_at?: string; }
interface Txn { delta: number; reason: string; ref?: string | null; created_at?: string; }
interface Detail extends Account { transactions: Txn[]; }

export default function AdminPage() {
  const [hasToken, setHasToken] = useState(false);
  const [tokenInput, setTokenInput] = useState('');
  const [q, setQ] = useState('');
  const [results, setResults] = useState<Account[]>([]);
  const [selected, setSelected] = useState<Detail | null>(null);
  const [amount, setAmount] = useState('');
  const [reason, setReason] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { setHasToken(!!getAdminToken()); }, []);

  const saveToken = () => { setAdminToken(tokenInput); setHasToken(!!tokenInput.trim()); setError(null); };
  const signOut = () => { clearAdminToken(); setHasToken(false); setResults([]); setSelected(null); };

  const search = async () => {
    setError(null); setBusy(true);
    try {
      const res = await adminFetch(`/admin/accounts?q=${encodeURIComponent(q)}&limit=50`);
      if (res.status === 403) { setError('Invalid or missing admin token.'); setHasToken(false); return; }
      if (!res.ok) { setError('Search failed.'); return; }
      setResults(await res.json());
    } finally { setBusy(false); }
  };

  const open = async (id: string) => {
    setError(null); setSelected(null);
    const res = await adminFetch(`/admin/accounts/${id}`);
    if (res.status === 403) { setError('Invalid or missing admin token.'); setHasToken(false); return; }
    if (!res.ok) { setError('Could not load account.'); return; }
    setSelected(await res.json()); setAmount(''); setReason('');
  };

  const grant = async () => {
    if (!selected) return;
    const n = parseInt(amount, 10);
    if (!Number.isFinite(n) || n < 1) { setError('Enter a positive amount.'); return; }
    setBusy(true); setError(null);
    try {
      const res = await adminFetch(`/admin/accounts/${selected.user_id}/grant`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ amount: n, reason: reason || undefined }),
      });
      if (res.status === 403) { setError('Invalid or missing admin token.'); setHasToken(false); return; }
      if (!res.ok) {
        const b = await res.json().catch(() => null);
        setError(b?.detail?.message || 'Grant failed.'); return;
      }
      const { credits_balance } = await res.json();
      setToast(`Granted ${n} credits → new balance ${credits_balance}`);
      setTimeout(() => setToast(null), 4000);
      await open(selected.user_id);
      await search();
    } finally { setBusy(false); }
  };

  if (!hasToken) {
    return (
      <main className="min-h-screen bg-canvas text-ink flex items-center justify-center p-6">
        <div className="card rounded-2xl p-6 max-w-sm w-full">
          <h1 className="font-display text-xl font-semibold mb-2">Admin access</h1>
          <p className="text-sm text-ink-2 mb-4">Paste the admin token to continue.</p>
          <input type="password" value={tokenInput} onChange={(e) => setTokenInput(e.target.value)}
                 placeholder="Admin token"
                 className="w-full rounded-lg bg-surface border border-line px-3 py-2 text-sm mb-3" />
          <button onClick={saveToken} className="btn btn-primary w-full">Continue</button>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-canvas text-ink p-6 max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="font-display text-2xl font-semibold">Admin · Accounts</h1>
        <button onClick={signOut} className="btn btn-ghost text-sm">Clear token</button>
      </div>
      <div className="flex gap-2 mb-4">
        <input value={q} onChange={(e) => setQ(e.target.value)}
               onKeyDown={(e) => e.key === 'Enter' && search()}
               placeholder="Search by email…"
               className="flex-1 rounded-lg bg-surface border border-line px-3 py-2 text-sm" />
        <button onClick={search} disabled={busy} className="btn btn-primary">Search</button>
      </div>
      {error && <p className="text-sm text-ember mb-3">{error}</p>}
      {toast && <p className="text-sm text-accent mb-3">{toast}</p>}
      <div className="grid gap-2 mb-6">
        {results.map((a) => (
          <button key={a.user_id} onClick={() => open(a.user_id)}
                  className="card rounded-xl px-4 py-3 text-left flex items-center justify-between hover:border-line-2">
            <span className="text-sm">{a.email}</span>
            <span className="font-mono text-sm text-ink-2">{a.credits_balance} cr</span>
          </button>
        ))}
        {!results.length && <p className="text-sm text-ink-3">No accounts loaded — search above.</p>}
      </div>
      {selected && (
        <div className="card rounded-2xl p-5">
          <div className="flex items-center justify-between mb-1">
            <h2 className="font-display text-lg font-semibold">{selected.email}</h2>
            <span className="font-mono text-sm text-ink-2">{selected.credits_balance} cr</span>
          </div>
          <p className="font-mono text-xs text-ink-3 mb-4">{selected.user_id}</p>
          <div className="flex gap-2 items-end mb-5">
            <label className="flex-1">
              <span className="block text-xs text-ink-3 mb-1">Amount</span>
              <input type="number" value={amount} onChange={(e) => setAmount(e.target.value)}
                     placeholder="1000"
                     className="w-full rounded-lg bg-surface border border-line px-3 py-2 text-sm" />
            </label>
            <label className="flex-1">
              <span className="block text-xs text-ink-3 mb-1">Reason (optional)</span>
              <input value={reason} onChange={(e) => setReason(e.target.value)}
                     placeholder="admin_grant"
                     className="w-full rounded-lg bg-surface border border-line px-3 py-2 text-sm" />
            </label>
            <button onClick={grant} disabled={busy} className="btn btn-primary">Grant</button>
          </div>
          <h3 className="text-xs uppercase tracking-wide text-ink-3 mb-2">Recent transactions</h3>
          <div className="grid gap-1">
            {selected.transactions.map((t, i) => (
              <div key={i} className="flex items-center justify-between text-sm border-b border-line py-1">
                <span className="text-ink-2">{t.reason}{t.ref ? ` · ${t.ref}` : ''}</span>
                <span className={`font-mono ${t.delta >= 0 ? 'text-accent' : 'text-ember'}`}>
                  {t.delta >= 0 ? '+' : ''}{t.delta}
                </span>
              </div>
            ))}
            {!selected.transactions.length && <p className="text-sm text-ink-3">No transactions.</p>}
          </div>
        </div>
      )}
    </main>
  );
}
