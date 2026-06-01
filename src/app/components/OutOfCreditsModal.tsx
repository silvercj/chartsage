'use client';
import { useState } from 'react';
import { apiFetch } from '../lib/api';
import { getSupabaseBrowser } from '../lib/supabase';
import { posthog } from '../lib/posthog';

export default function OutOfCreditsModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [done, setDone] = useState(false);
  const [busy, setBusy] = useState(false);
  if (!open) return null;

  async function notifyMe() {
    setBusy(true);
    posthog.capture?.('upgrade_intent_clicked', {});
    try {
      const { data } = await getSupabaseBrowser().auth.getSession();
      await apiFetch('/upgrade-intent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: data.session?.user?.email ?? null }),
      });
      setDone(true);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm px-4" onClick={onClose}>
      <div className="card shadow-card-lg rounded-2xl p-6 w-full max-w-md" onClick={(e) => e.stopPropagation()}>
        <h2 className="font-display text-xl text-ember mb-1">You're out of credits</h2>
        {done ? (
          <p className="text-sm text-ink-2 mb-5">
            Thanks — we'll email you the moment paid top-ups launch.
          </p>
        ) : (
          <>
            <p className="text-sm text-ink-2 mb-5">
              Paid top-ups are coming soon. Want us to let you know when you can buy more credits?
            </p>
            <button
              type="button"
              onClick={notifyMe}
              disabled={busy}
              className="btn btn-primary w-full"
            >
              {busy ? 'Saving…' : 'Notify me'}
            </button>
          </>
        )}
        <button type="button" onClick={onClose} className="mt-4 w-full text-sm text-ink-2 hover:text-ink">
          {done ? 'Close' : 'Maybe later'}
        </button>
      </div>
    </div>
  );
}
