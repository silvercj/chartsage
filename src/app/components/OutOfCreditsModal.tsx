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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-stone-900/40 px-4" onClick={onClose}>
      <div className="w-full max-w-sm bg-white rounded-2xl p-6 shadow-xl" onClick={(e) => e.stopPropagation()}>
        <h2 className="text-lg font-semibold text-stone-900 mb-1">You're out of credits</h2>
        {done ? (
          <p className="text-sm text-stone-600 mb-5">
            Thanks — we'll email you the moment paid top-ups launch.
          </p>
        ) : (
          <>
            <p className="text-sm text-stone-600 mb-5">
              Paid top-ups are coming soon. Want us to let you know when you can buy more credits?
            </p>
            <button
              type="button"
              onClick={notifyMe}
              disabled={busy}
              className="w-full px-4 py-2.5 bg-stone-900 text-white rounded-lg font-medium hover:bg-stone-800 disabled:opacity-50"
            >
              {busy ? 'Saving…' : 'Notify me'}
            </button>
          </>
        )}
        <button type="button" onClick={onClose} className="mt-4 w-full text-sm text-stone-400 hover:text-stone-700">
          {done ? 'Close' : 'Maybe later'}
        </button>
      </div>
    </div>
  );
}
