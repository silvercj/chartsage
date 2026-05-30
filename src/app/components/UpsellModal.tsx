'use client';
import { useState } from 'react';
import { getSupabaseBrowser } from '../lib/supabase';
import { posthog } from '../lib/posthog';

export default function UpsellModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [email, setEmail] = useState('');
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!open) return null;

  const redirectTo = `${window.location.origin}/auth/callback?next=${encodeURIComponent(window.location.pathname)}`;

  async function google() {
    posthog.capture?.('generate_more_upsell_cta', { method: 'google' });
    await getSupabaseBrowser().auth.signInWithOAuth({ provider: 'google', options: { redirectTo } });
  }
  async function magic(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    posthog.capture?.('generate_more_upsell_cta', { method: 'magic_link' });
    const { error } = await getSupabaseBrowser().auth.signInWithOtp({ email, options: { emailRedirectTo: redirectTo } });
    if (error) setError(error.message);
    else setSent(true);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-stone-900/40 px-4" onClick={onClose}>
      <div className="w-full max-w-sm bg-white rounded-2xl p-6 shadow-xl" onClick={(e) => e.stopPropagation()}>
        <h2 className="text-lg font-semibold text-stone-900 mb-1">Create a free account</h2>
        <p className="text-sm text-stone-600 mb-5">
          Sign in to generate more charts — your current report comes with you.
        </p>

        <button
          type="button"
          onClick={google}
          className="w-full px-4 py-2.5 bg-white ring-1 ring-stone-300 rounded-lg text-stone-800 font-medium hover:bg-stone-50"
        >
          Continue with Google
        </button>

        <div className="my-4 flex items-center gap-3 text-xs text-stone-400">
          <div className="flex-1 h-px bg-stone-200" /> or <div className="flex-1 h-px bg-stone-200" />
        </div>

        {sent ? (
          <p className="text-sm text-stone-600">Check your inbox for a magic link.</p>
        ) : (
          <form onSubmit={magic} className="space-y-3">
            <input
              type="email"
              required
              aria-label="Email address"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full px-3 py-2.5 rounded-lg ring-1 ring-stone-300 focus:ring-2 focus:ring-teal-500 outline-none text-stone-900"
            />
            <button type="submit" className="w-full px-4 py-2.5 bg-stone-900 text-white rounded-lg font-medium hover:bg-stone-800">
              Email me a magic link
            </button>
            {error && <p className="text-sm text-red-600">{error}</p>}
          </form>
        )}

        <button type="button" onClick={onClose} className="mt-4 w-full text-sm text-stone-400 hover:text-stone-700">
          Maybe later
        </button>
      </div>
    </div>
  );
}
