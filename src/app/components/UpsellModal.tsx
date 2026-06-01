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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm px-4" onClick={onClose}>
      <div className="card shadow-card-lg rounded-2xl p-6 w-full max-w-md" onClick={(e) => e.stopPropagation()}>
        <h2 className="font-display text-xl text-ink mb-1">Create a free account</h2>
        <p className="text-sm text-ink-2 mb-5">
          Sign in to generate more charts — your current report comes with you.
        </p>

        <button
          type="button"
          onClick={google}
          className="btn btn-ghost w-full"
        >
          Continue with Google
        </button>

        <div className="my-4 flex items-center gap-3 text-xs text-ink-3">
          <div className="flex-1 h-px bg-line" /> or <div className="flex-1 h-px bg-line" />
        </div>

        {sent ? (
          <p className="text-sm text-ink-2">Check your inbox for a magic link.</p>
        ) : (
          <form onSubmit={magic} className="space-y-3">
            <input
              type="email"
              required
              aria-label="Email address"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full bg-surface-2 border border-line-2 rounded-lg px-4 py-3 text-ink placeholder:text-ink-3 focus:border-accent outline-none"
            />
            <button type="submit" className="btn btn-primary w-full">
              Email me a magic link
            </button>
            {error && <p className="text-sm text-ember">{error}</p>}
          </form>
        )}

        <button type="button" onClick={onClose} className="mt-4 w-full text-sm text-ink-2 hover:text-ink">
          Maybe later
        </button>
      </div>
    </div>
  );
}
