'use client';
import { useState } from 'react';
import { getSupabaseBrowser } from '../lib/supabase';
import { posthog } from '../lib/posthog';

function callbackUrl(): string {
  const params = new URLSearchParams(window.location.search);
  const next = params.get('next') || '/reports';
  const safe = next.startsWith('/') && !next.startsWith('//') ? next : '/reports';
  return `${window.location.origin}/auth/callback?next=${encodeURIComponent(safe)}`;
}

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function google() {
    posthog.capture?.('login_method_selected', { method: 'google' });
    await getSupabaseBrowser().auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo: callbackUrl() },
    });
  }

  async function magicLink(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    posthog.capture?.('login_method_selected', { method: 'magic_link' });
    const { error } = await getSupabaseBrowser().auth.signInWithOtp({
      email,
      options: { emailRedirectTo: callbackUrl() },
    });
    if (error) setError(error.message);
    else setSent(true);
  }

  return (
    <div className="min-h-screen bg-canvas flex items-center justify-center px-6">
      <div className="card shadow-card-lg rounded-2xl p-8 max-w-md w-full">
        <p className="eyebrow mb-3">ChartSage</p>
        <h1 className="font-display text-2xl font-medium text-ink mb-1.5">Sign in or sign up</h1>
        <p className="text-ink-2 text-sm mb-6">New here? Either option below creates your account — 300 free credits to start.</p>

        <button
          type="button"
          onClick={google}
          className="btn btn-ghost w-full"
        >
          Continue with Google
        </button>

        <div className="my-5 flex items-center gap-3 font-mono text-xs text-ink-3">
          <div className="flex-1 h-px bg-line-2" /> or <div className="flex-1 h-px bg-line-2" />
        </div>

        {sent ? (
          <p className="text-ink-2 text-sm">
            Check your inbox — we sent a magic link to <strong className="text-ink">{email}</strong>.
          </p>
        ) : (
          <form onSubmit={magicLink} className="space-y-3">
            <input
              type="email"
              required
              aria-label="Email address"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full bg-surface-2 border border-line-2 rounded-lg px-4 py-3 text-ink placeholder:text-ink-3 focus:border-accent outline-none transition-colors"
            />
            <button
              type="submit"
              className="btn btn-primary w-full"
            >
              Email me a magic link
            </button>
            {error && <p className="text-sm text-ember">{error}</p>}
          </form>
        )}

        <p className="mt-6 text-sm text-ink-2 text-center">
          <a href="/" className="hover:text-ink transition-colors">← Back to home</a>
        </p>
      </div>
    </div>
  );
}
