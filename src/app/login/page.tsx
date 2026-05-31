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
    <div className="min-h-screen bg-stone-50 flex items-center justify-center px-4">
      <div className="w-full max-w-sm">
        <p className="text-xs uppercase tracking-widest text-stone-400 mb-2">ChartSage</p>
        <h1 className="text-3xl font-semibold tracking-tight text-stone-900 mb-6">Sign in</h1>

        <button
          type="button"
          onClick={google}
          className="w-full px-4 py-2.5 bg-white ring-1 ring-stone-300 rounded-lg text-stone-800 font-medium hover:bg-stone-50 transition-colors"
        >
          Continue with Google
        </button>

        <div className="my-5 flex items-center gap-3 text-xs text-stone-400">
          <div className="flex-1 h-px bg-stone-200" /> or <div className="flex-1 h-px bg-stone-200" />
        </div>

        {sent ? (
          <p className="text-sm text-stone-600">
            Check your inbox — we sent a magic link to <strong>{email}</strong>.
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
              className="w-full px-3 py-2.5 rounded-lg ring-1 ring-stone-300 focus:ring-2 focus:ring-teal-500 outline-none text-stone-900"
            />
            <button
              type="submit"
              className="w-full px-4 py-2.5 bg-stone-900 text-white rounded-lg font-medium hover:bg-stone-800 transition-colors"
            >
              Email me a magic link
            </button>
            {error && <p className="text-sm text-red-600">{error}</p>}
          </form>
        )}

        <p className="mt-6 text-sm text-stone-400 text-center">
          <a href="/" className="hover:text-stone-700">← Back to home</a>
        </p>
      </div>
    </div>
  );
}
