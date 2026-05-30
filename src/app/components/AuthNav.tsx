'use client';
import { useEffect, useState } from 'react';
import { getSupabaseBrowser } from '../lib/supabase';
import { posthog } from '../lib/posthog';

export default function AuthNav() {
  const [email, setEmail] = useState<string | null>(null);

  useEffect(() => {
    const supabase = getSupabaseBrowser();
    supabase.auth.getSession().then(({ data }) =>
      setEmail(data.session?.user?.email ?? null),
    );
    const { data: sub } = supabase.auth.onAuthStateChange((_e, session) =>
      setEmail(session?.user?.email ?? null),
    );
    return () => sub.subscription.unsubscribe();
  }, []);

  async function signOut() {
    posthog.capture?.('signed_out', {});
    await getSupabaseBrowser().auth.signOut();
    window.location.href = '/';
  }

  return (
    <nav className="fixed top-3 right-4 z-50 flex items-center gap-3 text-sm">
      {email ? (
        <>
          <a
            href="/reports"
            className="px-3 py-1.5 rounded-lg bg-white/90 ring-1 ring-stone-200 text-stone-700 hover:bg-white"
          >
            My reports
          </a>
          <span className="hidden sm:inline text-stone-400 max-w-[160px] truncate">{email}</span>
          <button
            type="button"
            onClick={signOut}
            className="px-3 py-1.5 rounded-lg text-stone-500 hover:text-stone-900"
          >
            Sign out
          </button>
        </>
      ) : (
        <a
          href="/login"
          className="px-3 py-1.5 rounded-lg bg-stone-900 text-white hover:bg-stone-800"
        >
          Sign in
        </a>
      )}
    </nav>
  );
}
