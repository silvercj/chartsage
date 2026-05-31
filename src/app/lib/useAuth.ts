'use client';
import { useEffect, useState } from 'react';
import { getSupabaseBrowser } from './supabase';
import { posthog } from './posthog';

/** Current signed-in user's email, or null when signed out. Reacts to auth changes. */
export function useAuthEmail(): string | null {
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
  return email;
}

/** Sign out and return to the home page.
 *  Uses scope:'local' (clears the local session without a server revoke that can
 *  hang on a stale session) and redirects regardless, so the button always works. */
export async function signOut(): Promise<void> {
  posthog.capture?.('signed_out', {});
  try {
    await getSupabaseBrowser().auth.signOut({ scope: 'local' });
  } catch {
    /* ignore — redirect anyway */
  }
  window.location.href = '/';
}
