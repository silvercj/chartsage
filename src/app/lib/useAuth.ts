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

/** Sign out and return to the home page. */
export async function signOut(): Promise<void> {
  posthog.capture?.('signed_out', {});
  await getSupabaseBrowser().auth.signOut();
  window.location.href = '/';
}
