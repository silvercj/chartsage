'use client';
import { useEffect } from 'react';
import { getSupabaseBrowser } from '../lib/supabase';
import { initPostHog, posthog } from '../lib/posthog';

export default function SessionWatcher() {
  useEffect(() => {
    initPostHog(); // idempotent — guarantees posthog is ready before identify()
    const supabase = getSupabaseBrowser();
    const { data: sub } = supabase.auth.onAuthStateChange((event, session) => {
      if (session?.user) {
        posthog.identify?.(session.user.id, { email: session.user.email });
      }
      if (event === 'SIGNED_IN') {
        posthog.capture?.('logged_in', {
          method: session?.user?.app_metadata?.provider,
        });
      }
      if (event === 'SIGNED_OUT') {
        posthog.reset?.();
      }
    });
    return () => sub.subscription.unsubscribe();
  }, []);
  return null;
}
