'use client';
import { useEffect } from 'react';
import { getSupabaseBrowser } from '../lib/supabase';
import { getAnonId } from '../lib/anon';
import { initPostHog, posthog } from '../lib/posthog';

export default function SessionWatcher() {
  useEffect(() => {
    initPostHog(); // idempotent — guarantees posthog is ready before identify()
    const supabase = getSupabaseBrowser();
    const { data: sub } = supabase.auth.onAuthStateChange((event, session) => {
      if (event === 'SIGNED_IN' && session?.user) {
        // Merge the prior anonymous PostHog person into the user so the
        // pre-signup funnel connects. A bare identify() won't merge once the
        // person is already identified as the anon id (set on load), so we
        // alias the anon id to the user id first, while distinct_id is still anon.
        const anon = getAnonId();
        if (anon) posthog.alias?.(session.user.id, anon);
        posthog.capture?.('logged_in', {
          method: session.user.app_metadata?.provider,
        });
      }
      if (session?.user) {
        posthog.identify?.(session.user.id, { email: session.user.email });
      }
      if (event === 'SIGNED_OUT') {
        posthog.reset?.();
      }
    });
    return () => sub.subscription.unsubscribe();
  }, []);
  return null;
}
