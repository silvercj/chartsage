'use client';
import { useEffect } from 'react';
import { getSupabaseBrowser } from '../lib/supabase';
import { getAnonId } from '../lib/anon';
import { initPostHog, posthog } from '../lib/posthog';

// supabase-js fires `SIGNED_IN` on EVERY page load / tab focus / token refresh
// when it restores a session, not just on a real login. Without a guard that
// makes `logged_in` (and the one-time anon→user alias) fire on every navigation,
// massively inflating the funnel. This per-user flag fires them once per actual
// login and is cleared on sign-out so the next genuine login re-fires.
const LOGGED_IN_FLAG = 'cs_logged_in_fired';

function readFlag(): string | null {
  try { return localStorage.getItem(LOGGED_IN_FLAG); } catch { return null; }
}
function writeFlag(v: string | null) {
  try { v === null ? localStorage.removeItem(LOGGED_IN_FLAG) : localStorage.setItem(LOGGED_IN_FLAG, v); } catch { /* ignore */ }
}

export default function SessionWatcher() {
  useEffect(() => {
    initPostHog(); // idempotent — guarantees posthog is ready before identify()
    const supabase = getSupabaseBrowser();
    const { data: sub } = supabase.auth.onAuthStateChange((event, session) => {
      if (event === 'SIGNED_IN' && session?.user && readFlag() !== session.user.id) {
        // First SIGNED_IN for this user since the last sign-out = a real login.
        // Merge the prior anonymous PostHog person into the user so the
        // pre-signup funnel connects. A bare identify() won't merge once the
        // person is already identified as the anon id (set on load), so we
        // alias the anon id to the user id first, while distinct_id is still anon.
        const anon = getAnonId();
        if (anon) posthog.alias?.(session.user.id, anon);
        posthog.capture?.('logged_in', {
          method: session.user.app_metadata?.provider,
        });
        writeFlag(session.user.id);
      }
      if (session?.user) {
        posthog.identify?.(session.user.id, { email: session.user.email });
      }
      if (event === 'SIGNED_OUT') {
        writeFlag(null);
        posthog.reset?.();
      }
    });
    return () => sub.subscription.unsubscribe();
  }, []);
  return null;
}
