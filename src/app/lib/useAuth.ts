'use client';
import { getSupabaseBrowser } from './supabase';
import { posthog } from './posthog';

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
