'use client';
import { createBrowserClient } from '@supabase/ssr';

let _client: ReturnType<typeof createBrowserClient> | null = null;

/** Singleton browser Supabase client (cookie-backed session, auto-refresh). */
export function getSupabaseBrowser() {
  if (!_client) {
    _client = createBrowserClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    );
  }
  return _client;
}

/** Current access token, or null when signed out. Used by apiFetch.
 *  Proactively refreshes when the token is expired or within 60s of expiring,
 *  so the backend always receives a token it will accept (getSession alone can
 *  hand back a stale token if the background refresh timer hasn't fired). */
export async function getAccessToken(): Promise<string | null> {
  try {
    const supabase = getSupabaseBrowser();
    const { data } = await supabase.auth.getSession();
    const session = data.session;
    if (!session) return null;
    const expMs = (session.expires_at ?? 0) * 1000;
    if (expMs && expMs < Date.now() + 60_000) {
      const { data: refreshed } = await supabase.auth.refreshSession();
      return refreshed.session?.access_token ?? session.access_token ?? null;
    }
    return session.access_token ?? null;
  } catch {
    return null;
  }
}
