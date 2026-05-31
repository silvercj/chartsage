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

/** Resolve `p`, but never block longer than `ms` (returns `fallback` on timeout).
 *  Guards against the Supabase auth client occasionally stalling on its internal
 *  lock right after an OAuth callback — a stall must degrade to "no token", not
 *  hang the whole page. */
function withTimeout<T>(p: Promise<T>, ms: number, fallback: T): Promise<T> {
  return Promise.race([
    p.catch(() => fallback),
    new Promise<T>((resolve) => setTimeout(() => resolve(fallback), ms)),
  ]);
}

/** Current access token, or null when signed out / unavailable. Used by apiFetch.
 *  Proactively refreshes when the token is expired or near-expiry, and bounds
 *  every auth call so a stuck client can never freeze callers. */
export async function getAccessToken(): Promise<string | null> {
  try {
    const supabase = getSupabaseBrowser();
    const sessionRes = await withTimeout(supabase.auth.getSession(), 3000, null);
    const session = sessionRes?.data?.session ?? null;
    if (!session) return null;
    const expMs = (session.expires_at ?? 0) * 1000;
    if (expMs && expMs < Date.now() + 60_000) {
      const refreshed = await withTimeout(supabase.auth.refreshSession(), 4000, null);
      const token = refreshed?.data?.session?.access_token;
      if (token) return token;
    }
    return session.access_token ?? null;
  } catch {
    return null;
  }
}
