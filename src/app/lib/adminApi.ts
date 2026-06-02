'use client';

const TOKEN_KEY = 'cs_admin_token';

export function getAdminToken(): string {
  if (typeof window === 'undefined') return '';
  return sessionStorage.getItem(TOKEN_KEY) || '';
}

export function setAdminToken(token: string): void {
  sessionStorage.setItem(TOKEN_KEY, token.trim());
}

export function clearAdminToken(): void {
  sessionStorage.removeItem(TOKEN_KEY);
}

/** Fetch against the backend with the admin token header (NOT the Supabase Bearer). */
export async function adminFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers || {});
  headers.set('X-Admin-Token', getAdminToken());
  return fetch(`${process.env.NEXT_PUBLIC_API_URL}${path}`, { ...init, headers });
}
