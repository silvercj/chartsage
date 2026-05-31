'use client';
import { getAnonId } from './anon';
import { getAccessToken } from './supabase';

export interface ApiError extends Error {
  status: number;
  code?: string;
  detail?: unknown;
}

export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const url = `${process.env.NEXT_PUBLIC_API_URL}${path}`;

  const send = async (): Promise<Response> => {
    const headers = new Headers(init.headers || {});
    const anonId = getAnonId();
    if (anonId) headers.set('X-Anon-Id', anonId);
    const token = await getAccessToken();
    if (token) headers.set('Authorization', `Bearer ${token}`);
    return fetch(url, { ...init, headers });
  };

  let res = await send();

  // Right after an OAuth redirect the Supabase client can take a moment to settle,
  // so the first authed call may 401 before the token is ready. For idempotent GETs,
  // wait briefly and retry once IF a session has since become available — this turns
  // the transient race into a clean 200 instead of bouncing the user to /login.
  const method = (init.method ?? 'GET').toUpperCase();
  if (res.status === 401 && method === 'GET') {
    await new Promise((r) => setTimeout(r, 600));
    if (await getAccessToken()) {
      res = await send();
    }
  }

  return res;
}

export async function apiJSON<T = any>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await apiFetch(path, init);
  if (!res.ok) {
    let body: any = null;
    try { body = await res.json(); } catch {}
    const err = new Error(body?.detail?.message || `Request failed (${res.status})`) as ApiError;
    err.status = res.status;
    err.code = body?.detail?.code;
    err.detail = body?.detail;
    throw err;
  }
  return res.json();
}
