'use client';
import { getAnonId } from './anon';
import { getAccessToken } from './supabase';

export interface ApiError extends Error {
  status: number;
  code?: string;
  detail?: unknown;
}

export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers || {});
  const anonId = getAnonId();
  if (anonId) headers.set('X-Anon-Id', anonId);

  const token = await getAccessToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);

  const url = `${process.env.NEXT_PUBLIC_API_URL}${path}`;
  return fetch(url, { ...init, headers });
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
