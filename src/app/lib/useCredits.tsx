'use client';
import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import type { Session } from '@supabase/supabase-js';
import { apiFetch } from './api';
import { getSupabaseBrowser } from './supabase';

interface CreditsState {
  balance: number | null;        // null = signed out / unknown
  session: Session | null;       // the settled Supabase session (null = signed out)
  authLoading: boolean;          // true until the initial auth state is known
  refetch: () => Promise<void>;
}

const CreditsContext = createContext<CreditsState>({
  balance: null,
  session: null,
  authLoading: true,
  refetch: async () => {},
});

/** Single source of settled auth + credit state for the app.
 *  Resolves the session via onAuthStateChange (which emits INITIAL_SESSION once
 *  the client has loaded the session from storage) rather than racing getSession
 *  on mount — so consumers never act on a half-settled auth state. The balance is
 *  only fetched once a session is actually present. */
export function CreditsProvider({ children }: { children: React.ReactNode }) {
  const [balance, setBalance] = useState<number | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [authLoading, setAuthLoading] = useState(true);

  const refetch = useCallback(async () => {
    try {
      const res = await apiFetch('/me');
      setBalance(res.ok ? ((await res.json()).credits_balance ?? null) : null);
    } catch {
      setBalance(null);
    }
  }, []);

  useEffect(() => {
    const supabase = getSupabaseBrowser();
    let resolved = false;
    const apply = (s: Session | null) => {
      resolved = true;
      setSession(s);
      setAuthLoading(false);
      if (s) refetch();
      else setBalance(null);
    };
    const { data: sub } = supabase.auth.onAuthStateChange((_event, s) => apply(s));
    // Fallback: if the initial event is somehow missed, resolve via getSession.
    const t = setTimeout(() => {
      if (!resolved) {
        supabase.auth.getSession().then(({ data }) => { if (!resolved) apply(data.session); });
      }
    }, 2500);
    return () => { clearTimeout(t); sub.subscription.unsubscribe(); };
  }, [refetch]);

  return (
    <CreditsContext.Provider value={{ balance, session, authLoading, refetch }}>
      {children}
    </CreditsContext.Provider>
  );
}

export function useCredits() {
  return useContext(CreditsContext);
}
