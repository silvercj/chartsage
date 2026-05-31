'use client';
import { createContext, useCallback, useContext, useEffect, useState } from 'react';
import { apiFetch } from './api';
import { getSupabaseBrowser } from './supabase';

interface CreditsState {
  balance: number | null;        // null = signed out / unknown
  refetch: () => Promise<void>;
}

const CreditsContext = createContext<CreditsState>({ balance: null, refetch: async () => {} });

export function CreditsProvider({ children }: { children: React.ReactNode }) {
  const [balance, setBalance] = useState<number | null>(null);

  const refetch = useCallback(async () => {
    const { data } = await getSupabaseBrowser().auth.getSession();
    if (!data.session) { setBalance(null); return; }
    try {
      const res = await apiFetch('/me');
      if (res.ok) setBalance((await res.json()).credits_balance ?? null);
    } catch { /* leave stale */ }
  }, []);

  useEffect(() => {
    refetch();
    const { data: sub } = getSupabaseBrowser().auth.onAuthStateChange(() => refetch());
    return () => sub.subscription.unsubscribe();
  }, [refetch]);

  return <CreditsContext.Provider value={{ balance, refetch }}>{children}</CreditsContext.Provider>;
}

export function useCredits() {
  return useContext(CreditsContext);
}
