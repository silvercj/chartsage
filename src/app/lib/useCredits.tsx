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
    try {
      const res = await apiFetch('/me');   // apiFetch resolves the (bounded) token; 401 when signed out
      setBalance(res.ok ? ((await res.json()).credits_balance ?? null) : null);
    } catch {
      setBalance(null);
    }
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
