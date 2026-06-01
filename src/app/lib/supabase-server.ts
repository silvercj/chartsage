import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';

// Read-only server client for Server Components (session check / redirects).
export function getSupabaseServer() {
  const cookieStore = cookies();
  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        get(name: string) {
          return cookieStore.get(name)?.value;
        },
        set() {},    // no-op: Server Components can't set cookies
        remove() {},
      },
    },
  );
}
