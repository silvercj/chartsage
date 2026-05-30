import { NextRequest, NextResponse } from 'next/server';
import { createServerClient, type CookieOptions } from '@supabase/ssr';
import { cookies } from 'next/headers';

export const dynamic = 'force-dynamic';

/** Only allow internal relative paths as redirect targets (no open redirect). */
function safeNext(raw: string | null): string {
  if (raw && raw.startsWith('/') && !raw.startsWith('//')) return raw;
  return '/';
}

export async function GET(request: NextRequest) {
  const { searchParams, origin } = new URL(request.url);
  const code = searchParams.get('code');
  const next = safeNext(searchParams.get('next'));

  if (code) {
    const cookieStore = cookies();
    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          getAll() {
            return cookieStore.getAll();
          },
          setAll(cookiesToSet: { name: string; value: string; options: CookieOptions }[]) {
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options),
            );
          },
        },
      },
    );

    const { data } = await supabase.auth.exchangeCodeForSession(code);

    // Migrate the anonymous visitor's existing report(s) into the new account.
    // Idempotent on the backend, so a failure here is non-fatal.
    const token = data.session?.access_token;
    const anon = cookieStore.get('chartsage_anon')?.value;
    if (token && anon) {
      try {
        await fetch(`${process.env.NEXT_PUBLIC_API_URL}/claim-anon-reports`, {
          method: 'POST',
          headers: { Authorization: `Bearer ${token}`, 'X-Anon-Id': anon },
        });
      } catch {
        /* non-fatal */
      }
    }

    return NextResponse.redirect(`${origin}/welcome?next=${encodeURIComponent(next)}`);
  }

  // No code present — OAuth was cancelled or the link expired.
  return NextResponse.redirect(`${origin}/login`);
}
