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

    // Only send brand-new accounts to onboarding; returning users go straight
    // to `next`. "New" = created in the last 2 min (this very sign-up), read
    // server-side from the Supabase user rather than a per-browser/per-domain
    // localStorage flag (which re-fires on a new device or the new domain).
    const user = data.user;
    const createdAtMs = user?.created_at ? new Date(user.created_at).getTime() : 0;
    const isNewSignup = createdAtMs > 0 && Date.now() - createdAtMs < 120_000;
    const dest = isNewSignup ? `/welcome?next=${encodeURIComponent(next)}` : next;
    return NextResponse.redirect(`${origin}${dest}`);
  }

  // No code present — OAuth was cancelled or the link expired.
  return NextResponse.redirect(`${origin}/login`);
}
