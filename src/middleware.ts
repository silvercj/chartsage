import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

const COOKIE_NAME = 'chartsage_anon';
const ONE_YEAR_SECONDS = 60 * 60 * 24 * 365;

export function middleware(req: NextRequest) {
  const res = NextResponse.next();
  const existing = req.cookies.get(COOKIE_NAME);
  if (!existing) {
    const uuid = crypto.randomUUID();
    res.cookies.set({
      name: COOKIE_NAME,
      value: uuid,
      httpOnly: true,
      sameSite: 'lax',
      secure: process.env.NODE_ENV === 'production',
      maxAge: ONE_YEAR_SECONDS,
      path: '/',
    });
    // Mirror to a non-httpOnly cookie so JS can read it for header injection + PostHog identify
    res.cookies.set({
      name: 'chartsage_anon_pub',
      value: uuid,
      httpOnly: false,
      sameSite: 'lax',
      secure: process.env.NODE_ENV === 'production',
      maxAge: ONE_YEAR_SECONDS,
      path: '/',
    });
  }
  return res;
}

export const config = {
  // Skip static files and Next.js internals
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};
