'use client';
import { usePathname } from 'next/navigation';
import { signOut } from '../lib/useAuth';
import { useCredits } from '../lib/useCredits';
import CreditsBadge from './CreditsBadge';

export default function AppHeader() {
  const pathname = usePathname();
  const { session } = useCredits();
  const email = session?.user?.email ?? null;

  // Never render on the chrome-free report routes: the PDF print route, the public
  // embed route, and the OG social-card route all render bare so an iframe/screenshot
  // has no header.
  if (
    pathname &&
    (pathname.endsWith('/embed') || pathname.endsWith('/print') || pathname.endsWith('/og'))
  )
    return null;

  // Marketing landing has its own nav; app header is for app routes only.
  if (pathname === '/') return null;

  return (
    <header className="border-b border-line bg-canvas/80 backdrop-blur">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
        <a href="/" className="flex items-center gap-2 font-display text-lg font-medium text-ink">
          <span className="inline-flex w-7 h-7 items-center justify-center rounded-lg bg-surface-2 border border-line-2">
            <svg viewBox="0 0 32 32" className="w-4 h-4" aria-hidden="true">
              <rect x="6.5" y="17" width="4.5" height="8.5" rx="1.4" fill="#5EEAD4" />
              <rect x="13.75" y="11.5" width="4.5" height="14" rx="1.4" fill="#2DD4BF" />
              <rect x="21" y="7" width="4.5" height="18.5" rx="1.4" fill="#0D9488" />
            </svg>
          </span>
          ChartSage
        </a>
        <div className="flex items-center gap-3 text-sm">
          {email ? (
            <>
              <a href="/reports" className="text-ink-2 hover:text-ink transition-colors">Reports</a>
              <a href="/credits" className="text-ink-2 hover:text-ink transition-colors">Credits</a>
              <CreditsBadge />
              <span className="hidden md:inline font-mono text-xs text-ink-3 max-w-[160px] truncate">{email}</span>
              <button type="button" onClick={signOut} className="text-ink-2 hover:text-ink transition-colors">
                Sign out
              </button>
            </>
          ) : (
            <a href="/login" className="btn btn-primary !px-4 !py-2">
              Sign in
            </a>
          )}
        </div>
      </nav>
    </header>
  );
}
