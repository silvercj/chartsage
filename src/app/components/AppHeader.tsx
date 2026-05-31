'use client';
import { usePathname } from 'next/navigation';
import { signOut } from '../lib/useAuth';
import { useCredits } from '../lib/useCredits';
import CreditsBadge from './CreditsBadge';

export default function AppHeader() {
  const pathname = usePathname();
  const { session } = useCredits();
  const email = session?.user?.email ?? null;

  // Never render on the PDF print route — it must stay chrome-free.
  if (pathname?.startsWith('/report/') && pathname.endsWith('/print')) return null;

  return (
    <header className="border-b border-stone-200 bg-white/80 backdrop-blur">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-4">
        <a href="/" className="flex items-center gap-2 font-semibold tracking-tight text-stone-900">
          <span className="inline-flex w-7 h-7 items-center justify-center rounded-lg bg-stone-900">
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
              <a href="/reports" className="text-stone-600 hover:text-stone-900">Reports</a>
              <a href="/credits" className="text-stone-600 hover:text-stone-900">Credits</a>
              <CreditsBadge />
              <span className="hidden md:inline text-stone-400 max-w-[160px] truncate">{email}</span>
              <button type="button" onClick={signOut} className="text-stone-500 hover:text-stone-900">
                Sign out
              </button>
            </>
          ) : (
            <a href="/login" className="px-3 py-1.5 rounded-lg bg-stone-900 text-white hover:bg-stone-800">
              Sign in
            </a>
          )}
        </div>
      </nav>
    </header>
  );
}
