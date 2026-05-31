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
        <a href="/" className="font-semibold tracking-tight text-stone-900">ChartSage</a>
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
