'use client';
import { usePathname } from 'next/navigation';
import { useAuthEmail, signOut } from '../lib/useAuth';

export default function AuthNav() {
  const pathname = usePathname();
  const email = useAuthEmail();

  // Report views render their own account controls inside the report toolbar
  // (and the /print route must stay clean for the PDF), so the floating nav
  // is suppressed there.
  if (pathname?.startsWith('/report/')) return null;

  return (
    <nav className="fixed top-3 right-4 z-50 flex items-center gap-3 text-sm">
      {email ? (
        <>
          <a
            href="/reports"
            className="px-3 py-1.5 rounded-lg bg-white/90 ring-1 ring-stone-200 text-stone-700 hover:bg-white"
          >
            My reports
          </a>
          <span className="hidden sm:inline text-stone-400 max-w-[160px] truncate">{email}</span>
          <button
            type="button"
            onClick={signOut}
            className="px-3 py-1.5 rounded-lg text-stone-500 hover:text-stone-900"
          >
            Sign out
          </button>
        </>
      ) : (
        <a
          href="/login"
          className="px-3 py-1.5 rounded-lg bg-stone-900 text-white hover:bg-stone-800"
        >
          Sign in
        </a>
      )}
    </nav>
  );
}
