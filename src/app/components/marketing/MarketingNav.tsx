'use client';
import { APP_HREF } from './content';

export default function MarketingNav() {
  return (
    <nav className="sticky top-0 z-50 border-b border-line bg-canvas/80 backdrop-blur">
      <div className="max-w-[1140px] mx-auto px-6 sm:px-8 h-16 flex items-center justify-between gap-4">
        <a href="/" className="flex items-center gap-2.5 text-ink">
          <span className="inline-flex w-8 h-8 items-center justify-center rounded-lg bg-surface-2 border border-line-2">
            <svg viewBox="0 0 32 32" className="w-4 h-4" aria-hidden="true">
              <rect x="6.5" y="17" width="4.5" height="8.5" rx="1.4" fill="#5EEAD4" />
              <rect x="13.75" y="11.5" width="4.5" height="14" rx="1.4" fill="#2DD4BF" />
              <rect x="21" y="7" width="4.5" height="18.5" rx="1.4" fill="#0D9488" />
            </svg>
          </span>
          <b className="font-display text-[19px] font-medium">ChartSage</b>
        </a>
        <div className="flex items-center gap-6 sm:gap-[26px]">
          <a href="#how" className="hidden md:inline text-sm text-ink-2 hover:text-ink transition-colors">How it works</a>
          <a href="#example" className="hidden md:inline text-sm text-ink-2 hover:text-ink transition-colors">Example</a>
          <a href="#pricing" className="hidden md:inline text-sm text-ink-2 hover:text-ink transition-colors">Pricing</a>
          <a href="#faq" className="hidden md:inline text-sm text-ink-2 hover:text-ink transition-colors">FAQ</a>
          <a href="/login" className="text-sm text-ink-2 hover:text-ink transition-colors">Sign in</a>
          <a href={APP_HREF} className="btn btn-primary !px-4 !py-[9px]">Upload a CSV</a>
        </div>
      </div>
    </nav>
  );
}
