export default function MarketingFooter() {
  return (
    <footer className="border-t border-line py-10">
      <div className="max-w-[1140px] mx-auto px-6 sm:px-8 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2.5 text-ink">
          <span className="inline-flex w-8 h-8 items-center justify-center rounded-lg bg-surface-2 border border-line-2">
            <svg viewBox="0 0 32 32" className="w-4 h-4" aria-hidden="true">
              <rect x="6.5" y="17" width="4.5" height="8.5" rx="1.4" fill="#5EEAD4" />
              <rect x="13.75" y="11.5" width="4.5" height="14" rx="1.4" fill="#2DD4BF" />
              <rect x="21" y="7" width="4.5" height="18.5" rx="1.4" fill="#0D9488" />
            </svg>
          </span>
          <b className="font-display text-[19px] font-medium">ChartSage</b>
        </div>
        <div className="flex flex-wrap gap-[22px] text-[13.5px] text-ink-2">
          <a href="#how" className="hover:text-ink transition-colors">How it works</a>
          <a href="#example" className="hover:text-ink transition-colors">Example</a>
          <a href="#pricing" className="hover:text-ink transition-colors">Pricing</a>
          <a href="#faq" className="hover:text-ink transition-colors">FAQ</a>
          <a href="/login" className="hover:text-ink transition-colors">Sign in</a>
          <a href="/terms" className="hover:text-ink transition-colors">Terms</a>
          <a href="/privacy" className="hover:text-ink transition-colors">Privacy</a>
        </div>
        <small className="font-mono text-[11.5px] text-ink-3">© ChartSage</small>
      </div>
    </footer>
  );
}
