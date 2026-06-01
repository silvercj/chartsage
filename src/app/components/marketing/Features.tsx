import { FEATURES } from './content';

// One small inline glyph per feature, in FEATURES order:
// charts · written summary · data-quality flag · PDF export · more charts · saved.
const ICONS = [
  // bar chart
  <svg key="charts" viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <line x1="7" y1="20" x2="7" y2="11" />
    <line x1="12" y1="20" x2="12" y2="5" />
    <line x1="17" y1="20" x2="17" y2="14" />
  </svg>,
  // written summary (lines of text)
  <svg key="summary" viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <line x1="5" y1="7" x2="19" y2="7" />
    <line x1="5" y1="12" x2="19" y2="12" />
    <line x1="5" y1="17" x2="13" y2="17" />
  </svg>,
  // data-quality flag (alert)
  <svg key="flag" viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <line x1="12" y1="8" x2="12" y2="13" />
    <line x1="12" y1="16.5" x2="12" y2="16.5" />
    <circle cx="12" cy="12" r="9" />
  </svg>,
  // PDF export (download)
  <svg key="pdf" viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M12 4v10" />
    <path d="M8 11l4 4 4-4" />
    <path d="M5 19h14" />
  </svg>,
  // more charts (grid / plus)
  <svg key="more" viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <rect x="4" y="4" width="7" height="7" rx="1.5" />
    <rect x="13" y="4" width="7" height="7" rx="1.5" />
    <rect x="4" y="13" width="7" height="7" rx="1.5" />
    <path d="M16.5 14v5M14 16.5h5" />
  </svg>,
  // saved (refresh / archive)
  <svg key="saved" viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M20 11a8 8 0 1 0-2.3 5.7" />
    <path d="M20 5v6h-6" />
  </svg>,
];

export default function Features() {
  return (
    <section className="py-[84px]">
      <div className="max-w-[1140px] mx-auto px-6 sm:px-8 flex flex-col items-center text-center">
        <span className="eyebrow inline-flex items-center gap-2.5 before:h-px before:w-6 before:bg-accent before:content-['']">
          What you get
        </span>
        <h2 className="font-display font-medium text-[34px] leading-[1.05] tracking-[-0.02em] mt-3.5">
          A report, not a chart dump.
        </h2>
        <div className="grid w-full grid-cols-1 md:grid-cols-3 gap-[18px] mt-12 text-left">
          {FEATURES.map((feature, i) => (
            <div key={feature.h} className="card !rounded-[14px] p-[22px]">
              <div className="flex w-[34px] h-[34px] items-center justify-center rounded-[9px] bg-surface-2 border border-line-2 text-accent mb-3.5">
                {ICONS[i]}
              </div>
              <h3 className="font-display font-medium text-[17px] mb-1.5">{feature.h}</h3>
              <p className="text-ink-2 text-sm font-light leading-relaxed">{feature.p}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
