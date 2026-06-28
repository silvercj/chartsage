'use client';
import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { posthog } from '../../lib/posthog';
import { openSampleReport } from '../../lib/sample';
import { HERO, APP_HREF } from './content';

export default function Hero() {
  const router = useRouter();
  const [loadingSample, setLoadingSample] = useState(false);

  useEffect(() => {
    posthog.capture?.('landing_hero_viewed');
  }, []);

  async function onSample(location: string) {
    if (loadingSample) return;
    setLoadingSample(true);
    const ok = await openSampleReport((href) => router.push(href), location);
    if (!ok) setLoadingSample(false);   // stay put on failure; navigation unmounts on success
  }

  return (
    <header className="relative overflow-hidden">
      {/* teal radial glow (the mockup's .hero::after) */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-24 -right-20 z-0 h-[480px] w-[600px] blur-[20px]"
        style={{
          background:
            'radial-gradient(circle at 60% 40%, rgb(45 212 191 / 0.16), transparent 60%)',
        }}
      />
      <div className="relative z-10 max-w-[1140px] mx-auto px-6 sm:px-8 pt-20 pb-[70px]">
        <div className="grid items-center gap-14 md:grid-cols-[1.05fr_0.95fr]">
          <div>
            <span className="eyebrow inline-flex items-center gap-2.5 before:h-px before:w-6 before:bg-accent before:content-['']">
              {HERO.eyebrow}
            </span>
            <h1 className="font-display font-normal text-[clamp(2.6rem,4.8vw,4rem)] leading-[1.04] tracking-[-0.025em] mt-5">
              {HERO.headlineA}
              <em className="italic font-medium text-accent">{HERO.headlineEm}</em>
              {HERO.headlineB}
            </h1>
            <p className="text-ink-2 text-lg leading-relaxed mt-5 max-w-[34ch] font-light">
              {HERO.sub}
            </p>
            <div className="flex flex-wrap items-center gap-3.5 mt-8">
              <a
                href={APP_HREF}
                onClick={() => posthog.capture?.('marketing_cta_clicked', { location: 'hero' })}
                className="btn btn-primary"
              >
                {HERO.ctaPrimary}
              </a>
              <button
                type="button"
                onClick={() => onSample('hero')}
                disabled={loadingSample}
                className="btn btn-ghost disabled:opacity-60"
              >
                {loadingSample ? 'Loading sample…' : HERO.ctaSecondary}
              </button>
            </div>
            <p className="text-ink-3 text-sm mt-3">No file handy? {HERO.ctaSecondary} — no signup needed.</p>
          </div>

          {/* hero report preview (light) — clickable: opens the real sample report */}
          <div className="theme-light">
            <div
              role="button"
              tabIndex={0}
              aria-label="Open a live sample report"
              onClick={() => onSample('hero_preview')}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onSample('hero_preview'); } }}
              className="bg-surface border border-line rounded-2xl shadow-card-lg overflow-hidden text-ink cursor-pointer transition-shadow hover:shadow-card-lg focus:outline-none focus:ring-2 focus:ring-accent"
            >
              <div className="flex items-start justify-between border-b border-line px-5 pt-[18px] pb-3.5">
                <div>
                  <h4 className="font-display font-medium text-lg">Q3 Sales Performance</h4>
                  <div className="font-mono text-[10px] uppercase tracking-[0.05em] text-ink-3 mt-1">
                    Generated · 2.1s · 10 charts
                  </div>
                </div>
                <span className="font-sans text-[11.5px] font-medium text-on-accent bg-accent rounded-lg px-[11px] py-1.5">
                  Export PDF
                </span>
              </div>
              <div className="grid grid-cols-3 border-b border-line">
                <div className="border-r border-line px-4 py-3.5">
                  <div className="font-mono font-semibold text-[21px] text-ink">$1.24M</div>
                  <div className="text-[11px] text-ink-3 mt-0.5">Revenue</div>
                  <div className="font-mono text-[10px] text-accent">▲ 18%</div>
                </div>
                <div className="border-r border-line px-4 py-3.5">
                  <div className="font-mono font-semibold text-[21px] text-ink">4,210</div>
                  <div className="text-[11px] text-ink-3 mt-0.5">Orders</div>
                  <div className="font-mono text-[10px] text-accent">▲ 9%</div>
                </div>
                <div className="px-4 py-3.5">
                  <div className="font-mono font-semibold text-[21px] text-ink">$294</div>
                  <div className="text-[11px] text-ink-3 mt-0.5">Avg order</div>
                  <div className="font-mono text-[10px] text-accent">▲ 6%</div>
                </div>
              </div>
              <div className="px-[18px] py-4">
                <div className="flex gap-3.5 font-mono text-[10px] text-ink-2 mb-2">
                  <span className="inline-flex items-center">
                    <i className="inline-block w-2 h-2 rounded-[2px] mr-[5px] align-middle" style={{ background: '#0C5C52' }} />
                    Revenue
                  </span>
                  <span className="inline-flex items-center">
                    <i className="inline-block w-2 h-2 rounded-[2px] mr-[5px] align-middle" style={{ background: '#C99A3F' }} />
                    Target
                  </span>
                </div>
                <svg viewBox="0 0 480 170" width="100%" height="150" preserveAspectRatio="none">
                  <g stroke="#E6E0D4" strokeWidth="1">
                    <line x1="0" y1="34" x2="480" y2="34" />
                    <line x1="0" y1="78" x2="480" y2="78" />
                    <line x1="0" y1="122" x2="480" y2="122" />
                  </g>
                  <polyline points="10,120 95,112 185,100 270,104 360,72 470,52" fill="none" stroke="#C99A3F" strokeWidth="2" strokeDasharray="4 5" />
                  <defs>
                    <linearGradient id="hero-rev" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#0C5C52" stopOpacity="0.2" />
                      <stop offset="100%" stopColor="#0C5C52" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  <path d="M10,128 L95,118 L185,92 L270,100 L360,60 L470,44 L470,150 L10,150 Z" fill="url(#hero-rev)" />
                  <polyline points="10,128 95,118 185,92 270,100 360,60 470,44" fill="none" stroke="#0C5C52" strokeWidth="2.5" />
                </svg>
              </div>
              <div className="font-display italic text-[13.5px] text-ink-2 px-5 pt-0.5 pb-[18px] leading-[1.5]">
                Revenue climbed steadily through the quarter, led by a 31% lift in the West region.
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
