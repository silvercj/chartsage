'use client';
import { posthog } from '../../lib/posthog';
import { PRICING, APP_HREF } from './content';

export default function Pricing() {
  return (
    <section id="pricing" className="py-[84px]">
      <div className="max-w-[1140px] mx-auto px-6 sm:px-8 flex flex-col items-center text-center">
        <span className="eyebrow inline-flex items-center gap-2.5 before:h-px before:w-6 before:bg-accent before:content-['']">
          Pricing
        </span>
        <h2 className="font-display font-medium text-[34px] leading-[1.05] tracking-[-0.02em] mt-3.5">
          Your first report is free.
        </h2>
        <p className="text-ink-2 text-[17px] font-light mt-3 max-w-[46ch]">
          No card, no account. Make an account and you get credits to keep going.
        </p>
        <div className="grid w-full grid-cols-1 lg:grid-cols-[1.1fr_1fr] items-center gap-10 mt-11 text-left bg-surface border border-line-2 rounded-[18px] p-[38px]">
          <div>
            <h3 className="font-display font-medium text-[26px] tracking-[-0.01em]">
              {PRICING.blurbH}
            </h3>
            <p className="text-ink-2 text-[15px] font-light mt-3">{PRICING.blurbP}</p>
            <a
              href={APP_HREF}
              onClick={() => posthog.capture?.('marketing_cta_clicked', { location: 'pricing' })}
              className="btn btn-primary mt-[22px]"
            >
              Upload a CSV — free
            </a>
          </div>
          <ul className="flex flex-col gap-3.5 list-none">
            {PRICING.rows.map((row) => (
              <li
                key={row.k}
                className="flex items-baseline justify-between border-b border-line pb-[13px] text-[14.5px]"
              >
                <span className="text-ink">{row.k}</span>
                <span
                  className={`font-mono text-[13px] ${row.muted ? 'text-ink-3' : 'text-accent'}`}
                >
                  {row.v}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
