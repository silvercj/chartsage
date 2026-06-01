'use client';
import { posthog } from '../../lib/posthog';
import { CLOSING, APP_HREF } from './content';

export default function ClosingCta() {
  return (
    <section className="relative overflow-hidden border-t border-line bg-surface-2 py-24 text-center">
      {/* subtle teal radial glow (the mockup's .closing::after) */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -bottom-40 left-1/2 -translate-x-1/2 h-[380px] w-[700px] blur-[24px]"
        style={{
          background:
            'radial-gradient(circle, rgb(45 212 191 / 0.12), transparent 62%)',
        }}
      />
      <div className="relative z-10 max-w-[1140px] mx-auto px-6 sm:px-8">
        <h2 className="font-display font-medium text-[clamp(2rem,3.6vw,2.8rem)] leading-[1.05] tracking-[-0.02em]">
          {CLOSING.h}
        </h2>
        <p className="text-ink-2 text-[17px] font-light mt-3 max-w-[46ch] mx-auto">
          {CLOSING.p}
        </p>
        <div className="mt-7">
          <a
            href={APP_HREF}
            onClick={() => posthog.capture?.('marketing_cta_clicked', { location: 'closing' })}
            className="btn btn-primary"
          >
            {CLOSING.cta}
          </a>
        </div>
      </div>
    </section>
  );
}
