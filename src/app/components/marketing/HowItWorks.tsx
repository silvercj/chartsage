import { HOW } from './content';

export default function HowItWorks() {
  return (
    <section id="how" className="py-[84px]">
      <div className="max-w-[1140px] mx-auto px-6 sm:px-8 flex flex-col items-center text-center">
        <span className="eyebrow inline-flex items-center gap-2.5 before:h-px before:w-6 before:bg-accent before:content-['']">
          How it works
        </span>
        <h2 className="font-display font-medium text-[34px] leading-[1.05] tracking-[-0.02em] mt-3.5">
          Three steps. About thirty seconds.
        </h2>
        <div className="grid w-full grid-cols-1 md:grid-cols-3 gap-[22px] mt-12 text-left">
          {HOW.map((step) => (
            <div key={step.n} className="card p-[26px]">
              <div className="font-mono text-xs text-accent tracking-[0.1em]">{step.n}</div>
              <h3 className="font-display font-medium text-xl mt-3.5 mb-2">{step.h}</h3>
              <p className="text-ink-2 text-[14.5px] font-light leading-relaxed">{step.p}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
