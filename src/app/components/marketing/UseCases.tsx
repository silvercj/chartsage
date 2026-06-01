import { USES } from './content';

export default function UseCases() {
  return (
    <section className="border-y border-line bg-surface-2 py-[84px]">
      <div className="max-w-[1140px] mx-auto px-6 sm:px-8 flex flex-col items-center text-center">
        <span className="eyebrow inline-flex items-center gap-2.5 before:h-px before:w-6 before:bg-accent before:content-['']">
          Who it&apos;s for
        </span>
        <h2 className="font-display font-medium text-[34px] leading-[1.05] tracking-[-0.02em] mt-3.5">
          Bring the data you already have.
        </h2>
        <p className="text-ink-2 text-[17px] font-light mt-3 max-w-[46ch]">
          If it opens in a spreadsheet, it works here.
        </p>
        <div className="grid w-full grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-11 text-left">
          {USES.map((use) => (
            <div key={use.h} className="bg-surface-2 border border-line rounded-[13px] p-5">
              <h4 className="font-display font-medium text-base mb-[7px]">{use.h}</h4>
              <p className="font-mono text-ink-3 text-[11.5px] leading-[1.7]">{use.p}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
