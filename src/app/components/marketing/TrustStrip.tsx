import { TRUST } from './content';

export default function TrustStrip() {
  return (
    <div className="border-y border-line bg-surface-2">
      <div className="max-w-[1140px] mx-auto px-6 sm:px-8">
        <div className="flex flex-wrap gap-y-2 gap-x-[34px] py-[18px] font-mono text-[12.5px] tracking-[0.02em] text-ink-3">
          {TRUST.map((item) => (
            <span
              key={item}
              className="inline-flex items-center gap-[9px] before:h-[5px] before:w-[5px] before:rounded-full before:bg-accent before:content-['']"
            >
              {item}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
