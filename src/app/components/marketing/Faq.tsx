import { FAQ } from './content';

export default function Faq() {
  return (
    <section id="faq" className="border-t border-line bg-surface-2 py-[84px]">
      <div className="max-w-[1140px] mx-auto px-6 sm:px-8 flex flex-col items-center text-center">
        <span className="eyebrow inline-flex items-center gap-2.5 before:h-px before:w-6 before:bg-accent before:content-['']">
          FAQ
        </span>
        <h2 className="font-display font-medium text-[34px] leading-[1.05] tracking-[-0.02em] mt-3.5">
          Questions worth asking.
        </h2>
        <div className="w-full max-w-[760px] mt-11 text-left">
          {FAQ.map((item) => (
            <div key={item.q} className="border-b border-line py-[22px]">
              <h4 className="font-semibold text-ink text-[16.5px] mb-2">{item.q}</h4>
              <p className="text-ink-2 text-[14.5px] font-light max-w-[68ch]">{item.a}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
