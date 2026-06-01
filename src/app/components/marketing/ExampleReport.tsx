'use client';
import ChartContent from '../../report/[id]/charts/ChartContent';
import { SAMPLE_REPORT } from './sampleReport';

export default function ExampleReport() {
  const r = SAMPLE_REPORT;
  return (
    <section id="example" className="border-y border-line bg-surface-2 py-[84px]">
      <div className="max-w-[1140px] mx-auto px-6 sm:px-8">
        {/* Dark-band heading sits outside the light card */}
        <div className="flex flex-col items-center text-center">
          <span className="eyebrow inline-flex items-center gap-2.5 before:h-px before:w-6 before:bg-accent before:content-['']">
            A real one
          </span>
          <h2 className="font-display font-medium text-[34px] leading-[1.05] tracking-[-0.02em] mt-3.5">
            Here&apos;s a report it made.
          </h2>
          <p className="text-ink-2 text-[17px] font-light mt-3 max-w-[56ch]">
            Output from a sample sales dataset — charts, a written summary, and a note on the one
            column that didn&apos;t look right. Nothing staged.
          </p>
        </div>

        {/* The embedded report renders light, exactly like production. */}
        <div className="theme-light max-w-3xl mx-auto mt-12">
          <div className="card shadow-card-lg rounded-2xl p-6 bg-surface text-ink">
            <p className="font-display text-2xl mb-2 text-ink">Regional Sales — FY24</p>
            <p className="font-mono text-xs uppercase tracking-wide text-ink-3 mb-5">
              Generated · sample · {r.charts.length} charts
            </p>
            <p className="font-display text-ink-2 mb-6 leading-relaxed">{r.summary}</p>
            <div className="grid gap-6">
              {r.charts.map((c) => (
                <div key={c.chart_id} className="border border-line rounded-xl p-4">
                  <div className="min-h-[260px]">
                    <ChartContent spec={c.spec} />
                  </div>
                  {c.caption && (
                    <p className="font-display italic text-ink-2 text-sm mt-3 leading-relaxed">
                      {c.caption}
                    </p>
                  )}
                </div>
              ))}
            </div>
            {r.data_quality.length > 0 && (
              <p className="mt-6 font-mono text-xs text-ink-3">Data note: {r.data_quality[0]}</p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
