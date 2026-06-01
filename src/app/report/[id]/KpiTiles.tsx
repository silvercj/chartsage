'use client';
import { getFormatter } from '../../lib/format';
import type { KeyMetric } from './useReportLayout';

const FMT: Record<KeyMetric['format'], 'number' | 'currency' | 'percentage'> = {
  number: 'number', currency: 'currency', percent: 'percentage',
};

export default function KpiTiles({ metrics }: { metrics?: KeyMetric[] }) {
  if (!metrics || metrics.length === 0) return null;
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-px bg-line border border-line rounded-2xl overflow-hidden mb-8">
      {metrics.map((m, i) => {
        const fmt = getFormatter(FMT[m.format]);
        return (
          <div key={i} className="bg-surface p-5">
            <div className="font-mono text-2xl font-semibold text-ink tracking-tight">{fmt(m.value)}</div>
            <div className="text-xs text-ink-3 mt-1">{m.label}</div>
          </div>
        );
      })}
    </div>
  );
}
