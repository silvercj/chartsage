'use client';

import dynamic from 'next/dynamic';

const BarChart = dynamic(() => import('./charts/BarChart'), { ssr: false });
const HistogramChart = dynamic(() => import('./charts/HistogramChart'), { ssr: false });
const ScatterChart = dynamic(() => import('./charts/ScatterChart'), { ssr: false });
const LineChart = dynamic(() => import('./charts/LineChart'), { ssr: false });
const PieChart = dynamic(() => import('./charts/PieChart'), { ssr: false });
const BoxPlot = dynamic(() => import('./charts/BoxPlot'), { ssr: false });
const Heatmap = dynamic(() => import('./charts/Heatmap'), { ssr: false });

interface Props {
  index: number;
  spec: any;
  caption: string;
}

const KIND_LABEL: Record<string, string> = {
  bar: 'Bar',
  histogram: 'Histogram',
  scatter: 'Scatter',
  line: 'Trend',
  pie: 'Composition',
  box: 'Distribution',
  heatmap: 'Heatmap',
};

export default function ChartCard({ index, spec, caption }: Props) {
  const renderer = (() => {
    switch (spec.kind) {
      case 'bar': return <BarChart spec={spec} />;
      case 'histogram': return <HistogramChart spec={spec} />;
      case 'scatter': return <ScatterChart spec={spec} />;
      case 'line': return <LineChart spec={spec} />;
      case 'pie': return <PieChart spec={spec} />;
      case 'box': return <BoxPlot spec={spec} />;
      case 'heatmap': return <Heatmap spec={spec} />;
      default: return <p className="text-sm text-red-600">Unsupported chart kind: {String(spec.kind)}</p>;
    }
  })();

  const kindLabel = KIND_LABEL[spec.kind] ?? spec.kind;

  return (
    <section className="bg-white rounded-2xl ring-1 ring-stone-200/80 shadow-[0_1px_3px_rgba(0,0,0,0.04)] hover:shadow-[0_4px_12px_rgba(0,0,0,0.06)] transition-shadow p-6 flex flex-col">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-baseline gap-3">
          <span className="text-xs font-mono text-stone-400 tabular-nums">
            {String(index).padStart(2, '0')}
          </span>
          <h2 className="text-base font-semibold text-stone-900 leading-snug tracking-tight">
            {spec.title}
          </h2>
        </div>
        <span className="text-[10px] uppercase tracking-widest text-stone-400 mt-1">
          {kindLabel}
        </span>
      </div>
      <div className="flex-1 min-h-[300px]">{renderer}</div>
      {caption && (
        <p className="text-sm text-stone-600 mt-4 pt-4 border-t border-stone-100 leading-relaxed">
          {caption}
        </p>
      )}
    </section>
  );
}
