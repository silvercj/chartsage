'use client';

import dynamic from 'next/dynamic';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

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
  chartId: string;
  onHide?: (chartId: string) => void;
}

const KIND_LABEL: Record<string, string> = {
  bar: 'Bar', histogram: 'Histogram', scatter: 'Scatter',
  line: 'Trend', pie: 'Composition', box: 'Distribution', heatmap: 'Heatmap',
};

function ChartContent({ spec }: { spec: any }) {
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
}

export default function ChartCard({ index, spec, caption, chartId, onHide }: Props) {
  const sortable = useSortable({ id: chartId });
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = sortable;

  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.4 : 1,
  };

  const kindLabel = KIND_LABEL[spec.kind] ?? spec.kind;

  return (
    <section
      ref={setNodeRef}
      style={style}
      className="bg-white rounded-2xl ring-1 ring-stone-200/80 shadow-[0_1px_3px_rgba(0,0,0,0.04)] hover:shadow-[0_4px_12px_rgba(0,0,0,0.06)] transition-shadow p-6 flex flex-col"
    >
      <div className="flex items-start justify-between mb-3 gap-2">
        <div className="flex items-baseline gap-3 min-w-0 flex-1">
          <button
            type="button"
            className="text-stone-300 hover:text-stone-600 cursor-grab active:cursor-grabbing -ml-1 px-1 leading-none focus:outline-none focus:ring-2 focus:ring-stone-400 rounded"
            aria-label="Drag to reorder"
            {...attributes}
            {...listeners}
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <circle cx="6" cy="5" r="1.5" /><circle cx="10" cy="5" r="1.5" /><circle cx="14" cy="5" r="1.5" />
              <circle cx="6" cy="10" r="1.5" /><circle cx="10" cy="10" r="1.5" /><circle cx="14" cy="10" r="1.5" />
              <circle cx="6" cy="15" r="1.5" /><circle cx="10" cy="15" r="1.5" /><circle cx="14" cy="15" r="1.5" />
            </svg>
          </button>
          <span className="text-xs font-mono text-stone-400 tabular-nums shrink-0">
            {String(index).padStart(2, '0')}
          </span>
          <h2 className="text-base font-semibold text-stone-900 leading-snug tracking-tight truncate">
            {spec.title}
          </h2>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className="text-[10px] uppercase tracking-widest text-stone-400 mt-1">
            {kindLabel}
          </span>
          {onHide && (
            <button
              type="button"
              onClick={() => onHide(chartId)}
              className="text-stone-300 hover:text-stone-700 transition-colors w-6 h-6 flex items-center justify-center rounded hover:bg-stone-100 focus:outline-none focus:ring-2 focus:ring-stone-400"
              aria-label="Move to sidebar"
              title="Move to sidebar"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>
      <div className="flex-1 min-h-[300px]">
        <ChartContent spec={spec} />
      </div>
      {caption && (
        <p className="text-sm text-stone-600 mt-4 pt-4 border-t border-stone-100 leading-relaxed">
          {caption}
        </p>
      )}
    </section>
  );
}
