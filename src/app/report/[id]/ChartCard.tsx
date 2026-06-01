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
  printMode?: boolean;   // hides interactive chrome (drag handle, kind badge) for PDF export
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
    default: return <p className="text-sm text-ember">Unsupported chart kind: {String(spec.kind)}</p>;
  }
}

export default function ChartCard({ index, spec, caption, chartId, onHide, printMode }: Props) {
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
      className="card shadow-card rounded-2xl p-5 flex flex-col"
    >
      <div className="flex items-start justify-between mb-3 gap-2">
        <div className="flex items-baseline gap-3 min-w-0 flex-1">
          {!printMode && (
            <button
              type="button"
              className="text-ink-3 hover:text-ink-2 cursor-grab active:cursor-grabbing -ml-1 px-1 leading-none focus:outline-none focus:ring-2 focus:ring-accent rounded"
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
          )}
          <span className="font-mono text-xs text-ink-3 tabular-nums shrink-0">
            {String(index).padStart(2, '0')}
          </span>
          <h2 className="font-display text-lg text-ink leading-snug truncate">
            {spec.title}
          </h2>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {!printMode && (
            <span className="font-mono text-xs uppercase tracking-wide text-ink-3 mt-1">
              {kindLabel}
            </span>
          )}
          {onHide && (
            <button
              type="button"
              onClick={() => onHide(chartId)}
              className="text-ink-3 hover:text-ink transition-colors w-6 h-6 flex items-center justify-center rounded hover:bg-surface-2 focus:outline-none focus:ring-2 focus:ring-accent"
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
        <p className="font-display italic text-ink-2 text-sm mt-4 pt-4 border-t border-line leading-relaxed">
          {caption}
        </p>
      )}
    </section>
  );
}
