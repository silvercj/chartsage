'use client';

import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import ChartContent from './charts/ChartContent';
import { isWideChart } from './charts/isWideChart';

interface Props {
  index: number;
  spec: any;
  caption: string;
  chartId: string;
  onHide?: (chartId: string) => void;
  onToggleCollapse?: (chartId: string) => void;
  collapsed?: boolean;
  printMode?: boolean;   // hides interactive chrome (drag handle, kind badge) for PDF export
}

const KIND_LABEL: Record<string, string> = {
  bar: 'Bar', histogram: 'Histogram', scatter: 'Scatter',
  line: 'Trend', pie: 'Composition', box: 'Distribution', heatmap: 'Heatmap',
};

export default function ChartCard({ index, spec, caption, chartId, onHide, onToggleCollapse, collapsed, printMode }: Props) {
  const sortable = useSortable({ id: chartId });
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = sortable;

  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.4 : 1,
  };

  const kindLabel = KIND_LABEL[spec.kind] ?? spec.kind;
  const wide = isWideChart(spec);   // many-category bar chart → full-width + collapsible

  return (
    <section
      ref={setNodeRef}
      style={style}
      className={`card shadow-card rounded-2xl p-5 flex flex-col${wide && !collapsed ? ' lg:col-span-2' : ''}`}
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
          <h2 className="font-display text-lg text-ink leading-snug line-clamp-2">
            {spec.title}
          </h2>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          {!printMode && (
            <span className="font-mono text-xs uppercase tracking-wide text-ink-3 mt-1">
              {kindLabel}
            </span>
          )}
          {wide && !printMode && onToggleCollapse && (
            <button
              type="button"
              onClick={() => onToggleCollapse(chartId)}
              className="text-ink-3 hover:text-ink transition-colors w-6 h-6 flex items-center justify-center rounded hover:bg-surface-2 focus:outline-none focus:ring-2 focus:ring-accent"
              aria-label={collapsed ? 'Expand chart' : 'Collapse chart'}
              title={collapsed ? 'Expand to full ranking' : 'Collapse to top 12'}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                {collapsed ? (
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 9V4.5M9 9H4.5M9 9 3.75 3.75M9 15v4.5M9 15H4.5M9 15l-5.25 5.25M15 9h4.5M15 9V4.5M15 9l5.25-5.25M15 15h4.5M15 15v4.5m0-4.5 5.25 5.25" />
                )}
              </svg>
            </button>
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
        <ChartContent spec={spec} collapsed={collapsed} />
      </div>
      {caption && (
        <p className="font-display italic text-ink-2 text-sm mt-4 pt-4 border-t border-line leading-relaxed">
          {caption}
        </p>
      )}
    </section>
  );
}
