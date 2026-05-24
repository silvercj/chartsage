'use client';
import { useSortable } from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

interface Props {
  chartId: string;
  title: string;
  kind: string;
  onPromote: (chartId: string) => void;
}

const KIND_LABEL: Record<string, string> = {
  bar: 'Bar', histogram: 'Histogram', scatter: 'Scatter',
  line: 'Trend', pie: 'Composition', box: 'Distribution', heatmap: 'Heatmap',
};

export default function SidebarCard({ chartId, title, kind, onPromote }: Props) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id: chartId });

  const style: React.CSSProperties = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.4 : 1,
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      className="bg-white rounded-xl ring-1 ring-stone-200/80 p-3 mb-2 flex items-start gap-2"
    >
      <button
        type="button"
        className="text-stone-300 hover:text-stone-600 cursor-grab active:cursor-grabbing px-0.5 leading-none focus:outline-none focus:ring-2 focus:ring-stone-400 rounded shrink-0"
        aria-label="Drag to reorder"
        {...attributes}
        {...listeners}
      >
        <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
          <circle cx="6" cy="5" r="1.5" /><circle cx="10" cy="5" r="1.5" /><circle cx="14" cy="5" r="1.5" />
          <circle cx="6" cy="10" r="1.5" /><circle cx="10" cy="10" r="1.5" /><circle cx="14" cy="10" r="1.5" />
          <circle cx="6" cy="15" r="1.5" /><circle cx="10" cy="15" r="1.5" /><circle cx="14" cy="15" r="1.5" />
        </svg>
      </button>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-stone-800 leading-snug truncate">{title}</p>
        <p className="text-[10px] uppercase tracking-widest text-stone-400 mt-0.5">
          {KIND_LABEL[kind] ?? kind}
        </p>
      </div>
      <button
        type="button"
        onClick={() => onPromote(chartId)}
        className="text-stone-400 hover:text-stone-900 transition-colors w-6 h-6 flex items-center justify-center rounded hover:bg-stone-100 shrink-0 focus:outline-none focus:ring-2 focus:ring-stone-400"
        aria-label="Move to main report"
        title="Move to main"
      >
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" d="M11 17l-5-5m0 0l5-5m-5 5h12" />
        </svg>
      </button>
    </div>
  );
}
