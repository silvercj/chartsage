'use client';
import { useState } from 'react';
import SidebarCard from './SidebarCard';
import type { ChartWithCaption } from './useReportLayout';

interface Props {
  charts: ChartWithCaption[];
  onPromote: (chartId: string) => void;
}

export default function Sidebar({ charts, onPromote }: Props) {
  const [open, setOpen] = useState(true);

  if (!open) {
    return (
      <aside className="w-10 shrink-0">
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="w-10 h-10 rounded-lg bg-white ring-1 ring-stone-200/80 flex items-center justify-center text-stone-500 hover:text-stone-900 hover:bg-stone-50"
          aria-label="Expand sidebar"
          title={`${charts.length} extra charts`}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M11 19l-7-7 7-7M19 19l-7-7 7-7" />
          </svg>
        </button>
      </aside>
    );
  }

  return (
    <aside className="w-72 shrink-0">
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="text-xs uppercase tracking-widest text-stone-400">
          More charts · {charts.length}
        </h3>
        <button
          type="button"
          onClick={() => setOpen(false)}
          className="text-stone-400 hover:text-stone-700 text-sm"
          aria-label="Collapse sidebar"
        >
          ›
        </button>
      </div>
      {charts.length === 0 ? (
        <p className="text-sm text-stone-400 italic">
          Drag a chart here or click ×, or hit "Generate 5 more" above.
        </p>
      ) : (
        <div>
          {charts.map((c) => (
            <SidebarCard
              key={c.chart_id}
              chartId={c.chart_id}
              title={c.spec.title}
              kind={c.spec.kind}
              onPromote={onPromote}
            />
          ))}
        </div>
      )}
    </aside>
  );
}
