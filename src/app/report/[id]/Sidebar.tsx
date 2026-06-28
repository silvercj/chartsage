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
          className="w-10 h-10 rounded-lg bg-surface border border-line flex items-center justify-center text-ink-2 hover:text-ink hover:bg-surface-2 transition-colors"
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
    <aside className="w-full lg:w-72 shrink-0">
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="font-mono text-xs uppercase tracking-wide text-ink-3">
          More charts · {charts.length}
        </h3>
        <button
          type="button"
          onClick={() => setOpen(false)}
          className="text-ink-3 hover:text-ink text-sm transition-colors"
          aria-label="Collapse sidebar"
        >
          ›
        </button>
      </div>
      {charts.length === 0 ? (
        <p className="text-sm text-ink-3 italic">
          Drag a chart here or click ×, or hit "Generate 5 more" above.
        </p>
      ) : (
        <div>
          {charts.map((c) => (
            <SidebarCard
              key={c.chart_id}
              chartId={c.chart_id}
              spec={c.spec}
              onPromote={onPromote}
            />
          ))}
        </div>
      )}
    </aside>
  );
}
