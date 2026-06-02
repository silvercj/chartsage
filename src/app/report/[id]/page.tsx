'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import {
  DndContext,
  closestCenter,
  DragEndEvent,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
} from '@dnd-kit/core';
import {
  SortableContext,
  sortableKeyboardCoordinates,
  rectSortingStrategy,
} from '@dnd-kit/sortable';
import { useReportLayout, type Report } from './useReportLayout';
import { apiFetch } from '../../lib/api';
import ReportFeedback from './ReportFeedback';

const ChartCard = dynamic(() => import('./ChartCard'), { ssr: false });
const KpiTiles = dynamic(() => import('./KpiTiles'));
const ReportSummary = dynamic(() => import('./ReportSummary'));
const DataQualityCallout = dynamic(() => import('./DataQualityCallout'));
const Sidebar = dynamic(() => import('./Sidebar'), { ssr: false });
const Toolbar = dynamic(() => import('./Toolbar'), { ssr: false });

function Loading() {
  return (
    <div className="theme-light flex flex-col items-center justify-center min-h-screen bg-canvas text-ink">
      <div className="animate-spin rounded-full h-9 w-9 border-2 border-line-2 border-t-accent mb-4" />
      <p className="text-ink-2 text-sm">Loading report…</p>
    </div>
  );
}

function ErrorView({ message }: { message: string }) {
  return (
    <div className="theme-light flex flex-col items-center justify-center min-h-screen bg-canvas text-ink">
      <div className="text-center">
        <h2 className="font-display text-2xl font-medium text-ink">Could not load report</h2>
        <p className="mt-2 text-ink-2">{message}</p>
        <a href="/app" className="btn btn-primary mt-6">
          Back to upload
        </a>
      </div>
    </div>
  );
}

export default function ReportPage({ params }: { params: { id: string } }) {
  const [initial, setInitial] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiFetch(`/report/${params.id}`)
      .then(async (r) => {
        if (r.status === 404) throw new Error('This report has expired. Generate a new one.');
        if (!r.ok) throw new Error('Failed to load report');
        return r.json();
      })
      .then(setInitial)
      .catch((e) => setError(e.message));
  }, [params.id]);

  if (error) return <ErrorView message={error} />;
  if (!initial) return <Loading />;

  return <ReportView sessionId={params.id} initialReport={initial} />;
}

function ReportView({ sessionId, initialReport }: { sessionId: string; initialReport: Report }) {
  const { report, mainCharts, sidebarCharts, reorder, move, replaceReport, saveError } =
    useReportLayout(initialReport, sessionId);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }),
  );

  function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event;
    if (!over) return;
    if (active.id === over.id) return;

    const activeId = String(active.id);
    const overId = String(over.id);

    const activeEntry = report.layout.find((e) => e.chart_id === activeId);
    const overEntry = report.layout.find((e) => e.chart_id === overId);
    if (!activeEntry || !overEntry) return;

    if (activeEntry.position === overEntry.position) {
      // Reorder within the same position
      const siblings = report.layout
        .filter((e) => e.position === activeEntry.position)
        .sort((a, b) => a.order - b.order)
        .map((e) => e.chart_id);
      const fromIdx = siblings.indexOf(activeId);
      const toIdx = siblings.indexOf(overId);
      if (fromIdx === -1 || toIdx === -1) return;
      reorder(activeId, toIdx);
    } else {
      // Cross-position: move to the other side
      move(activeId, overEntry.position);
    }
  }

  return (
    <div className="theme-light bg-canvas text-ink min-h-screen">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Toolbar sessionId={sessionId} report={report} onReportUpdated={replaceReport} />

        <ReportSummary summary={report.summary} generatedAt={report.generated_at} />
        {report.data_quality && report.data_quality.length > 0 && (
          <DataQualityCallout notes={report.data_quality} />
        )}

        <KpiTiles metrics={report.key_metrics} />

        {saveError && (
          <div className="mt-4 p-3 bg-surface-2 border border-line text-ember text-sm rounded-lg">
            {saveError}
          </div>
        )}

        <DndContext
          sensors={sensors}
          collisionDetection={closestCenter}
          onDragEnd={handleDragEnd}
        >
          <div className="flex gap-6 mt-10">
            <main className="flex-1 min-w-0">
              <SortableContext
                items={mainCharts.map((c) => c.chart_id)}
                strategy={rectSortingStrategy}
              >
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {mainCharts.map((c, idx) => (
                    <ChartCard
                      key={c.chart_id}
                      chartId={c.chart_id}
                      index={idx + 1}
                      spec={c.spec}
                      caption={c.caption}
                      onHide={(id) => move(id, 'sidebar')}
                    />
                  ))}
                </div>
              </SortableContext>
            </main>
            <SortableContext
              items={sidebarCharts.map((c) => c.chart_id)}
              strategy={rectSortingStrategy}
            >
              <Sidebar
                charts={sidebarCharts}
                onPromote={(id) => move(id, 'main')}
              />
            </SortableContext>
          </div>
        </DndContext>

        <ReportFeedback reportId={sessionId} />

        <footer className="mt-16 pt-6 border-t border-line font-mono text-xs text-ink-3 flex justify-between">
          <span>Report id: {sessionId.slice(0, 8)}</span>
          <a href="/app" className="hover:text-ink-2 transition-colors">New report →</a>
        </footer>
      </div>
    </div>
  );
}
