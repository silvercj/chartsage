'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { apiFetch } from '../../lib/api';

export interface ChartLayoutEntry {
  chart_id: string;
  position: 'main' | 'sidebar';
  order: number;
}

export interface ChartWithCaption {
  chart_id: string;
  spec: any;
  caption: string;
}

export interface Report {
  generated_at: string;
  summary: string;
  data_quality: string[];
  charts: ChartWithCaption[];
  layout: ChartLayoutEntry[];
  metadata: Record<string, any>;
}

const PATCH_DEBOUNCE_MS = 500;

export function useReportLayout(initial: Report, sessionId: string) {
  const [report, setReport] = useState<Report>(initial);
  const [saveError, setSaveError] = useState<string | null>(null);
  const patchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastSavedLayoutRef = useRef<ChartLayoutEntry[]>(initial.layout);
  const patchDisabledRef = useRef<boolean>(false);

  const queuePatch = useCallback((nextLayout: ChartLayoutEntry[]) => {
    if (patchDisabledRef.current) return;   // permanent stop after a 5xx
    if (patchTimerRef.current) clearTimeout(patchTimerRef.current);
    patchTimerRef.current = setTimeout(async () => {
      try {
        const res = await apiFetch(`/report/${sessionId}/layout`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(nextLayout),
        });
        if (!res.ok) {
          // 5xx → backend is unhealthy. Stop trying for this session.
          if (res.status >= 500) {
            patchDisabledRef.current = true;
          }
          throw new Error(`Save failed (${res.status})`);
        }
        lastSavedLayoutRef.current = nextLayout;
        setSaveError(null);
      } catch (e: any) {
        const suffix = patchDisabledRef.current
          ? ' — your changes are local only.'
          : '';
        setSaveError((e.message || 'Could not save layout') + suffix);
        // Revert to last saved state
        setReport((r) => ({ ...r, layout: lastSavedLayoutRef.current }));
      }
    }, PATCH_DEBOUNCE_MS);
  }, [sessionId]);

  useEffect(() => () => {
    if (patchTimerRef.current) clearTimeout(patchTimerRef.current);
  }, []);

  /** Reorder a chart within its current position to a new index. */
  const reorder = useCallback((chartId: string, newIndex: number) => {
    setReport((r) => {
      const entry = r.layout.find((e) => e.chart_id === chartId);
      if (!entry) return r;
      const others = r.layout
        .filter((e) => e.position === entry.position && e.chart_id !== chartId)
        .sort((a, b) => a.order - b.order);
      others.splice(newIndex, 0, entry);
      const reordered = others.map((e, i) => ({ ...e, order: i }));
      const next: ChartLayoutEntry[] = [
        ...r.layout.filter((e) => e.position !== entry.position),
        ...reordered,
      ];
      queuePatch(next);
      return { ...r, layout: next };
    });
  }, [queuePatch]);

  /** Move a chart to a different position (main <-> sidebar). */
  const move = useCallback((chartId: string, newPosition: 'main' | 'sidebar') => {
    setReport((r) => {
      const entry = r.layout.find((e) => e.chart_id === chartId);
      if (!entry || entry.position === newPosition) return r;

      // Remove from old position and compact
      const oldSiblings = r.layout
        .filter((e) => e.position === entry.position && e.chart_id !== chartId)
        .sort((a, b) => a.order - b.order)
        .map((e, i) => ({ ...e, order: i }));

      // Append to new position at the end
      const newSiblings = r.layout
        .filter((e) => e.position === newPosition)
        .sort((a, b) => a.order - b.order);
      const moved: ChartLayoutEntry = {
        ...entry,
        position: newPosition,
        order: newSiblings.length,
      };
      const updatedNew = [...newSiblings, moved];

      const next = [...oldSiblings, ...updatedNew];
      queuePatch(next);
      return { ...r, layout: next };
    });
  }, [queuePatch]);

  /** Replace the report wholesale (used after generate-more). */
  const replaceReport = useCallback((newReport: Report) => {
    if (patchTimerRef.current) clearTimeout(patchTimerRef.current);
    lastSavedLayoutRef.current = newReport.layout;
    patchDisabledRef.current = false;   // backend roundtripped, try saves again
    setSaveError(null);
    setReport(newReport);
  }, []);

  // Partition for the UI
  const mainCharts = report.layout
    .filter((e) => e.position === 'main')
    .sort((a, b) => a.order - b.order)
    .map((e) => report.charts.find((c) => c.chart_id === e.chart_id))
    .filter((c): c is ChartWithCaption => !!c);

  const sidebarCharts = report.layout
    .filter((e) => e.position === 'sidebar')
    .sort((a, b) => a.order - b.order)
    .map((e) => report.charts.find((c) => c.chart_id === e.chart_id))
    .filter((c): c is ChartWithCaption => !!c);

  return {
    report,
    mainCharts,
    sidebarCharts,
    reorder,
    move,
    replaceReport,
    saveError,
  };
}
