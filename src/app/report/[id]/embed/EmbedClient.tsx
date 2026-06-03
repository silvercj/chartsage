'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { apiFetch } from '../../../lib/api';
import type { Report, ChartWithCaption } from '../useReportLayout';

// Reuse the exact same chart components the interactive report uses.
// ChartCard renders ECharts, so it must be client-only (ssr:false), matching page.tsx.
const ChartCard = dynamic(() => import('../ChartCard'), { ssr: false });
const KpiTiles = dynamic(() => import('../KpiTiles'));
const ReportSummary = dynamic(() => import('../ReportSummary'));

interface ReportMeta {
  is_public: boolean;
  title: string;
  description: string;
  og_image_url: string | null;
  owned: boolean;
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div className="theme-light bg-canvas text-ink-2 min-h-screen flex items-center justify-center p-8 text-sm">
      {children}
    </div>
  );
}

export default function EmbedClient({ id }: { id: string }) {
  const [meta, setMeta] = useState<ReportMeta | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [status, setStatus] = useState<'loading' | 'private' | 'ready'>('loading');

  useEffect(() => {
    let cancelled = false;

    (async () => {
      // 1. Gate on the lightweight meta endpoint. If it fails or the report is
      //    not public, render a placeholder and fetch NO report data.
      let m: ReportMeta;
      try {
        const res = await apiFetch(`/report/${id}/meta`);
        if (!res.ok) throw new Error('meta');
        m = await res.json();
      } catch {
        if (!cancelled) setStatus('private');
        return;
      }
      if (cancelled) return;
      if (!m.is_public) {
        setMeta(m);
        setStatus('private');
        return;
      }
      setMeta(m);

      // 2. Public → fetch the full report JSON.
      try {
        const res = await apiFetch(`/report/${id}`);
        if (!res.ok) throw new Error('report');
        const data: Report = await res.json();
        if (cancelled) return;
        setReport(data);
        setStatus('ready');
      } catch {
        if (!cancelled) setStatus('private');
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [id]);

  if (status === 'loading') {
    return <Centered>Loading…</Centered>;
  }

  if (status === 'private' || !report) {
    return <Centered>This report isn&rsquo;t public.</Centered>;
  }

  // Order charts the same way the interactive report does: main column first,
  // then sidebar, each in layout order. Embed is a single vertical stack.
  const orderedCharts = (['main', 'sidebar'] as const).flatMap((position) =>
    report.layout
      .filter((e) => e.position === position)
      .sort((a, b) => a.order - b.order)
      .map((e) => report.charts.find((c) => c.chart_id === e.chart_id))
      .filter((c): c is ChartWithCaption => !!c),
  );

  return (
    <div className="theme-light bg-canvas text-ink min-h-screen">
      <div className="max-w-3xl mx-auto px-6 py-8">
        {report.summary && (
          <ReportSummary summary={report.summary} generatedAt={report.generated_at} />
        )}

        <KpiTiles metrics={report.key_metrics} />

        <div className="flex flex-col gap-6">
          {orderedCharts.map((c, idx) => (
            <ChartCard
              key={c.chart_id}
              chartId={c.chart_id}
              index={idx + 1}
              spec={c.spec}
              caption={c.caption}
              printMode
            />
          ))}
        </div>
      </div>
    </div>
  );
}
