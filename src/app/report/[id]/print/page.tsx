'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';

const ChartCard = dynamic(() => import('../ChartCard'), { ssr: false });
const ReportSummary = dynamic(() => import('../ReportSummary'));
const DataQualityCallout = dynamic(() => import('../DataQualityCallout'));

interface ChartLayoutEntry {
  chart_id: string;
  position: 'main' | 'sidebar';
  order: number;
}
interface ChartWithCaption { chart_id: string; spec: any; caption: string; }
interface Report {
  generated_at: string;
  summary: string;
  data_quality: string[];
  charts: ChartWithCaption[];
  layout: ChartLayoutEntry[];
  metadata: Record<string, any>;
}

export default function PrintReportPage({ params }: { params: { id: string } }) {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/report/${params.id}`)
      .then(async (r) => {
        if (!r.ok) throw new Error('Failed to load report');
        return r.json();
      })
      .then(setReport)
      .catch((e) => setError(e.message));
  }, [params.id]);

  useEffect(() => {
    if (!report) return;
    // Signal readiness after the charts have had time to render.
    // ECharts mounts asynchronously; this rAF + small timeout is enough.
    const t = setTimeout(() => {
      document.body.setAttribute('data-charts-ready', 'true');
    }, 1500);
    return () => clearTimeout(t);
  }, [report]);

  if (error) return <p style={{ padding: 40 }}>Error: {error}</p>;
  if (!report) return <p style={{ padding: 40 }}>Loading…</p>;

  const mainCharts = report.layout
    .filter((e) => e.position === 'main')
    .sort((a, b) => a.order - b.order)
    .map((e) => report.charts.find((c) => c.chart_id === e.chart_id))
    .filter((c): c is ChartWithCaption => !!c);

  return (
    <>
      <style jsx global>{`
        @media print {
          @page { size: A4; margin: 50px 40px; }
          body { background: white !important; }
          .print-page-break { page-break-after: always; }
          .no-print { display: none !important; }
        }
        body { background: white; }
        .print-container { font-family: var(--font-geist-sans), system-ui, sans-serif; }
        /* keep each chart card whole on one page; let cards flow & pack naturally */
        .avoid-break { break-inside: avoid; page-break-inside: avoid; }
        .report-footer { break-before: avoid; page-break-before: avoid; }
      `}</style>
      <div className="theme-light bg-white text-ink print-container max-w-[700px] mx-auto p-8">
        {/* Cover page: title + summary + data quality */}
        <header className="mb-8">
          <p className="font-mono text-xs uppercase tracking-widest text-ink-3 mb-2">ChartSage Report</p>
          <h1 className="font-display text-4xl font-semibold tracking-tight text-ink mb-3">Insights</h1>
          <p className="font-mono text-xs text-ink-2">
            Generated {new Date(report.generated_at).toLocaleDateString(undefined, {
              year: 'numeric', month: 'long', day: 'numeric',
            })}
          </p>
        </header>

        <article className="prose prose-sm max-w-none text-ink-2 leading-relaxed mb-8">
          {report.summary.split(/\n\s*\n/).filter(Boolean).map((p, i) => (
            <p key={i}>{p}</p>
          ))}
        </article>

        {report.data_quality && report.data_quality.length > 0 && (
          <DataQualityCallout notes={report.data_quality} />
        )}

        <div className="print-page-break" />

        {/* Charts flow naturally; each card stays whole on a page (avoid-break),
            so ~2 pack per page and the footer trails the last one without orphaning. */}
        {mainCharts.map((c, i) => (
          <div key={c.chart_id} className="avoid-break mb-6">
            <ChartCard
              chartId={c.chart_id}
              index={i + 1}
              spec={c.spec}
              caption={c.caption}
              printMode
            />
          </div>
        ))}

        <footer className="report-footer mt-8 pt-4 border-t border-line font-mono text-xs text-ink-3 text-center">
          ChartSage · Report {params.id.slice(0, 8)}
        </footer>
      </div>
    </>
  );
}
