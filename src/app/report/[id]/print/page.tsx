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

const CHARTS_PER_PAGE = 2;

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

  // Group charts into pages of CHARTS_PER_PAGE
  const pages: ChartWithCaption[][] = [];
  for (let i = 0; i < mainCharts.length; i += CHARTS_PER_PAGE) {
    pages.push(mainCharts.slice(i, i + CHARTS_PER_PAGE));
  }

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
        .print-container { font-family: Inter, system-ui, sans-serif; }
      `}</style>
      <div className="print-container max-w-[700px] mx-auto p-8">
        {/* Cover page: title + summary + data quality */}
        <header className="mb-8">
          <p className="text-xs uppercase tracking-widest text-stone-400 mb-2">ChartSage Report</p>
          <h1 className="text-4xl font-semibold tracking-tight text-stone-900 mb-3">Insights</h1>
          <p className="text-xs text-stone-500">
            Generated {new Date(report.generated_at).toLocaleDateString(undefined, {
              year: 'numeric', month: 'long', day: 'numeric',
            })}
          </p>
        </header>

        <article className="prose prose-sm max-w-none text-stone-700 leading-relaxed mb-8">
          {report.summary.split(/\n\s*\n/).filter(Boolean).map((p, i) => (
            <p key={i}>{p}</p>
          ))}
        </article>

        {report.data_quality && report.data_quality.length > 0 && (
          <DataQualityCallout notes={report.data_quality} />
        )}

        <div className="print-page-break" />

        {/* Charts: CHARTS_PER_PAGE per page */}
        {pages.map((pageCharts, pageIdx) => (
          <div key={pageIdx} className={pageIdx < pages.length - 1 ? 'print-page-break' : ''}>
            {pageCharts.map((c, i) => (
              <div key={c.chart_id} className="mb-6">
                <ChartCard
                  chartId={c.chart_id}
                  index={pageIdx * CHARTS_PER_PAGE + i + 1}
                  spec={c.spec}
                  caption={c.caption}
                />
              </div>
            ))}
          </div>
        ))}

        <footer className="mt-12 pt-4 border-t border-stone-200 text-xs text-stone-400 text-center">
          ChartSage · Report {params.id.slice(0, 8)}
        </footer>
      </div>
    </>
  );
}
