'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { apiFetch } from '../../../lib/api';

// Charts are client-only (ECharts). Reuse the exact same renderer the report uses.
const ChartContent = dynamic(() => import('../charts/ChartContent'), { ssr: false });

interface ReportMeta {
  is_public: boolean;
  title: string;
  description: string;
  og_image_url: string | null;
  owned: boolean;
}
interface LayoutEntry {
  chart_id: string;
  position: 'main' | 'sidebar';
  order: number;
}
interface ChartWithCaption {
  chart_id: string;
  spec: any;
  caption: string;
}
interface Report {
  charts: ChartWithCaption[];
  layout: LayoutEntry[];
}

// Headline chart = the first chart in the main column (lowest order); fall back to the
// first chart of any kind. This is the chart shown big on the social card.
function headlineChart(report: Report): ChartWithCaption | null {
  const mainOrder = [...report.layout]
    .filter((e) => e.position === 'main')
    .sort((a, b) => a.order - b.order);
  for (const e of mainOrder) {
    const c = report.charts.find((cc) => cc.chart_id === e.chart_id);
    if (c) return c;
  }
  return report.charts[0] ?? null;
}

export default function OgCard({ id }: { id: string }) {
  const [meta, setMeta] = useState<ReportMeta | null>(null);
  const [chart, setChart] = useState<ChartWithCaption | null>(null);
  const [status, setStatus] = useState<'loading' | 'blank' | 'ready'>('loading');

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // Gate on the public-visibility meta (never render a non-public report's data).
        const mres = await apiFetch(`/report/${id}/meta`);
        if (!mres.ok) throw new Error('meta');
        const m: ReportMeta = await mres.json();
        if (cancelled) return;
        if (!m.is_public) {
          setStatus('blank');
          return;
        }
        setMeta(m);
        const rres = await apiFetch(`/report/${id}`);
        if (!rres.ok) throw new Error('report');
        const r: Report = await rres.json();
        if (cancelled) return;
        setChart(headlineChart(r));
        setStatus('ready');
      } catch {
        if (!cancelled) setStatus('blank');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [id]);

  // Tell the server-side screenshotter when to capture: wait for ECharts to paint on the
  // ready path; flag fast on the blank path so the fallback card captures without hanging.
  useEffect(() => {
    if (status === 'loading') return;
    const t = setTimeout(
      () => document.body.setAttribute('data-charts-ready', 'true'),
      status === 'ready' ? 1600 : 200,
    );
    return () => clearTimeout(t);
  }, [status]);

  const title = meta?.title || 'ChartSage report';

  return (
    <div
      className="theme-light bg-canvas text-ink overflow-hidden flex flex-col"
      style={{ width: 1200, height: 630 }}
    >
      <div className="flex items-start justify-between px-14 pt-12 gap-8">
        <div className="min-w-0">
          <p className="eyebrow text-accent mb-2">ChartSage</p>
          <h1
            className="font-display text-[2.5rem] font-medium leading-tight"
            style={{
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              overflow: 'hidden',
            }}
          >
            {title}
          </h1>
        </div>
        <span className="inline-flex w-14 h-14 shrink-0 items-center justify-center rounded-xl bg-surface-2 border border-line-2">
          <svg viewBox="0 0 32 32" className="w-8 h-8" aria-hidden="true">
            <rect x="6.5" y="17" width="4.5" height="8.5" rx="1.4" fill="#5EEAD4" />
            <rect x="13.75" y="11.5" width="4.5" height="14" rx="1.4" fill="#2DD4BF" />
            <rect x="21" y="7" width="4.5" height="18.5" rx="1.4" fill="#0D9488" />
          </svg>
        </span>
      </div>

      <div className="flex-1 px-12 pt-5 pb-3 flex items-center">
        <div className="w-full card shadow-card rounded-2xl px-7 pt-5 pb-3">
          {chart ? (
            <ChartContent spec={chart.spec} />
          ) : (
            <p className="text-ink-2 text-base py-20 text-center">
              Turn any spreadsheet into a beautiful, interactive report — in seconds.
            </p>
          )}
        </div>
      </div>

      <div className="px-14 pb-7">
        <p className="text-ink-3 text-base font-medium">
          chartsage.app · the instant report, not the chart
        </p>
      </div>
    </div>
  );
}
