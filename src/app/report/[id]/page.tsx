'use client';

import { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';

const ChartCard = dynamic(() => import('./ChartCard'), { ssr: false });
const ReportSummary = dynamic(() => import('./ReportSummary'));
const DataQualityCallout = dynamic(() => import('./DataQualityCallout'));

interface ChartWithCaption {
  spec: any;
  caption: string;
}

interface Report {
  generated_at: string;
  summary: string;
  data_quality: string[];
  charts: ChartWithCaption[];
  metadata: Record<string, any>;
}

export default function ReportPage({ params }: { params: { id: string } }) {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/report/${params.id}`)
      .then(async (r) => {
        if (r.status === 404) throw new Error('This report has expired. Generate a new one.');
        if (!r.ok) throw new Error('Failed to load report');
        return r.json();
      })
      .then(setReport)
      .catch((e) => setError(e.message));
  }, [params.id]);

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-stone-50">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-stone-900">Could not load report</h2>
          <p className="mt-2 text-stone-600">{error}</p>
          <a href="/" className="mt-6 inline-block px-5 py-2.5 bg-stone-900 text-white text-sm rounded-lg hover:bg-stone-800 transition-colors">
            Back to upload
          </a>
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-stone-50">
        <div className="animate-spin rounded-full h-9 w-9 border-2 border-stone-300 border-t-stone-900 mb-4" />
        <p className="text-stone-600 text-sm">Loading report…</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-stone-50">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <ReportSummary summary={report.summary} generatedAt={report.generated_at} />
        {report.data_quality && report.data_quality.length > 0 && (
          <DataQualityCallout notes={report.data_quality} />
        )}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-10">
          {report.charts.map((c, idx) => (
            <ChartCard key={idx} index={idx + 1} spec={c.spec} caption={c.caption} />
          ))}
        </div>
        <footer className="mt-16 pt-6 border-t border-stone-200 text-xs text-stone-400 flex justify-between">
          <span>Report id: {params.id.slice(0, 8)}</span>
          <a href="/" className="hover:text-stone-600">New report →</a>
        </footer>
      </div>
    </div>
  );
}
