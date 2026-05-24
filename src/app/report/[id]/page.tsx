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
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-gray-900">Could not load report</h2>
          <p className="mt-2 text-gray-600">{error}</p>
          <a href="/" className="mt-4 inline-block px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
            Back to upload
          </a>
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50">
        <div className="animate-spin rounded-full h-10 w-10 border-4 border-blue-500 border-t-transparent mb-4" />
        <p className="text-gray-700">Loading report…</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-2 md:px-8 py-8 bg-gray-50 min-h-screen">
      <ReportSummary summary={report.summary} generatedAt={report.generated_at} />
      {report.data_quality && report.data_quality.length > 0 && (
        <DataQualityCallout notes={report.data_quality} />
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-8">
        {report.charts.map((c, idx) => (
          <ChartCard key={idx} spec={c.spec} caption={c.caption} />
        ))}
      </div>
    </div>
  );
}
