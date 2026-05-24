'use client';
import { useState } from 'react';
import type { Report } from './useReportLayout';

interface Props {
  sessionId: string;
  onReportUpdated: (next: Report) => void;
}

export default function Toolbar({ sessionId, onReportUpdated }: Props) {
  const [generating, setGenerating] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleGenerateMore() {
    setGenerating(true);
    setError(null);
    try {
      const res = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/report/${sessionId}/generate-more`,
        { method: 'POST' },
      );
      if (res.status === 503) {
        setError('Claude is busy. Try again in 30 seconds.');
        return;
      }
      if (!res.ok) throw new Error(`Failed (${res.status})`);
      const updated: Report = await res.json();
      onReportUpdated(updated);
    } catch (e: any) {
      setError(e.message || 'Failed to generate more charts.');
    } finally {
      setGenerating(false);
    }
  }

  function handleExportPdf() {
    setExporting(true);
    setError(null);
    const url = `${process.env.NEXT_PUBLIC_API_URL}/report/${sessionId}/export.pdf`;
    window.open(url, '_blank');
    // We can't reliably detect download completion, so reset the spinner after a short delay
    setTimeout(() => setExporting(false), 1500);
  }

  return (
    <div className="sticky top-0 z-10 -mx-4 sm:-mx-6 lg:-mx-8 px-4 sm:px-6 lg:px-8 py-3 mb-6 bg-stone-50/90 backdrop-blur border-b border-stone-200 flex items-center justify-end gap-3">
      {error && (
        <span className="text-sm text-red-600 mr-auto">{error}</span>
      )}
      <button
        type="button"
        onClick={handleGenerateMore}
        disabled={generating}
        className="px-4 py-2 text-sm font-medium text-stone-700 bg-white ring-1 ring-stone-200 rounded-lg hover:bg-stone-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {generating ? 'Generating…' : 'Generate 5 more'}
      </button>
      <button
        type="button"
        onClick={handleExportPdf}
        disabled={exporting}
        className="px-4 py-2 text-sm font-medium text-white bg-stone-900 rounded-lg hover:bg-stone-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {exporting ? 'Preparing PDF…' : 'Export PDF'}
      </button>
    </div>
  );
}
