'use client';
import { useState } from 'react';
import { apiFetch } from '../../lib/api';
import { posthog } from '../../lib/posthog';
import type { Report } from './useReportLayout';
import UpsellModal from '../../components/UpsellModal';
import OutOfCreditsModal from '../../components/OutOfCreditsModal';
import { useCredits } from '../../lib/useCredits';
import { GENERATE_MORE_COST } from '../../lib/credits';

interface Props {
  sessionId: string;
  onReportUpdated: (next: Report) => void;
}

export default function Toolbar({ sessionId, onReportUpdated }: Props) {
  const [generating, setGenerating] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showUpsell, setShowUpsell] = useState(false);
  const [showOutOfCredits, setShowOutOfCredits] = useState(false);
  const { refetch } = useCredits();

  async function handleGenerateMore() {
    setGenerating(true);
    setError(null);
    posthog.capture?.('generate_more_clicked', { reportId: sessionId });
    try {
      const res = await apiFetch(`/report/${sessionId}/generate-more`, { method: 'POST' });
      if (res.status === 503) {
        setError('Claude is busy. Try again in 30 seconds.');
        return;
      }
      if (res.status === 402) {
        let body: any = null;
        try { body = await res.json(); } catch {}
        if (body?.detail?.code === 'OUT_OF_CREDITS') setShowOutOfCredits(true);
        else setShowUpsell(true);   // UPGRADE_REQUIRED (anon)
        return;
      }
      if (!res.ok) throw new Error(`Failed (${res.status})`);
      const updated: Report = await res.json();
      onReportUpdated(updated);
      refetch();   // balance changed
    } catch (e: any) {
      setError(e.message || 'Failed to generate more charts.');
    } finally {
      setGenerating(false);
    }
  }

  function handleExportPdf() {
    setExporting(true);
    setError(null);
    posthog.capture?.('export_pdf_clicked', { reportId: sessionId });
    apiFetch(`/report/${sessionId}/export.pdf`)
      .then(async (r) => {
        if (!r.ok) throw new Error(`Export failed (${r.status})`);
        const blob = await r.blob();
        const blobUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = blobUrl;
        a.download = `chartsage-${sessionId.slice(0, 8)}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(blobUrl);
      })
      .catch((e) => setError(e.message || 'Export failed.'))
      .finally(() => setExporting(false));
  }

  return (
    <>
      <div className="sticky top-0 z-10 -mx-4 sm:-mx-6 lg:-mx-8 px-4 sm:px-6 lg:px-8 py-3 mb-6 bg-stone-50/90 backdrop-blur border-b border-stone-200 flex items-center justify-end gap-3">
        {error && <span className="text-sm text-red-600 mr-auto">{error}</span>}
        <button
          type="button"
          onClick={handleGenerateMore}
          disabled={generating}
          className="px-4 py-2 text-sm font-medium text-stone-700 bg-white ring-1 ring-stone-200 rounded-lg hover:bg-stone-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {generating ? 'Generating…' : `Generate 5 more · ${GENERATE_MORE_COST}`}
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
      <UpsellModal open={showUpsell} onClose={() => setShowUpsell(false)} />
      <OutOfCreditsModal open={showOutOfCredits} onClose={() => setShowOutOfCredits(false)} />
    </>
  );
}
