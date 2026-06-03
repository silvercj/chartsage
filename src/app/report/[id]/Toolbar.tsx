'use client';
import { useEffect, useState } from 'react';
import { apiFetch } from '../../lib/api';
import { posthog } from '../../lib/posthog';
import type { Report } from './useReportLayout';
import UpsellModal from '../../components/UpsellModal';
import OutOfCreditsModal from '../../components/OutOfCreditsModal';
import AddChartModal from './AddChartModal';
import ShareModal from './ShareModal';
import { useCredits } from '../../lib/useCredits';
import { GENERATE_MORE_COST, DEEP_ANALYSIS_COST } from '../../lib/credits';

interface Props {
  sessionId: string;
  report: Report;
  onReportUpdated: (next: Report) => void;
}

export default function Toolbar({ sessionId, report, onReportUpdated }: Props) {
  const [generating, setGenerating] = useState(false);
  const [deepening, setDeepening] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [showUpsell, setShowUpsell] = useState(false);
  const [showOutOfCredits, setShowOutOfCredits] = useState(false);
  const [showAddChart, setShowAddChart] = useState(false);
  const [owned, setOwned] = useState(false);
  const [isPublic, setIsPublic] = useState(false);
  const [showShare, setShowShare] = useState(false);
  const { refetch } = useCredits();

  useEffect(() => {
    apiFetch(`/report/${sessionId}/meta`)
      .then((r) => (r.ok ? r.json() : null))
      .then((m) => {
        if (m) {
          setOwned(!!m.owned);
          setIsPublic(!!m.is_public);
        }
      })
      .catch(() => {});
  }, [sessionId]);

  async function handleGenerateMore() {
    setGenerating(true);
    setError(null);
    setNotice(null);
    posthog.capture?.('generate_more_clicked', { reportId: sessionId });
    try {
      const res = await apiFetch(`/report/${sessionId}/generate-more`, { method: 'POST' });
      if (res.status === 503) {
        setError('The AI is busy. Try again in 30 seconds.');
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

  async function handleDeepen() {
    setDeepening(true);
    setError(null);
    setNotice(null);
    posthog.capture?.('deepen_clicked', { reportId: sessionId });
    try {
      const res = await apiFetch(`/report/${sessionId}/deepen`, { method: 'POST' });
      if (res.status === 503) {
        setError('The AI is busy. Try again in 30 seconds.');
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
      const before = report.charts?.length ?? 0;
      const updated: Report = await res.json();
      const added = (updated.charts?.length ?? 0) - before;
      onReportUpdated(updated);
      refetch();   // balance changed (only when charts were actually added)
      // Deepen only adds charts; when it finds nothing new the backend returns the
      // report unchanged AND skips the charge, so make that explicit instead of
      // silently swapping in an identical report.
      setNotice(
        added > 0
          ? `Deepened — added ${added} new chart${added === 1 ? '' : 's'}.`
          : "No new angles found — your report already covers the data well. You weren't charged.",
      );
      setTimeout(() => setNotice(null), 8000);
    } catch (e: any) {
      setError(e.message || 'Failed to deepen this report.');
    } finally {
      setDeepening(false);
    }
  }

  function handleExport(ext: string) {
    setExporting(true);
    setError(null);
    posthog.capture?.('export_clicked', { reportId: sessionId, format: ext });
    apiFetch(`/report/${sessionId}/export.${ext}`)
      .then(async (r) => {
        if (!r.ok) throw new Error(`Export failed (${r.status})`);
        const blob = await r.blob();
        const blobUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = blobUrl;
        a.download = `chartsage-${sessionId.slice(0, 8)}.${ext}`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(blobUrl);
      })
      .catch((e) => setError(e.message || 'Export failed.'))
      .finally(() => setExporting(false));
  }

  const EXPORT_FORMATS: { ext: string; label: string }[] = [
    { ext: 'pdf', label: 'PDF' },
    { ext: 'pptx', label: 'PowerPoint' },
    { ext: 'xlsx', label: 'Excel' },
    { ext: 'zip', label: 'Images' },
    { ext: 'md', label: 'Markdown' },
  ];

  return (
    <>
      <div className="sticky top-0 z-10 -mx-4 sm:-mx-6 lg:-mx-8 px-4 sm:px-6 lg:px-8 py-3 mb-6 bg-canvas/90 backdrop-blur border-b border-line flex items-center justify-end gap-3">
        {error && <span className="text-sm text-ember mr-auto">{error}</span>}
        {!error && notice && <span className="text-sm text-ink-2 mr-auto">{notice}</span>}
        <button
          type="button"
          onClick={() => setShowAddChart(true)}
          className="btn btn-ghost"
        >
          + Add a chart
        </button>
        {!report.metadata?.deep && (
          <button
            type="button"
            onClick={handleDeepen}
            disabled={deepening}
            className="btn btn-ghost"
          >
            {deepening ? (
              <span className="inline-flex items-center gap-2">
                <span className="h-3.5 w-3.5 rounded-full border-2 border-line-2 border-t-accent animate-spin" />
                Deepening…
              </span>
            ) : (
              `Deepen this report · ${DEEP_ANALYSIS_COST}`
            )}
          </button>
        )}
        <button
          type="button"
          onClick={handleGenerateMore}
          disabled={generating}
          className="btn btn-ghost"
        >
          {generating ? (
            <span className="inline-flex items-center gap-2">
              <span className="h-3.5 w-3.5 rounded-full border-2 border-line-2 border-t-accent animate-spin" />
              Generating…
            </span>
          ) : (
            `Generate 5 more · ${GENERATE_MORE_COST}`
          )}
        </button>
        {owned && (
          <button
            type="button"
            onClick={() => setShowShare(true)}
            className="btn btn-ghost"
          >
            Share
          </button>
        )}
        <details className="relative group">
          <summary
            className="btn btn-primary list-none cursor-pointer select-none [&::-webkit-details-marker]:hidden aria-disabled:opacity-50 aria-disabled:cursor-not-allowed"
            aria-disabled={exporting}
            onClick={(e) => { if (exporting) e.preventDefault(); }}
          >
            {exporting ? (
              <span className="inline-flex items-center gap-2">
                <span className="h-3.5 w-3.5 rounded-full border-2 border-on-accent/40 border-t-on-accent animate-spin" />
                Preparing…
              </span>
            ) : (
              <>Export <span aria-hidden className="text-xs opacity-80">▾</span></>
            )}
          </summary>
          <div className="absolute right-0 mt-2 w-44 card p-1 shadow-glow z-20">
            {EXPORT_FORMATS.map((f) => (
              <button
                key={f.ext}
                type="button"
                disabled={exporting}
                onClick={(e) => {
                  (e.currentTarget.closest('details') as HTMLDetailsElement | null)?.removeAttribute('open');
                  handleExport(f.ext);
                }}
                className="w-full text-left rounded-lg px-3 py-2 text-sm text-ink hover:bg-surface-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {f.label}
              </button>
            ))}
          </div>
        </details>
      </div>
      <AddChartModal
        open={showAddChart}
        onClose={() => setShowAddChart(false)}
        sessionId={sessionId}
        onReportUpdated={(next) => { onReportUpdated(next); refetch(); }}
        onOutOfCredits={(code) => {
          if (code === 'OUT_OF_CREDITS') setShowOutOfCredits(true);
          else setShowUpsell(true);   // UPGRADE_REQUIRED (anon)
        }}
      />
      <UpsellModal open={showUpsell} onClose={() => setShowUpsell(false)} />
      <OutOfCreditsModal open={showOutOfCredits} onClose={() => setShowOutOfCredits(false)} />
      <ShareModal
        key={`share-${isPublic}`}
        open={showShare}
        onClose={() => setShowShare(false)}
        sessionId={sessionId}
        initialIsPublic={isPublic}
      />
    </>
  );
}
