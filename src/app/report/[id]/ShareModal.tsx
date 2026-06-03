'use client';
import { useState } from 'react';
import { apiFetch } from '../../lib/api';
import { posthog } from '../../lib/posthog';

interface Props {
  open: boolean;
  onClose: () => void;
  sessionId: string;
  initialIsPublic: boolean;
}

export default function ShareModal({ open, onClose, sessionId, initialIsPublic }: Props) {
  const [isPublic, setIsPublic] = useState(initialIsPublic);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<'link' | 'embed' | null>(null);

  if (!open) return null;

  const publicUrl = typeof window !== 'undefined' ? `${window.location.origin}/report/${sessionId}` : '';
  const embedUrl = `${publicUrl}/embed`;
  const embedSnippet = `<iframe src="${embedUrl}" width="100%" height="600" style="border:0" loading="lazy"></iframe>`;

  async function publish() {
    setBusy(true);
    setError(null);
    try {
      const res = await apiFetch(`/report/${sessionId}/publish`, { method: 'POST' });
      if (!res.ok) throw new Error(`Failed (${res.status})`);
      posthog.capture?.('report_published', { reportId: sessionId });
      setIsPublic(true);
    } catch (e: any) {
      setError(e.message || 'Failed to publish this report.');
    } finally {
      setBusy(false);
    }
  }

  async function unpublish() {
    setBusy(true);
    setError(null);
    try {
      const res = await apiFetch(`/report/${sessionId}/unpublish`, { method: 'POST' });
      if (!res.ok) throw new Error(`Failed (${res.status})`);
      posthog.capture?.('report_unpublished', { reportId: sessionId });
      setIsPublic(false);
    } catch (e: any) {
      setError(e.message || 'Failed to make this report private.');
    } finally {
      setBusy(false);
    }
  }

  function copy(kind: 'link' | 'embed', text: string) {
    navigator.clipboard?.writeText(text).then(
      () => {
        setCopied(kind);
        setTimeout(() => setCopied((c) => (c === kind ? null : c)), 2000);
      },
      () => {},
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm px-4" onClick={onClose}>
      <div className="card shadow-card-lg rounded-2xl p-6 w-full max-w-md" onClick={(e) => e.stopPropagation()}>
        {!isPublic ? (
          <>
            <h2 className="font-display text-xl text-ink mb-1">Share this report</h2>
            <p className="text-sm text-ink-2 mb-5">
              Publishing makes this report and its charts public and indexable by search engines. Your uploaded file is
              never shared. You can make it private again anytime.
            </p>
            <button type="button" onClick={publish} disabled={busy} className="btn btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed">
              {busy ? (
                <span className="inline-flex items-center gap-2">
                  <span className="h-3.5 w-3.5 rounded-full border-2 border-on-accent/40 border-t-on-accent animate-spin" />
                  Publishing…
                </span>
              ) : (
                'Publish'
              )}
            </button>
            {error && <p className="mt-4 text-sm text-ember">{error}</p>}
            <button type="button" onClick={onClose} disabled={busy} className="mt-4 w-full text-sm text-ink-2 hover:text-ink disabled:opacity-50">
              Cancel
            </button>
          </>
        ) : (
          <>
            <h2 className="font-display text-xl text-ink mb-1">Report is public</h2>
            <p className="text-sm text-ink-2 mb-4">Anyone with the link can view this report.</p>

            <label className="block text-sm font-medium text-ink mb-1.5">Public link</label>
            <div className="flex gap-2 mb-4">
              <input
                type="text"
                readOnly
                value={publicUrl}
                onFocus={(e) => e.currentTarget.select()}
                className="flex-1 min-w-0 bg-surface-2 border border-line-2 rounded-lg px-3 py-2 text-sm text-ink outline-none focus:border-accent"
              />
              <button type="button" onClick={() => copy('link', publicUrl)} className="btn btn-ghost shrink-0">
                {copied === 'link' ? 'Copied' : 'Copy'}
              </button>
            </div>

            <label className="block text-sm font-medium text-ink mb-1.5">Embed</label>
            <div className="space-y-2 mb-4">
              <textarea
                readOnly
                rows={3}
                value={embedSnippet}
                onFocus={(e) => e.currentTarget.select()}
                className="w-full bg-surface-2 border border-line-2 rounded-lg px-3 py-2 text-sm text-ink outline-none focus:border-accent resize-none font-mono"
              />
              <button type="button" onClick={() => copy('embed', embedSnippet)} className="btn btn-ghost w-full">
                {copied === 'embed' ? 'Copied' : 'Copy embed code'}
              </button>
            </div>

            {error && <p className="mb-4 text-sm text-ember">{error}</p>}

            <div className="flex items-center justify-between gap-3">
              <button type="button" onClick={unpublish} disabled={busy} className="btn btn-ghost disabled:opacity-50 disabled:cursor-not-allowed">
                {busy ? (
                  <span className="inline-flex items-center gap-2">
                    <span className="h-3.5 w-3.5 rounded-full border-2 border-line-2 border-t-accent animate-spin" />
                    Working…
                  </span>
                ) : (
                  'Make private'
                )}
              </button>
              <button type="button" onClick={onClose} disabled={busy} className="text-sm text-ink-2 hover:text-ink disabled:opacity-50">
                Done
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
