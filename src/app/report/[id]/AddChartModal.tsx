'use client';
import { useState } from 'react';
import { apiFetch } from '../../lib/api';
import { posthog } from '../../lib/posthog';
import { ADD_CHART_COST } from '../../lib/credits';
import type { Report } from './useReportLayout';

interface Props {
  open: boolean;
  onClose: () => void;
  sessionId: string;
  onReportUpdated: (next: Report) => void;
  /** Called on a 402 with the response code so the Toolbar can show the right modal. */
  onOutOfCredits: (code: 'OUT_OF_CREDITS' | 'UPGRADE_REQUIRED') => void;
}

type Tab = 'type' | 'describe';

// Friendly label → backend tool name.
const CHART_TYPES: { label: string; tool: string }[] = [
  { label: 'Frequency bar', tool: 'frequency_bar_chart' },
  { label: 'Aggregation bar', tool: 'aggregation_bar_chart' },
  { label: 'Histogram', tool: 'histogram_chart' },
  { label: 'Scatter', tool: 'scatter_chart' },
  { label: 'Trend line', tool: 'line_chart' },
  { label: 'Pie', tool: 'pie_chart' },
  { label: 'Box plot', tool: 'box_plot' },
  { label: 'Heatmap', tool: 'heatmap_chart' },
  { label: 'Treemap', tool: 'treemap_chart' },
  { label: 'Grouped bar', tool: 'grouped_bar_chart' },
  { label: 'Dual-axis', tool: 'dual_axis_chart' },
];

export default function AddChartModal({ open, onClose, sessionId, onReportUpdated, onOutOfCredits }: Props) {
  const [tab, setTab] = useState<Tab>('type');
  const [prompt, setPrompt] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  function close() {
    if (submitting) return;
    setError(null);
    onClose();
  }

  async function submit(payload: { mode: 'type'; chart_type: string } | { mode: 'describe'; prompt: string }) {
    setSubmitting(true);
    setError(null);
    posthog.capture?.('add_chart_clicked', { reportId: sessionId, mode: payload.mode });
    try {
      const res = await apiFetch(`/report/${sessionId}/add-chart`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.status === 503) {
        setError('The AI is busy. Try again in 30 seconds.');
        return;
      }
      if (res.status === 402) {
        let body: any = null;
        try { body = await res.json(); } catch {}
        const code = body?.detail?.code === 'OUT_OF_CREDITS' ? 'OUT_OF_CREDITS' : 'UPGRADE_REQUIRED';
        onClose();
        onOutOfCredits(code);
        return;
      }
      if (res.status === 422) {
        setError("Couldn't build that chart — try another type or description.");
        return;
      }
      if (!res.ok) throw new Error(`Failed (${res.status})`);
      const updated: Report = await res.json();
      onReportUpdated(updated);
      setPrompt('');
      onClose();
    } catch (e: any) {
      setError(e.message || 'Failed to add chart.');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm px-4" onClick={close}>
      <div className="card shadow-card-lg rounded-2xl p-6 w-full max-w-md" onClick={(e) => e.stopPropagation()}>
        <h2 className="font-display text-xl text-ink mb-1">Add a chart</h2>
        <p className="text-sm text-ink-2 mb-4">
          Pick a chart type or describe what you want to see. <span className="text-ink-3">Add chart · {ADD_CHART_COST}</span>
        </p>

        <div className="flex gap-1 p-1 rounded-lg bg-surface-2 mb-4">
          <button
            type="button"
            disabled={submitting}
            onClick={() => { setTab('type'); setError(null); }}
            className={`flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
              tab === 'type' ? 'bg-surface text-ink shadow-card' : 'text-ink-2 hover:text-ink'
            }`}
          >
            Pick a type
          </button>
          <button
            type="button"
            disabled={submitting}
            onClick={() => { setTab('describe'); setError(null); }}
            className={`flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
              tab === 'describe' ? 'bg-surface text-ink shadow-card' : 'text-ink-2 hover:text-ink'
            }`}
          >
            Describe it
          </button>
        </div>

        {tab === 'type' ? (
          <div className="grid grid-cols-2 gap-2">
            {CHART_TYPES.map((c) => (
              <button
                key={c.tool}
                type="button"
                disabled={submitting}
                onClick={() => submit({ mode: 'type', chart_type: c.tool })}
                className="rounded-lg border border-line-2 px-3 py-2.5 text-sm text-ink text-left hover:bg-surface-2 hover:border-accent disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {c.label}
              </button>
            ))}
          </div>
        ) : (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              const p = prompt.trim();
              if (!p) return;
              submit({ mode: 'describe', prompt: p });
            }}
            className="space-y-3"
          >
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={submitting}
              rows={3}
              maxLength={280}
              placeholder="e.g. revenue by channel over time"
              className="w-full bg-surface-2 border border-line-2 rounded-lg px-4 py-3 text-ink placeholder:text-ink-3 focus:border-accent outline-none resize-none disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={submitting || !prompt.trim()}
              className="btn btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {submitting ? (
                <span className="inline-flex items-center gap-2">
                  <span className="h-3.5 w-3.5 rounded-full border-2 border-on-accent/40 border-t-on-accent animate-spin" />
                  Adding…
                </span>
              ) : (
                `Add chart · ${ADD_CHART_COST}`
              )}
            </button>
          </form>
        )}

        {submitting && tab === 'type' && (
          <p className="mt-4 inline-flex items-center gap-2 text-sm text-ink-2">
            <span className="h-3.5 w-3.5 rounded-full border-2 border-line-2 border-t-accent animate-spin" />
            Adding chart…
          </p>
        )}
        {error && <p className="mt-4 text-sm text-ember">{error}</p>}

        <button type="button" onClick={close} disabled={submitting} className="mt-4 w-full text-sm text-ink-2 hover:text-ink disabled:opacity-50">
          Cancel
        </button>
      </div>
    </div>
  );
}
