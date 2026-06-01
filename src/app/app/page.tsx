'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useRouter } from 'next/navigation';
import Papa from 'papaparse';
import { apiFetch } from '../lib/api';
import OutOfCreditsModal from '../components/OutOfCreditsModal';
import { useCredits } from '../lib/useCredits';
import { REPORT_COST } from '../lib/credits';

interface DataPreview {
  columns: string[];
  data: Record<string, any>[];
}

const STEPS = ['Reading file', 'Analyzing with AI', 'Writing report', 'Done'];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<DataPreview | null>(null);
  const [step, setStep] = useState(0);
  const [showOutOfCredits, setShowOutOfCredits] = useState(false);
  const { balance, refetch } = useCredits();
  const router = useRouter();

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted.length === 0) return;
    const f = accepted[0];
    if (f.size > 10 * 1024 * 1024) {
      setError('File must be under 10MB.');
      return;
    }
    if (!/\.(csv|xlsx)$/i.test(f.name)) {
      setError('Please upload a .csv or .xlsx file.');
      return;
    }
    setFile(f);
    setError(null);
    setPreview(null);

    if (f.name.toLowerCase().endsWith('.csv')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        const parsed = Papa.parse(text, { header: true, skipEmptyLines: true });
        if (parsed.errors.length === 0) {
          const cols = (parsed.meta.fields || []).map((c) => c.toLowerCase());
          const rows = parsed.data.slice(0, 10).map((r: any) => {
            const out: Record<string, any> = {};
            for (const k of Object.keys(r)) out[k.toLowerCase()] = r[k];
            return out;
          });
          setPreview({ columns: cols, data: rows });
        }
      };
      reader.readAsText(f);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
    maxFiles: 1,
  });

  async function generate() {
    if (!file) return;
    setIsProcessing(true);
    setError(null);
    setStep(0);
    try {
      setStep(1);
      const fd = new FormData();
      fd.append('file', file);
      const res = await apiFetch('/generate-report', { method: 'POST', body: fd });
      let body: any = null;
      try { body = await res.json(); } catch {}

      if (res.status === 403 && body?.detail?.code === 'ANON_LIMIT_REACHED') {
        router.push('/anon-limit');
        return;
      }
      if (res.status === 402 && body?.detail?.code === 'OUT_OF_CREDITS') {
        setShowOutOfCredits(true);
        setIsProcessing(false);
        return;
      }
      if (res.status === 503 && body?.detail?.code === 'BUSY') {
        setError(body?.detail?.message ?? 'Service busy. Please retry in 30 seconds.');
        setIsProcessing(false);
        return;
      }
      if (!res.ok) {
        const detail = body?.detail;
        throw new Error(typeof detail === 'string' ? detail : detail?.message ?? 'Failed to generate report.');
      }
      setStep(2);
      refetch();
      router.push(`/report/${body.session_id}`);
    } catch (e: any) {
      setError(e.message || 'Generation failed.');
      setIsProcessing(false);
    }
  }

  return (
    <div className="min-h-screen bg-canvas">
      <div className="max-w-3xl mx-auto px-6 py-16 reveal">
        <header className="mb-12">
          <p className="eyebrow mb-3">CSV → INSIGHT IN ~30 SECONDS</p>
          <h1 className="font-display text-4xl sm:text-5xl font-normal tracking-tight text-ink leading-[1.05] mb-4">
            Drop your data. Read the <em className="italic font-medium text-accent">story</em> inside it.
          </h1>
          <p className="text-ink-2 mt-4 max-w-prose leading-relaxed">
            Drop a CSV or Excel file. We profile your data, use AI to pick the charts that
            tell the most useful story, and return a narrated report in seconds.
          </p>
        </header>

        <div
          {...getRootProps()}
          className={`card border-dashed rounded-2xl p-10 text-center cursor-pointer transition-colors ${
            isDragActive ? 'border-accent' : 'border-line-2 hover:border-accent'
          }`}
        >
          <input {...getInputProps()} />
          <span className="inline-flex w-12 h-12 mx-auto mb-3 items-center justify-center rounded-xl bg-surface-2 border border-line-2">
            <svg
              className="w-6 h-6 text-ink-2"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
            </svg>
          </span>
          <p className="text-ink font-medium mb-1">Drop a file, or click to choose.</p>
          <p className="font-mono text-xs text-ink-3">.csv or .xlsx · up to 10 MB</p>
        </div>

        {error && (
          <div className="mt-4 p-4 card border-ember/40 text-ember text-sm rounded-xl">
            {error}
          </div>
        )}

        {file && !isProcessing && (
          <div className="mt-6 card shadow-card p-5 rounded-2xl flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
            <div>
              <p className="font-medium text-ink">{file.name}</p>
              <p className="font-mono text-xs text-ink-3">{(file.size / 1024).toFixed(0)} KB</p>
            </div>
            <button
              onClick={generate}
              className="btn btn-primary w-full sm:w-auto"
            >
              {balance !== null ? `Generate report · ${REPORT_COST}` : 'Generate report →'}
            </button>
          </div>
        )}

        {isProcessing && (
          <div className="mt-6 card shadow-card p-6 rounded-2xl">
            <div className="flex items-center gap-4">
              <div className="animate-spin rounded-full h-6 w-6 border-2 border-surface-2 border-t-accent flex-shrink-0" />
              <div className="flex-1">
                <p className="font-medium text-ink text-sm">{STEPS[step]}</p>
                <div className="mt-3 flex gap-2">
                  {STEPS.map((label, i) => (
                    <div key={label} className="flex-1">
                      <div
                        className={`h-1 rounded-full ${i <= step ? 'bg-accent' : 'bg-surface-2'}`}
                      />
                      <p
                        className={`mt-1.5 font-mono text-[10px] uppercase tracking-wide ${
                          i === step ? 'text-accent' : i < step ? 'text-ink-2' : 'text-ink-3'
                        }`}
                      >
                        {label}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {preview && (
          <div className="mt-8 overflow-x-auto card shadow-card rounded-2xl">
            <div className="px-5 py-3 border-b border-line flex items-baseline justify-between">
              <h3 className="font-display text-base text-ink">Preview</h3>
              <span className="font-mono text-xs text-ink-3">{preview.data.length} rows · {preview.columns.length} columns</span>
            </div>
            <table className="min-w-full text-sm">
              <thead>
                <tr className="bg-surface-2">
                  {preview.columns.map((c) => (
                    <th key={c} className="px-4 py-2.5 text-left font-mono text-xs uppercase tracking-wide text-ink-3">{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.data.map((row, i) => (
                  <tr key={i} className="border-t border-line">
                    {preview.columns.map((c) => (
                      <td key={c} className="px-4 py-2 text-ink-2">{row[c]?.toString() ?? ''}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
        <OutOfCreditsModal open={showOutOfCredits} onClose={() => setShowOutOfCredits(false)} />
    </div>
  );
}
