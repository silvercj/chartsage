'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useRouter } from 'next/navigation';
import Papa from 'papaparse';
import { apiFetch } from './lib/api';
import OutOfCreditsModal from './components/OutOfCreditsModal';
import { useCredits } from './lib/useCredits';
import { REPORT_COST } from './lib/credits';

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
    <div className="min-h-screen bg-stone-50">
      <div className="max-w-3xl mx-auto px-4 sm:px-6 py-16">
        <header className="mb-12">
          <p className="text-xs uppercase tracking-widest text-stone-400 mb-3">ChartSage</p>
          <h1 className="text-4xl md:text-5xl font-semibold tracking-tight text-stone-900 mb-3">
            Turn data into insight.
          </h1>
          <p className="text-stone-600 text-[15px] leading-relaxed max-w-2xl">
            Drop a CSV or Excel file. We profile your data, ask Claude to pick the 5–7 charts that
            tell the most useful story, and return a narrated report in under 10 seconds.
          </p>
        </header>

        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all cursor-pointer ${
            isDragActive
              ? 'border-teal-500 bg-teal-50/50'
              : 'border-stone-300 hover:border-stone-400 hover:bg-white'
          }`}
        >
          <input {...getInputProps()} />
          <svg
            className="w-10 h-10 mx-auto mb-3 text-stone-400"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
          </svg>
          <p className="text-stone-700 font-medium mb-1">Drop a file, or click to choose.</p>
          <p className="text-sm text-stone-500">.csv or .xlsx · up to 10 MB</p>
        </div>

        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-100 text-red-700 text-sm rounded-xl">
            {error}
          </div>
        )}

        {file && !isProcessing && (
          <div className="mt-6 p-5 bg-white border border-stone-200 rounded-2xl flex justify-between items-center">
            <div>
              <p className="font-medium text-stone-900">{file.name}</p>
              <p className="text-sm text-stone-500">{(file.size / 1024).toFixed(0)} KB</p>
            </div>
            <button
              onClick={generate}
              className="px-5 py-2.5 bg-stone-900 text-white text-sm font-medium rounded-lg hover:bg-stone-800 transition-colors"
            >
              {balance !== null ? `Generate report · ${REPORT_COST}` : 'Generate report →'}
            </button>
          </div>
        )}

        {isProcessing && (
          <div className="mt-6 p-6 bg-white border border-stone-200 rounded-2xl">
            <div className="flex items-center gap-4">
              <div className="animate-spin rounded-full h-6 w-6 border-2 border-stone-200 border-t-stone-900 flex-shrink-0" />
              <div className="flex-1">
                <p className="font-medium text-stone-900 text-sm">{STEPS[step]}</p>
                <div className="mt-2 flex gap-1.5">
                  {STEPS.map((_, i) => (
                    <div
                      key={i}
                      className={`h-1 flex-1 rounded-full ${i <= step ? 'bg-stone-900' : 'bg-stone-200'}`}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {preview && (
          <div className="mt-8 overflow-x-auto bg-white border border-stone-200 rounded-2xl">
            <div className="px-5 py-3 border-b border-stone-100 flex items-baseline justify-between">
              <h3 className="font-semibold text-stone-900 text-sm">Preview</h3>
              <span className="text-xs text-stone-400">{preview.data.length} rows · {preview.columns.length} columns</span>
            </div>
            <table className="min-w-full text-sm">
              <thead>
                <tr className="bg-stone-50">
                  {preview.columns.map((c) => (
                    <th key={c} className="px-4 py-2.5 text-left font-medium text-stone-600 text-xs uppercase tracking-wide">{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.data.map((row, i) => (
                  <tr key={i} className="border-t border-stone-100">
                    {preview.columns.map((c) => (
                      <td key={c} className="px-4 py-2 text-stone-700">{row[c]?.toString() ?? ''}</td>
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
