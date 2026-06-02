'use client';

import { useState, useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { useRouter } from 'next/navigation';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { apiFetch } from '../lib/api';
import OutOfCreditsModal from '../components/OutOfCreditsModal';
import { useCredits } from '../lib/useCredits';
import { REPORT_COST, DEEP_ANALYSIS_COST } from '../lib/credits';

const STEPS = ['Reading file', 'Analyzing with AI', 'Writing report', 'Done'];
const PREVIEW_ROWS = 10;

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState(0);
  const [showOutOfCredits, setShowOutOfCredits] = useState(false);
  const [focus, setFocus] = useState('');
  const [deep, setDeep] = useState(false);
  const [isLargeFile, setIsLargeFile] = useState(false);

  // Parse state — the selected sheet's rows/columns and the user's column selection.
  const [sheetNames, setSheetNames] = useState<string[]>([]);
  const [sheet, setSheet] = useState<string>('');
  const [columns, setColumns] = useState<string[]>([]);
  const [excluded, setExcluded] = useState<Set<string>>(new Set());
  const [rows, setRows] = useState<Record<string, any>[]>([]);

  // Keep the parsed workbook so re-selecting a sheet doesn't re-read the file.
  const wbRef = useRef<XLSX.WorkBook | null>(null);

  const { balance, refetch } = useCredits();
  const router = useRouter();

  const loadSheet = useCallback((wb: XLSX.WorkBook, name: string) => {
    const ws = wb.Sheets[name];
    const parsed: Record<string, any>[] = ws
      ? XLSX.utils.sheet_to_json(ws, { defval: '' })
      : [];
    const cols = parsed.length > 0 ? Object.keys(parsed[0]) : [];
    setSheet(name);
    setRows(parsed);
    setColumns(cols);
    setExcluded(new Set());
  }, []);

  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length === 0) return;
      const f = accepted[0];
      if (f.size > 50 * 1024 * 1024) {
        setError('File must be under 50MB.');
        return;
      }
      if (!/\.(csv|xlsx)$/i.test(f.name)) {
        setError('Please upload a .csv or .xlsx file.');
        return;
      }
      setFile(f);
      setError(null);
      setIsLargeFile(f.size > 10 * 1024 * 1024);
      // Reset any prior parse state.
      wbRef.current = null;
      setSheetNames([]);
      setSheet('');
      setColumns([]);
      setExcluded(new Set());
      setRows([]);

      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const buf = e.target?.result as ArrayBuffer;
          const wb = XLSX.read(buf, { type: 'array' });
          wbRef.current = wb;
          setSheetNames(wb.SheetNames);
          const first = wb.SheetNames[0];
          if (first) loadSheet(wb, first);
        } catch {
          setError('Could not read that file. Please check it is a valid .csv or .xlsx.');
        }
      };
      reader.onerror = () => setError('Could not read that file.');
      reader.readAsArrayBuffer(f);
    },
    [loadSheet]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
    maxFiles: 1,
  });

  function toggleColumn(col: string) {
    setExcluded((prev) => {
      const next = new Set(prev);
      if (next.has(col)) next.delete(col);
      else next.add(col);
      return next;
    });
  }

  const keep = columns.filter((c) => !excluded.has(c));
  const hasData = columns.length > 0;
  const allExcluded = hasData && keep.length === 0;

  async function generate() {
    if (!file) return;
    if (allExcluded) {
      setError('Select at least one column to include.');
      return;
    }
    setIsProcessing(true);
    setError(null);
    setStep(0);
    try {
      setStep(1);
      // Emit the chosen sheet (minus dropped columns) as a CSV and send that.
      const csv = Papa.unparse(
        rows.map((r) => Object.fromEntries(keep.map((c) => [c, r[c]])))
      );
      const blob = new File(
        [csv],
        (file?.name?.replace(/\.(xlsx|csv)$/i, '') || 'data') + '.csv',
        { type: 'text/csv' }
      );
      const fd = new FormData();
      fd.append('file', blob);
      if (focus.trim()) fd.append('custom_prompt', focus.trim());
      fd.append('deep', deep ? 'true' : 'false');
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

  const previewRows = rows.slice(0, PREVIEW_ROWS);

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
          <p className="font-mono text-xs text-ink-3">.csv or .xlsx · up to 50 MB</p>
        </div>

        {isLargeFile && !isProcessing && (
          <p className="mt-4 font-mono text-xs text-ink-2">
            Large file — we&apos;ll analyze a representative sample.
          </p>
        )}

        {error && (
          <div className="mt-4 p-4 card border-ember/40 text-ember text-sm rounded-xl">
            {error}
          </div>
        )}

        {file && !isProcessing && (
          <div className="mt-6 card shadow-card p-5 rounded-2xl">
            <div>
              <label htmlFor="focus" className="block font-mono text-[10px] uppercase tracking-wide text-ink-3 mb-1.5">
                Anything specific to focus on? (optional)
              </label>
              <textarea
                id="focus"
                value={focus}
                onChange={(e) => setFocus(e.target.value)}
                maxLength={280}
                rows={2}
                placeholder="e.g. margins by region, or what's driving the recent drop"
                className="w-full resize-none bg-surface-2 border border-line rounded-lg px-3 py-2 text-sm text-ink placeholder:text-ink-3 focus:outline-none focus:border-accent"
              />
              <p className="mt-1 text-right font-mono text-[10px] text-ink-3">{focus.length}/280</p>
            </div>
            <div className="mt-4 pt-4 border-t border-line">
              <label htmlFor="deep" className="flex items-start gap-3 cursor-pointer select-none">
                <input
                  id="deep"
                  type="checkbox"
                  checked={deep}
                  onChange={(e) => setDeep(e.target.checked)}
                  className="mt-0.5 accent-accent cursor-pointer"
                />
                <span>
                  <span className="block text-sm font-medium text-ink">Deep analysis · {DEEP_ANALYSIS_COST}</span>
                  <span className="block text-xs text-ink-3 mt-0.5">
                    Runs multiple passes for a richer report. Costs {DEEP_ANALYSIS_COST} credits.
                  </span>
                </span>
              </label>
            </div>
            <div className="mt-4 flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4">
              <div>
                <p className="font-medium text-ink">{file.name}</p>
                <p className="font-mono text-xs text-ink-3">{(file.size / 1024).toFixed(0)} KB</p>
              </div>
              <button
                onClick={generate}
                disabled={allExcluded}
                className="btn btn-primary w-full sm:w-auto disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {deep
                  ? `Generate report · ${DEEP_ANALYSIS_COST}`
                  : balance !== null
                    ? `Generate report · ${REPORT_COST}`
                    : 'Generate report →'}
              </button>
            </div>
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

        {sheetNames.length > 1 && !isProcessing && (
          <div className="mt-6 card shadow-card p-5 rounded-2xl flex flex-col sm:flex-row sm:items-center gap-3">
            <label htmlFor="sheet-picker" className="font-mono text-xs uppercase tracking-wide text-ink-3">
              Sheet
            </label>
            <select
              id="sheet-picker"
              value={sheet}
              onChange={(e) => {
                if (wbRef.current) loadSheet(wbRef.current, e.target.value);
              }}
              className="bg-surface-2 border border-line rounded-lg px-3 py-2 text-sm text-ink focus:outline-none focus:border-accent"
            >
              {sheetNames.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </div>
        )}

        {hasData && !isProcessing && (
          <div className="mt-8 overflow-x-auto card shadow-card rounded-2xl">
            <div className="px-5 py-3 border-b border-line flex flex-wrap items-baseline justify-between gap-2">
              <div className="flex items-baseline gap-3">
                <h3 className="font-display text-base text-ink">Preview</h3>
                <span className="font-mono text-xs text-ink-3">
                  {rows.length} rows · {keep.length}/{columns.length} columns
                </span>
              </div>
              <span className="font-mono text-[10px] uppercase tracking-wide text-ink-3">
                Uncheck a column to drop it
              </span>
            </div>
            <table className="min-w-full text-sm">
              <thead>
                <tr className="bg-surface-2">
                  {columns.map((c) => {
                    const included = !excluded.has(c);
                    return (
                      <th
                        key={c}
                        className={`px-4 py-2.5 text-left font-mono text-xs uppercase tracking-wide ${
                          included ? 'text-ink-3' : 'text-ink-3/40 line-through'
                        }`}
                      >
                        <label className="flex items-center gap-2 cursor-pointer select-none">
                          <input
                            type="checkbox"
                            checked={included}
                            onChange={() => toggleColumn(c)}
                            className="accent-accent cursor-pointer"
                            aria-label={`Include column ${c}`}
                          />
                          <span>{c}</span>
                        </label>
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {previewRows.map((row, i) => (
                  <tr key={i} className="border-t border-line">
                    {columns.map((c) => {
                      const included = !excluded.has(c);
                      return (
                        <td
                          key={c}
                          className={`px-4 py-2 ${included ? 'text-ink-2' : 'text-ink-3/40'}`}
                        >
                          {row[c]?.toString() ?? ''}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
            {allExcluded && (
              <div className="px-5 py-3 border-t border-line text-ember text-xs">
                Select at least one column to generate a report.
              </div>
            )}
          </div>
        )}
      </div>
        <OutOfCreditsModal open={showOutOfCredits} onClose={() => setShowOutOfCredits(false)} />
    </div>
  );
}
