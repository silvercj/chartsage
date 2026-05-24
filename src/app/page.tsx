'use client';

import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useRouter } from 'next/navigation';
import Papa from 'papaparse';

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
  const router = useRouter();

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted.length === 0) return;
    const f = accepted[0];
    if (f.size > 10 * 1024 * 1024) {
      setError('File must be under 10MB.');
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
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/generate-report`, {
        method: 'POST',
        body: fd,
      });
      let body: any = null;
      try { body = await res.json(); } catch {}
      if (res.status === 503 && body?.detail?.status === 'busy') {
        setError(body?.detail?.message ?? 'Service busy. Please retry in 30 seconds.');
        setIsProcessing(false);
        return;
      }
      if (!res.ok) {
        const detail = body?.detail;
        throw new Error(typeof detail === 'string' ? detail : 'Failed to generate report.');
      }
      setStep(2);
      router.push(`/report/${body.session_id}`);
    } catch (e: any) {
      setError(e.message || 'Generation failed.');
      setIsProcessing(false);
    }
  }

  return (
    <div className="container mx-auto px-4 py-10 max-w-4xl">
      <h1 className="text-4xl font-bold text-center text-gray-900 mb-2">Turn data into insight</h1>
      <p className="text-center text-lg text-gray-600 mb-10">
        Drop a CSV or Excel file. Get a narrated report with charts in under 10 seconds.
      </p>

      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-10 text-center transition-colors ${
          isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
        }`}
      >
        <input {...getInputProps()} />
        <p className="text-gray-700 mb-1">Drag and drop, or click to select.</p>
        <p className="text-sm text-gray-500">.csv or .xlsx, up to 10MB.</p>
      </div>

      {error && (
        <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-lg">{error}</div>
      )}

      {file && !isProcessing && (
        <div className="mt-6 p-4 bg-gray-50 rounded-lg flex justify-between items-center">
          <div>
            <p className="font-medium">{file.name}</p>
            <p className="text-sm text-gray-500">{(file.size / 1024).toFixed(0)} KB</p>
          </div>
          <button
            onClick={generate}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Generate report
          </button>
        </div>
      )}

      {isProcessing && (
        <div className="mt-6 p-6 bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="flex items-center justify-center mb-4">
            <div className="animate-spin rounded-full h-8 w-8 border-4 border-blue-500 border-t-transparent" />
          </div>
          <p className="text-center font-medium text-gray-900">{STEPS[step]}</p>
        </div>
      )}

      {preview && (
        <div className="mt-8 overflow-x-auto bg-white rounded-lg shadow-sm border border-gray-200">
          <h3 className="px-4 py-3 font-semibold text-gray-900 border-b border-gray-200">Preview</h3>
          <table className="min-w-full text-sm">
            <thead className="bg-gray-50">
              <tr>{preview.columns.map((c) => <th key={c} className="px-4 py-2 text-left">{c}</th>)}</tr>
            </thead>
            <tbody>
              {preview.data.map((row, i) => (
                <tr key={i} className="border-t border-gray-100">
                  {preview.columns.map((c) => (
                    <td key={c} className="px-4 py-2 text-gray-700">{row[c]?.toString() ?? ''}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
